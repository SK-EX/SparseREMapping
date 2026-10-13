#!/usr/bin/env python3
"""
Multi-View Photometric Depth Estimation
=========================================
自监督训练: 不需要 GT 深度，用多视角光度一致性做监督

训练时:
  View1 RGB → UNet → mask1 + depth1
  depth1 + pose(1→2) → warp View2 → 光度误差(mask内)

推理时:
  单张 RGB → depth (mm)

用法:
  python3 train/train.py --epochs 100
"""

import os, sys, random, json, argparse, logging, time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-5s | %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

PILE_DIAMETER_MM = 90.0
IMG_SIZE = 512

# ======================================================================
# Model: U-Net with dual head (mask + depth)
# ======================================================================
def _load_resnet34():
    import torchvision.models as models
    resnet = models.resnet34(weights=None)
    local = PROJECT_ROOT / 'resnet34-b627a593.pth'
    if local.exists():
        resnet.load_state_dict(torch.load(local, map_location='cpu', weights_only=True))
    return resnet

class UNetDepth(nn.Module):
    """UNet + ResNet34 → mask + depth (mm)"""

    def __init__(self, pretrained_mask_ckpt=None):
        super().__init__()
        resnet = _load_resnet34()

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        self.up4 = self._up(512 + 256, 256)
        self.up3 = self._up(256 + 128, 128)
        self.up2 = self._up(128 + 64, 64)
        self.up1 = self._up(64 + 64, 64)

        self.shared = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True))

        # Mask head (from pretrained ScourUNet, frozen)
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 1, 1))

        # Load pretrained mask weights
        if pretrained_mask_ckpt and Path(pretrained_mask_ckpt).exists():
            logger.info(f"Loading pretrained mask from {pretrained_mask_ckpt}")
            ckpt = torch.load(pretrained_mask_ckpt, map_location='cpu', weights_only=True)
            # Load only matching keys
            own_state = self.state_dict()
            for name, param in ckpt.items():
                if name in own_state and 'depth' not in name:
                    own_state[name].copy_(param)
            # Freeze mask-related weights
            for name, param in self.named_parameters():
                if 'depth' not in name and 'depth_scale' not in name:
                    param.requires_grad = False
            logger.info("Mask encoder frozen, only depth head trainable")

        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 1, 1))

        self.depth_scale = nn.Parameter(torch.tensor(PILE_DIAMETER_MM), requires_grad=True)

    def _up(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d4 = self.up4(torch.cat([e5, F.interpolate(e4, e5.shape[2:], mode='bilinear', align_corners=False)], 1))
        d3 = self.up3(torch.cat([d4, F.interpolate(e3, d4.shape[2:], mode='bilinear', align_corners=False)], 1))
        d2 = self.up2(torch.cat([d3, F.interpolate(e2, d3.shape[2:], mode='bilinear', align_corners=False)], 1))
        d1 = self.up1(torch.cat([d2, F.interpolate(e1, d2.shape[2:], mode='bilinear', align_corners=False)], 1))

        feat = self.shared(d1)
        if feat.shape[2:] != x.shape[2:]:
            feat = F.interpolate(feat, x.shape[2:], mode='bilinear', align_corners=False)

        mask = torch.sigmoid(self.mask_head(feat))
        depth_raw = self.depth_head(feat)

        # Convert to mm: negative = scour, positive = bed
        depth = torch.tanh(depth_raw) * self.depth_scale.abs()

        return mask, depth


# ======================================================================
# Differentiable Warping
# ======================================================================
def warp_view(src_img, ref_depth, K_ref, K_src, R_rel, t_rel):
    """
    用 ref_depth + 相对位姿将 src_img warp 到 ref 视角

    Args:
        src_img: (B, 3, H, W) source image
        ref_depth: (B, 1, H, W) predicted depth of ref view
        K_ref, K_src: (B, 3, 3) intrinsics
        R_rel, t_rel: relative rotation (3,3) and translation (3,1)

    Returns:
        warped: (B, 3, H, W) warped source image
        valid_mask: (B, 1, H, W) valid pixels
    """
    B, _, H, W = ref_depth.shape
    device = ref_depth.device

    # Pixel grid
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32),
                            indexing='ij')
    ones = torch.ones_like(xx)
    pix_homo = torch.stack([xx, yy, ones], dim=-1)  # (H, W, 3)
    pix_homo = pix_homo.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)

    # Back-project ref pixels to 3D (camera coordinates)
    K_ref_inv = torch.inverse(K_ref)  # (B, 3, 3)
    pts_ref = (K_ref_inv @ pix_homo.reshape(B, -1, 3).transpose(1, 2))  # (B, 3, H*W)
    pts_ref = pts_ref * ref_depth.reshape(B, 1, -1)  # (B, 3, H*W)

    # Transform to src camera coordinates
    pts_src = R_rel @ pts_ref + t_rel  # (B, 3, H*W)

    # Project to src image
    pts_src_2d = K_src @ pts_src  # (B, 3, H*W)
    valid = pts_src_2d[:, 2:3] > 1e-6
    pts_src_2d = pts_src_2d[:, :2] / (pts_src_2d[:, 2:3] + 1e-6)

    # Normalize to [-1, 1]
    grid_x = pts_src_2d[:, 0:1] / (W - 1) * 2 - 1
    grid_y = pts_src_2d[:, 1:2] / (H - 1) * 2 - 1
    grid = torch.cat([grid_x, grid_y], dim=1).reshape(B, 2, H, W).permute(0, 2, 3, 1)

    # Sample
    warped = F.grid_sample(src_img, grid.clamp(-1, 1),
                           mode='bilinear', padding_mode='zeros',
                           align_corners=True)
    valid_mask = valid.reshape(B, 1, H, W).float()

    return warped, valid_mask


# ======================================================================
# Dataset
# ======================================================================
class MultiViewDepthDataset(Dataset):
    """Load image pairs with COLMAP poses"""

    def __init__(self, split='train', val_ratio=0.2, augment=True):
        import pycolmap
        self.augment = augment and split == 'train'

        sparse_dir = PROJECT_ROOT / 'result/scour_colmap/sparse/0'
        recon = pycolmap.Reconstruction(str(sparse_dir))

        # Build list of pairs
        self.pairs = []
        img_names = sorted([img.name for img in recon.images.values()])
        for i in range(len(img_names)):
            j = (i + 1) % len(img_names)
            self.pairs.append((img_names[i], img_names[j]))

        # Camera params
        self.cameras = {}
        self.img_dir = PROJECT_ROOT / 'data/random_145_0_25_pure_currents/images'
        for img_id, img in recon.images.items():
            cam = recon.camera(img.camera_id)
            K = cam.calibration_matrix().astype(np.float32)
            pose = np.array(img.cam_from_world.matrix()).astype(np.float32)
            self.cameras[img.name] = {
                'K': K, 'R': pose[:3, :3], 't': pose[:3, 3:4],
                'cam_from_world': pose,
            }

        # Split
        random.Random(42).shuffle(self.pairs)
        n = len(self.pairs)
        split_n = int(n * (1 - val_ratio))
        if split == 'train':
            self.pairs = self.pairs[:split_n]
        else:
            self.pairs = self.pairs[split_n:]

        logger.info(f"[{split}] {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def _load(self, name):
        img = cv2.imread(str(self.img_dir / name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img

    def __getitem__(self, idx):
        n1, n2 = self.pairs[idx]

        img1 = self._load(n1)
        img2 = self._load(n2)

        cam1 = self.cameras[n1]
        cam2 = self.cameras[n2]

        # Random flip
        if self.augment and random.random() < 0.5:
            img1 = np.fliplr(img1).copy()
            img2 = np.fliplr(img2).copy()

        img1_t = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0

        # Relative pose (cam_from_world is 3x4, make 4x4)
        P1 = torch.eye(4)
        P1[:3] = torch.from_numpy(cam1['cam_from_world'])
        P2 = torch.eye(4)
        P2[:3] = torch.from_numpy(cam2['cam_from_world'])
        P_rel = P2 @ torch.inverse(P1)
        R_rel = P_rel[:3, :3]
        t_rel = P_rel[:3, 3:4]

        K1 = torch.from_numpy(cam1['K'])
        K2 = torch.from_numpy(cam2['K'])

        return {
            'img1': img1_t, 'img2': img2_t,
            'K1': K1, 'K2': K2,
            'R_rel': R_rel, 't_rel': t_rel,
            'name': n1
        }


# ======================================================================
# Losses
# ======================================================================
def photometric_loss(img1, warped_img2, valid_mask, mask):
    """L1 + SSIM 光度损失 (只mask内)"""
    region = (mask > 0.3) & (valid_mask > 0.5)

    if region.sum() < 100:
        return torch.tensor(0.0, device=img1.device, requires_grad=True)

    l1 = F.l1_loss(img1[region.expand_as(img1)],
                   warped_img2[region.expand_as(img1)])

    # Simple SSIM approximation (3x3 box)
    def ssim(x, y):
        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)
        sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return ssim_map

    ssim_loss = (1 - ssim(img1, warped_img2).mean())

    return 0.85 * ssim_loss + 0.15 * l1


def smoothness_loss(depth, img):
    """边缘感知平滑损失"""
    dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    img_ds = F.interpolate(img, depth.shape[2:], mode='bilinear', align_corners=False)
    if img_ds.shape[1] > 1:
        img_ds = img_ds.mean(dim=1, keepdim=True)
    wx = torch.exp(-(img_ds[:, :, :, 1:] - img_ds[:, :, :, :-1]).abs() * 5)
    wy = torch.exp(-(img_ds[:, :, 1:, :] - img_ds[:, :, :-1, :]).abs() * 5)
    return (wx * dx.abs()).mean() + (wy * dy.abs()).mean()


def physics_loss(depth, mask):
    """冲刷坑物理约束: 坑深范围 [0, 2.5*D], 坑内为负值"""
    # scour depth
    scour_depth = -depth  # positive scour depth
    loss_max = F.relu(scour_depth - 2.5 * PILE_DIAMETER_MM).mean()
    loss_min = F.relu(10.0 - scour_depth).mean()

    # Masked: scour region should have negative depth
    scour_region = depth[mask > 0.3]
    if len(scour_region) > 10:
        loss_sign = F.relu(scour_region.mean())  # should be negative
    else:
        loss_sign = torch.tensor(0.0, device=depth.device)

    return loss_max + loss_min + loss_sign


# ======================================================================
# Trainer
# ======================================================================
class DepthTrainer:
    def __init__(self, epochs=100, batch_size=4, lr=1e-4):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = DataLoader(
            MultiViewDepthDataset('train', augment=True),
            batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_loader = DataLoader(
            MultiViewDepthDataset('val', augment=False),
            batch_size=batch_size, shuffle=False, num_workers=0)

        ckpt_path = str(PROJECT_ROOT / 'result/checkpoints/best_model.pth')
        if not Path(ckpt_path).exists():
            ckpt_path = None
        self.model = UNetDepth(pretrained_mask_ckpt=ckpt_path).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, epochs, eta_min=1e-6)
        self.scaler = GradScaler()

        os.makedirs(PROJECT_ROOT / 'result/checkpoints', exist_ok=True)

    def train_epoch(self):
        self.model.train()
        losses = {'photo': 0, 'smooth': 0, 'physics': 0}

        for batch in tqdm(self.train_loader, desc=f"Train", leave=False):
            img1 = batch['img1'].to(self.device)
            img2 = batch['img2'].to(self.device)
            K1 = batch['K1'].to(self.device)
            K2 = batch['K2'].to(self.device)
            R_rel = batch['R_rel'].to(self.device)
            t_rel = batch['t_rel'].to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                mask, depth = self.model(img1)
                warped, valid = warp_view(img2, depth.detach(), K1, K2, R_rel, t_rel)
                loss_photo = photometric_loss(img1, warped, valid, mask)
                loss_smooth = smoothness_loss(depth, img1)
                loss_phys = physics_loss(depth, mask)
                total = loss_photo + 0.01 * loss_smooth + 0.05 * loss_phys

            self.scaler.scale(total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses['photo'] += loss_photo.item()
            losses['smooth'] += loss_smooth.item()
            losses['physics'] += loss_phys.item()

        n = len(self.train_loader)
        return {k: v / n for k, v in losses.items()}

    def train(self):
        logger.info(f"Multi-View Depth | {self.epochs} epochs | device={self.device}")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            losses = self.train_epoch()
            self.scheduler.step()
            logger.info(
                f"E{epoch:3d} | photo={losses['photo']:.4f} "
                f"smooth={losses['smooth']:.4f} phys={losses['physics']:.4f} "
                f"| {time.time() - t0:.1f}s")

            if epoch % 20 == 0:
                torch.save(self.model.state_dict(),
                           PROJECT_ROOT / f'result/checkpoints/depth_epoch_{epoch}.pth')

        torch.save(self.model.state_dict(),
                   PROJECT_ROOT / 'result/checkpoints/depth_best.pth')
        logger.info("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    trainer = DepthTrainer(args.epochs, args.batch_size, args.lr)
    trainer.train()


if __name__ == '__main__':
    main()
