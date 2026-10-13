#!/usr/bin/env python3
"""
冲刷坑分割 — 有监督训练
======================
用 LabelMe 标注的 scour pits 多边形做 GT mask，
训练 UNet + ResNet34 分割网络。

用法:
  python3 train/train_supervised.py --epochs 100
"""

import os, sys, json, random, argparse, logging, time
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

IMG_SIZE = 512


# ============================================================
# 从 polygon 生成二值 GT mask
# ============================================================
def polygon_to_mask(polygon_pts, H, W):
    """将 polygon 点列表渲染为二值 mask"""
    mask = np.zeros((H, W), dtype=np.float32)
    if len(polygon_pts) < 3:
        return mask
    pts = np.array(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1.0)
    return mask


# ============================================================
# Dataset
# ============================================================
class ScourSegDataset(Dataset):
    """加载 RGB + LabelMe 标注 → 返回 image + binary_mask"""

    def __init__(self, data_dir, split='train', val_ratio=0.2, augment=True):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / 'images'
        self.anno_dir = self.data_dir / 'anno'
        self.augment = augment and split == 'train'

        # 收集匹配的图像-标注对
        self.samples = []
        for jf in sorted(self.anno_dir.glob('*.json')):
            import json
            with open(jf) as f:
                ann = json.load(f)
            img_name = Path(ann.get('imagePath', '')).name
            if not img_name:
                img_name = jf.stem + '.jpg'
            img_path = self.img_dir / img_name
            if img_path.exists():
                self.samples.append((img_path, jf))

        # 分割训练/验证
        random.Random(42).shuffle(self.samples)
        n = len(self.samples)
        split_n = int(n * (1 - val_ratio))
        if split == 'train':
            self.samples = self.samples[:split_n]
        else:
            self.samples = self.samples[split_n:]
        logger.info(f"[{split}] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # 读取原图
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = img.shape[:2]

        # 读取标注, 生成 GT mask
        with open(json_path) as f:
            ann = json.load(f)
        gt_mask = np.zeros((H_orig, W_orig), dtype=np.float32)
        for shape in ann.get('shapes', []):
            if shape['label'] == 'scour pits':
                m = polygon_to_mask(shape['points'], H_orig, W_orig)
                gt_mask = np.maximum(gt_mask, m)

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Augmentation (train only)
        if self.augment and random.random() < 0.5:
            img = np.fliplr(img).copy()
            gt_mask = np.fliplr(gt_mask).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(gt_mask).unsqueeze(0).float()

        return img_t, mask_t, img_path.stem


# ============================================================
# Model: UNet + ResNet34 → 单头分割
# ============================================================
def _load_resnet34():
    import torchvision.models as models
    resnet = models.resnet34(weights=None)
    local = PROJECT_ROOT / 'resnet34-b627a593.pth'
    if local.exists():
        resnet.load_state_dict(torch.load(local, map_location='cpu', weights_only=True))
    return resnet


class ScourUNet(nn.Module):
    """UNet with ResNet34 encoder → scour pit binary mask"""

    def __init__(self):
        super().__init__()
        resnet = _load_resnet34()

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1   # 64
        self.enc3 = resnet.layer2   # 128
        self.enc4 = resnet.layer3   # 256
        self.enc5 = resnet.layer4   # 512

        self.up4 = self._up(512 + 256, 256)
        self.up3 = self._up(256 + 128, 128)
        self.up2 = self._up(128 + 64, 64)
        self.up1 = self._up(64 + 64, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 1, 1),
        )

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

        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(d1, x.shape[2:], mode='bilinear', align_corners=False)

        logits = self.final_conv(d1)
        return logits


# ============================================================
# Losses
# ============================================================
def dice_loss(pred, target, smooth=1.0):
    """DICE loss: 1 - (2*|pred∩target|) / (|pred|+|target|)"""
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(pred_logits, target_mask):
    """BCE + DICE 组合损失"""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
    pred_sigmoid = torch.sigmoid(pred_logits)
    dice = dice_loss(pred_sigmoid, target_mask)
    return bce + dice, {'bce': bce.item(), 'dice': dice.item()}


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def compute_metrics(pred_logits, target_mask, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    target = target_mask.float()

    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()

    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall}


# ============================================================
# Trainer
# ============================================================
class SegTrainer:
    def __init__(self, data_dir, epochs=100, batch_size=8, lr=1e-4):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = DataLoader(
            ScourSegDataset(data_dir, 'train', augment=True),
            batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_loader = DataLoader(
            ScourSegDataset(data_dir, 'val', augment=False),
            batch_size=batch_size, shuffle=False, num_workers=0)

        self.model = ScourUNet().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, epochs, eta_min=1e-6)
        self.scaler = GradScaler()

        self.out_dir = PROJECT_ROOT / 'result/checkpoints_supervised/random147_4 h8t1_4'
        os.makedirs(self.out_dir, exist_ok=True)

        self.best_dice = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_bce = 0
        total_dice = 0

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for img, mask, _ in pbar:
            img = img.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            with autocast():
                logits = self.model(img)
                loss, comps = combined_loss(logits, mask)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_bce += comps['bce']
            total_dice += comps['dice']
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        n = len(self.train_loader)
        return {'loss': total_loss / n, 'bce': total_bce / n, 'dice': total_dice / n}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        metrics_sum = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
        total_loss = 0

        for img, mask, _ in self.val_loader:
            img = img.to(self.device)
            mask = mask.to(self.device)

            logits = self.model(img)
            loss, _ = combined_loss(logits, mask)
            total_loss += loss.item()

            m = compute_metrics(logits, mask)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

        n = len(self.val_loader)
        return {
            'loss': total_loss / n,
            'iou': metrics_sum['iou'] / n,
            'dice': metrics_sum['dice'] / n,
            'precision': metrics_sum['precision'] / n,
            'recall': metrics_sum['recall'] / n,
        }

    def train(self):
        logger.info(f"Supervised Segmentation | {self.epochs} epochs | device={self.device}")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch()
            val_m = self.validate()
            self.scheduler.step()

            logger.info(
                f"E{epoch:3d} | train_loss={train_m['loss']:.4f} "
                f"val_loss={val_m['loss']:.4f} | "
                f"IoU={val_m['iou']:.4f} DICE={val_m['dice']:.4f} "
                f"P={val_m['precision']:.3f} R={val_m['recall']:.3f} "
                f"| {time.time()-t0:.1f}s")

            # 保存最佳模型
            if val_m['dice'] > self.best_dice:
                self.best_dice = val_m['dice']
                torch.save(self.model.state_dict(), self.out_dir / 'best_seg_model.pth')
                logger.info(f"  → Best model saved (DICE={self.best_dice:.4f})")

            # 定期保存
            if epoch % 20 == 0:
                torch.save(self.model.state_dict(), self.out_dir / f'seg_epoch_{epoch}.pth')

        torch.save(self.model.state_dict(), self.out_dir / 'seg_final.pth')
        logger.info(f"Done! Best DICE={self.best_dice:.4f}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data', type=str,
                        default=str(PROJECT_ROOT / 'data/random147_4 h8t1_4'))
    args = parser.parse_args()

    trainer = SegTrainer(args.data, args.epochs, args.batch_size, args.lr)
    trainer.train()


if __name__ == '__main__':
    main()
