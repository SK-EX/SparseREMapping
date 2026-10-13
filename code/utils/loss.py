"""
Loss Functions for ScourDepthFormer
====================================

Multi-task loss components:
  1. Elevation regression (Huber + physics)
  2. Segmentation (Dice + Focal + Boundary-aware)
  3. Edge-aware smoothness (image-guided)
  4. Physical priors (scour depth range, pile geometry)
  5. Multi-view consistency

Author: ScourDepth Project, 2026
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Utilities
# ======================================================================

def _sobel_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute spatial gradient magnitude via Sobel operator."""
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)  # RGB → grayscale
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), sx)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), sy)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


# ======================================================================
# Segmentation Losses
# ======================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.contiguous().view(-1)
        t = target.contiguous().view(-1)
        inter = (p * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.clamp(1e-7, 1 - 1e-7)
        pt = p * target + (1 - p) * (1 - target)
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        return -(at * (1 - pt) ** self.gamma * torch.log(pt)).mean()


# ======================================================================
# Regularization Losses
# ======================================================================

class EdgeAwareSmoothnessLoss(nn.Module):
    """Penalize depth variations in texture-flat regions, allow them at edges."""

    def forward(self, depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if image.shape[2:] != depth.shape[2:]:
            image = F.interpolate(image, depth.shape[2:], mode='bilinear',
                                  align_corners=False)
        if image.shape[1] > 1:
            image = image.mean(dim=1, keepdim=True)  # RGB → grayscale
        dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        ix = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
        iy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()
        return ((torch.exp(-ix * 5) * dx.abs()).mean() +
                (torch.exp(-iy * 5) * dy.abs()).mean())


class GradientConsistencyLoss(nn.Module):
    """Encourage depth gradients to align with image gradients."""
    def forward(self, depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        d_grad = _sobel_gradient(depth)
        i_grad = _sobel_gradient(image)
        if i_grad.shape != d_grad.shape:
            i_grad = F.interpolate(i_grad, d_grad.shape[2:], mode='bilinear',
                                   align_corners=False)
        return F.mse_loss(d_grad / (d_grad.max() + 1e-6),
                          i_grad / (i_grad.max() + 1e-6))


# ======================================================================
# Physics Priors
# ======================================================================

class PhysicsPriorLoss(nn.Module):
    """
    Physical constraints for scour pit:
    - Depth range: 0 < scour_depth < 2.5 × pile_diameter
    - Spatial extent: scour concentrated within ~2D radius from center
    - Net erosion: mean elevation in scour region should be negative
    """
    def __init__(self, pile_diameter_mm: float = 90.0, max_depth_mm: float = 250.0):
        super().__init__()
        self.D = pile_diameter_mm
        self.max_depth = max_depth_mm

    def forward(self, elevation: torch.Tensor, scour_mask: torch.Tensor,
                pile_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        losses = {}
        depth = -elevation.clamp(max=0)
        losses['depth_max'] = F.relu(depth - self.max_depth).mean()
        losses['depth_min'] = F.relu(10.0 - depth * scour_mask).mean()

        # Spatial constraint: scour beyond ~2D from center unlikely
        B, _, H, W = scour_mask.shape
        yy = torch.linspace(-1, 1, H, device=elevation.device)
        xx = torch.linspace(-1, 1, W, device=elevation.device)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        dist = torch.sqrt(gx**2 + gy**2).unsqueeze(0).unsqueeze(0)
        losses['scour_extent'] = (scour_mask * F.relu(dist - 0.75)).mean()

        return losses


# ======================================================================
# Combined Loss
# ======================================================================

class ScourDepthLoss(nn.Module):
    """
    Combined multi-task loss for ScourDepthFormer.

    L_total = λ_elev * L_huber(elevation, gt)
            + λ_seg  * (L_dice + L_focal)
            + λ_grad * L_gradient_consistency
            + λ_smooth * L_edge_aware_smoothness
            + λ_phys * L_physics_prior
            + λ_consist * L_multiview_consistency

    The loss is designed to handle missing annotations gracefully —
    components with missing targets are set to zero.
    """

    def __init__(self,
                 lambda_elev: float = 1.0,
                 lambda_seg: float = 1.0,
                 lambda_grad: float = 0.1,
                 lambda_smooth: float = 0.05,
                 lambda_phys: float = 0.1,
                 lambda_consist: float = 0.1,
                 pile_diameter_mm: float = 90.0):
        super().__init__()
        self.l_elev = lambda_elev
        self.l_seg = lambda_seg
        self.l_grad = lambda_grad
        self.l_smooth = lambda_smooth
        self.l_phys = lambda_phys
        self.l_consist = lambda_consist

        self.huber = nn.HuberLoss(delta=5.0)
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.grad_consistency = GradientConsistencyLoss()
        self.smoothness = EdgeAwareSmoothnessLoss()
        self.physics = PhysicsPriorLoss(pile_diameter_mm)

    def forward(self,
                pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                rgb: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred:   {'elevation': (B,1,H,W), 'scour_mask': (B,1,H,W), ...}
            target: {'elevation_map': (B,1,H,W) optional,
                     'scour_mask': (B,1,H,W) optional,
                     'pile_mask': (B,1,H,W) optional}
            rgb:    (B,3,H,W) original image, for smoothness/gradient losses

        Returns:
            dict of per-component losses with 'total' key
        """
        losses: Dict[str, torch.Tensor] = {}
        device = pred['elevation'].device

        # ── Elevation Regression ──
        gt_elev = target.get('elevation_map')
        if gt_elev is not None:
            p = pred['elevation']
            if p.shape != gt_elev.shape:
                gt_elev = F.interpolate(gt_elev, p.shape[2:],
                                        mode='bilinear', align_corners=True)
            losses['elev_huber'] = self.huber(p, gt_elev)
        else:
            losses['elev_huber'] = torch.tensor(0.0, device=device)

        # ── Segmentation ──
        gt_mask = target.get('scour_mask')
        if gt_mask is not None:
            with torch.cuda.amp.autocast(enabled=False):
                pm = pred['scour_mask'].float().clamp(1e-7, 1 - 1e-7)
                tm = gt_mask.float()
                if pm.shape != tm.shape:
                    tm = F.interpolate(tm, pm.shape[2:], mode='nearest')
            losses['seg_dice'] = self.dice(pm, tm)
            losses['seg_focal'] = self.focal(pm, tm)
        else:
            losses['seg_dice'] = losses['seg_focal'] = torch.tensor(0.0, device=device)

        # ── Gradient Consistency ──
        losses['grad_consist'] = self.grad_consistency(
            pred['elevation'], rgb if rgb is not None else pred['elevation'])

        # ── Edge-Aware Smoothness ──
        losses['smooth'] = self.smoothness(
            pred['elevation'], rgb if rgb is not None else pred['elevation'])

        # ── Physics Priors ──
        phys = self.physics(pred['elevation'], pred['scour_mask'],
                            target.get('pile_mask'))
        losses.update({f'phys_{k}': v for k, v in phys.items()})

        # ── Multi-View Consistency ──
        B = pred['elevation'].shape[0]
        if B >= 2:
            masked_elev = pred['elevation'] * pred['scour_mask']
            mean_depth = -masked_elev.sum([1, 2, 3]) / \
                         (pred['scour_mask'].sum([1, 2, 3]) + 1e-6)
            losses['consist'] = mean_depth.std()
        else:
            losses['consist'] = torch.tensor(0.0, device=device)

        # ── Total ──
        losses['total'] = (
            self.l_elev * losses['elev_huber'] +
            self.l_seg * (losses['seg_dice'] + losses['seg_focal']) +
            self.l_grad * losses['grad_consist'] +
            self.l_smooth * losses['smooth'] +
            self.l_phys * sum(phys.values()) +
            self.l_consist * losses['consist']
        )

        return losses
