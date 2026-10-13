"""
Camera Pose Estimation for Multi-View Scour Pit Analysis
=========================================================

Estimates relative camera pose from single RGB images of pile+scour bed
scenes. The pose is used to normalize the scour pit appearance across
different viewpoints, enabling consistent morphology measurement.

Method:
  - DINOv2 global features → MLP → 6-DoF pose (quaternion + translation)
  - Focal length estimated from FoV cues
  - Self-supervised via multi-view consistency (no pose GT needed)

Reference:
  VGGSfM (Wang et al., arXiv:2312.13771, 2023)
  COLMAP (Schönberger & Frahm, CVPR 2016)
  PoseNet (Kendall et al., ICCV 2015)

Author: ScourDepth Project, 2026
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Quaternion Utilities
# ======================================================================

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion (w, x, y, z) to 3×3 rotation matrix.

    Args:
        q: (..., 4) [w, x, y, z] normalized quaternion
    Returns:
        R: (..., 3, 3) rotation matrix
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1),
    ], dim=-2)
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert 3×3 rotation matrix to unit quaternion [w,x,y,z]."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)
    mask = trace > 0
    t = torch.sqrt(trace[mask] + 1.0)
    q[mask, 0] = 0.5 * t
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / (2 * t)
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / (2 * t)
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / (2 * t)
    return q


def relative_pose_error(pose1: Dict[str, torch.Tensor],
                        pose2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compute difference between two camera poses."""
    R1 = quaternion_to_rotation_matrix(pose1['rotation'])
    R2 = quaternion_to_rotation_matrix(pose2['rotation'])
    R_rel = torch.bmm(R2.transpose(1, 2), R1)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    t_diff = (pose1['translation'] - pose2['translation']).norm(dim=-1)
    return {'rotation_error_rad': angle, 'translation_error': t_diff}


# ======================================================================
# Camera Pose Estimator
# ======================================================================

class CameraPoseEstimator(nn.Module):
    """
    Estimate 6-DoF camera pose from image features.

    Uses a lightweight MLP on pooled features to predict:
      - Rotation (unit quaternion, 4-DoF)
      - Translation (3-DoF)
      - Focal length (1-DoF, optional)
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim // (2 ** (num_layers - 1 - i))
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)

        final_dim = hidden_dim // (2 ** (num_layers - 1))
        self.head_rotation = nn.Linear(final_dim, 4)
        self.head_translation = nn.Linear(final_dim, 3)
        self.head_focal = nn.Linear(final_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, C) or (B, N, C) image features
        Returns:
            pose dict with 'rotation', 'translation', 'focal_length'
        """
        if features.dim() == 3:
            features = features.mean(dim=1)  # pool over tokens
        x = self.encoder(features)
        return {
            'rotation': F.normalize(self.head_rotation(x), dim=-1),
            'translation': self.head_translation(x),
            'focal_length': F.softplus(self.head_focal(x)) + 100.0,
        }


# ======================================================================
# Multi-View Pose Consistency
# ======================================================================

class PoseConsistencyLoss(nn.Module):
    """
    Self-supervised pose consistency loss.

    Encourages consistent relative pose predictions across views
    of the same scene (same scour pit).
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, poses: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = poses['rotation'].shape[0]
        if B < 2:
            return torch.tensor(0.0, device=poses['rotation'].device)

        loss = 0.0
        count = 0
        for i in range(B):
            for j in range(i + 1, B):
                err = relative_pose_error(
                    {k: v[i:i+1] for k, v in poses.items()},
                    {k: v[j:j+1] for k, v in poses.items()})
                loss += err['rotation_error_rad'].mean()
                count += 1
        return self.weight * loss / max(count, 1)


# ======================================================================
# Camera Coordinate Transform
# ======================================================================

class CameraTransformer(nn.Module):
    """
    Transform 3D coordinates between camera frames.

    Given estimated camera poses, project elevation maps between views
    for multi-view consistency training.
    """

    def __init__(self):
        super().__init__()

    def world_to_camera(self, points_3d: torch.Tensor,
                        pose: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Transform points from world to camera coordinates."""
        R = quaternion_to_rotation_matrix(pose['rotation'])
        t = pose['translation'].unsqueeze(-1)
        return torch.bmm(R, points_3d) + t

    def camera_to_image(self, points_3d: torch.Tensor,
                        focal_length: torch.Tensor,
                        img_size: Tuple[int, int]) -> torch.Tensor:
        """Project 3D camera points to 2D image plane."""
        fx = fy = focal_length
        cx, cy = img_size[1] / 2, img_size[0] / 2
        x = points_3d[:, 0] / (points_3d[:, 2] + 1e-6) * fx + cx
        y = points_3d[:, 1] / (points_3d[:, 2] + 1e-6) * fy + cy
        return torch.stack([x, y], dim=-1)

    def unproject_depth(self, depth: torch.Tensor,
                        pose: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Unproject depth map to 3D point cloud in world coordinates.

        Args:
            depth: (B, H, W) depth map
            pose: camera pose dict
        Returns:
            points_3d: (B, 3, H, W) world coordinates
        """
        B, H, W = depth.shape
        depth = depth.view(B, 1, H, W)
        device = depth.device

        # Pixel grid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij')
        grid = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)

        # Camera → World
        R = quaternion_to_rotation_matrix(pose['rotation'])
        t = pose['translation'].unsqueeze(-1).unsqueeze(-1)

        points_cam = grid * depth.permute(0, 2, 3, 1)
        points_cam = points_cam.reshape(B, -1, 3).transpose(1, 2)  # (B, 3, N)
        points_world = torch.bmm(R.transpose(1, 2), points_cam - t)

        return points_world.reshape(B, 3, H, W)
