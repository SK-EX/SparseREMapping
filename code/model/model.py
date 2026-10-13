"""
ScourDepthFormer — 冲刷坑形态与高程映射网络
=============================================

基于 DINOv2 backbone 的 terrain-aware 深度估计网络,
专门针对单桩周围冲刷坑微地形的精细感知。

Key contributions:
  1. Terrain Attention Module — 学习桩周地形的结构化先验
  2. Camera Pose Estimator — 规范化多视角透视畸变
  3. Multi-Scale FPN Decoder — 从 ViT 特征恢复高分辨率高程图
  4. Dual-Head Output — 高程映射 + 冲刷坑分割

Architecture:
  RGB ─→ DINOv2 (frozen) ─→ Multi-Scale Features
         │                    ├─→ Camera Pose Estimator
         │                    ├─→ FPN Decoder
         │                    │    └─→ Terrain Attention
         │                    ├─→ Elevation Head → elevation map (±mm)
         │                    └─→ Scour Mask Head → binary mask

Reference:
  DINOv2 (Oquab et al., arXiv:2304.07193, 2023)
  Depth Anything V2 (Yang et al., arXiv:2406.09414, 2024)
  VGGSfM (Wang et al., arXiv:2312.13771, 2023)

Author:  ScourDepth Project
Date:   2026-07
"""

import math, sys, logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# DINOv2 backbone from Depth Anything V2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Depth-Anything-V2'))
from depth_anything_v2.dinov2 import DINOv2


# ======================================================================
# Utility Modules
# ======================================================================

def _get_activation_fn(activation: str):
    """Return activation function by name."""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Unknown activation: {activation}")


class LayerNorm2d(nn.Module):
    """2D Layer Normalization."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ======================================================================
# Basic Building Blocks
# ======================================================================

class ConvBlock(nn.Module):
    """Conv2d + Norm + Activation, with optional residual."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, groups: int = 1,
                 norm: str = 'batch', activation: str = 'gelu',
                 residual: bool = False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, groups=groups, bias=(norm == 'none'))
        self.norm = nn.BatchNorm2d(out_channels) if norm == 'batch' else \
                    LayerNorm2d(out_channels) if norm == 'layer' else nn.Identity()
        self.act = _get_activation_fn(activation) if activation != 'none' else nn.Identity()
        self.residual = residual and (in_channels == out_channels) and (stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.residual:
            x = x + identity
        return x


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class SpatialAttention(nn.Module):
    """Spatial attention via large-kernel depthwise conv."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class ResBlock(nn.Module):
    """Residual block with optional SE and spatial attention."""

    def __init__(self, channels: int, use_se: bool = False,
                 use_spatial_attn: bool = False):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, activation='gelu')
        self.conv2 = ConvBlock(channels, channels, activation='none')
        self.se = ChannelAttention(channels) if use_se else nn.Identity()
        self.spatial = SpatialAttention() if use_spatial_attn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.spatial(x)
        return self.act(x + residual)


# ======================================================================
# Feature Projection (ViT tokens → 2D feature maps)
# ======================================================================

class ViTFeatureProjector(nn.Module):
    """Project ViT token sequences to multi-scale 2D feature maps."""

    def __init__(self, vit_dim: int = 1024,
                 out_channels: List[int] = [256, 256, 256, 256]):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(vit_dim, oc, 1),
                nn.BatchNorm2d(oc),
                nn.GELU(),
                ResBlock(oc),
            ) for oc in out_channels
        ])

    def forward(self, features: List[torch.Tensor],
                patch_h: int, patch_w: int) -> List[torch.Tensor]:
        """Tokens → 2D maps."""
        out = []
        for feat, proj in zip(features, self.projections):
            B, N, C = feat.shape
            fm = feat.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
            out.append(proj(fm))
        return out


# ======================================================================
# Camera Pose Estimator
# ======================================================================

class CameraPoseEstimator(nn.Module):
    """
    Estimate relative camera pose from global image features.

    Uses DINOv2 CLS token to predict 6-DoF camera pose
    (quaternion rotation + translation vector).

    This enables the model to disentangle true scour morphology
    from perspective distortion across different viewpoints.

    Reference: VGGSfM (Wang et al., 2023), COLMAP (Schönberger et al., 2016)
    """

    def __init__(self, feat_dim: int = 1024, hidden_dim: int = 512,
                 num_layers: int = 3):
        super().__init__()
        self.feat_dim = feat_dim

        layers = []
        in_dim = feat_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)

        self.head_rotation = nn.Linear(hidden_dim // 2, 4)
        self.head_translation = nn.Linear(hidden_dim // 2, 3)
        self.head_focal = nn.Linear(hidden_dim // 2, 1)

    def forward(self, tokens: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: (B, N, C) ViT patch tokens
        Returns:
            pose dict with rotation (B,4), translation (B,3), focal (B,1)
        """
        # Global feature pooling (mean over tokens)
        if tokens.dim() == 3:
            global_feat = tokens.mean(dim=1)  # (B, C)
        else:
            global_feat = tokens

        feat = self.encoder(global_feat)

        rotation = F.normalize(self.head_rotation(feat), dim=-1)
        translation = self.head_translation(feat)
        focal = F.softplus(self.head_focal(feat)) + 100.0  # ensure positive

        result = {
            'rotation': rotation,
            'translation': translation,
            'focal_length': focal,
        }
        if return_features:
            result['features'] = feat
        return result


# ======================================================================
# Terrain Attention Module
# ======================================================================

class TerrainAttention(nn.Module):
    """
    Pile-surrounding terrain attention module.

    Learns a set of terrain query vectors that attend to image features
    to capture the structured morphology of scour pits (horseshoe shape,
    radial symmetry, etc.).

    Design:
      - N learnable terrain queries capture different morphological aspects
        (e.g., upstream scour, downstream deposition, lateral extent)
      - Cross-attention with image features extracts terrain-relevant signals
      - Self-attention among queries models inter-dependencies
      - Output is fused back into feature map via reverse attention
    """

    def __init__(self, dim: int = 256, n_queries: int = 32,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_queries = n_queries
        self.dim = dim

        # Learnable terrain queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)

        # Cross-attention: queries ← image features
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm_ffn = nn.LayerNorm(dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: (B, C, H, W) feature map from decoder
        Returns:
            enhanced: (B, C, H, W) terrain-enhanced feature map
        """
        B, C, H, W = feat_map.shape
        if C != self.dim:
            raise ValueError(f"Feature dim {C} != module dim {self.dim}")

        # Flatten: (B, C, H, W) → (B, H*W, C)
        x = feat_map.flatten(2).permute(0, 2, 1)

        # Expand queries
        q = self.queries.expand(B, -1, -1)

        # Cross-attention: terrain queries attend to image
        q = self.norm_cross(q + self.cross_attn(q, x, x)[0])

        # Self-attention: queries refine each other
        q = self.norm_self(q + self.self_attn(q, q, q)[0])

        # FFN
        q = self.norm_ffn(q + self.ffn(q))

        # Reverse attention: project queries back to feature map
        q_proj = self.output_proj(q)  # (B, K, C)
        attn = torch.bmm(x, q_proj.transpose(1, 2)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        enhanced = torch.bmm(attn, q_proj)
        enhanced = enhanced.permute(0, 2, 1).reshape(B, C, H, W)

        return feat_map + 0.1 * enhanced


# ======================================================================
# Multi-Scale FPN Decoder
# ======================================================================

class MultiScaleDecoder(nn.Module):
    """
    FPN-style decoder that reconstructs high-resolution features
    from ViT multi-scale token features.

    Architecture: Deep → Shallow progressive upsampling with
    skip connections from earlier ViT layers.
    """

    def __init__(self, vit_dim: int = 1024,
                 feature_dims: List[int] = [256, 256, 256, 256],
                 decoder_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.projector = ViTFeatureProjector(vit_dim, feature_dims)

        # Deepest → Shallowest decoder blocks
        n = len(feature_dims)
        self.upsample_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()

        in_ch = feature_dims[-1]
        for i in range(n - 1, 0, -1):
            skip_ch = feature_dims[i - 1]
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ConvBlock(in_ch, skip_ch, kernel_size=3, activation='gelu'),
                )
            )
            self.fusion_blocks.append(
                ConvBlock(skip_ch * 2, skip_ch, kernel_size=3, activation='gelu')
            )
            in_ch = skip_ch

        # Final upsampling to decoder_dim
        self.final_upsample = nn.Sequential(
            ConvBlock(feature_dims[0], 256, kernel_size=3, activation='gelu'),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(256, 128, kernel_size=3, activation='gelu'),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(128, out_dim, kernel_size=3, activation='gelu'),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(out_dim, out_dim, kernel_size=3, activation='gelu'),
        )

    def forward(self, vit_features: List[torch.Tensor],
                patch_h: int, patch_w: int) -> torch.Tensor:
        """
        Args:
            vit_features: List of (B, N, C) ViT intermediate features
            patch_h, patch_w: ViT patch grid height/width
        Returns:
            decoder_feat: (B, out_dim, H', W')
        """
        feat_maps = self.projector(vit_features, patch_h, patch_w)

        # Progressive upsampling
        x = feat_maps[-1]
        for i, (up, fuse) in enumerate(
            zip(self.upsample_blocks, self.fusion_blocks)):
            x = up(x)
            skip = feat_maps[-(i + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            x = fuse(torch.cat([x, skip], dim=1))

        return self.final_upsample(x)


# ======================================================================
# Output Heads
# ======================================================================

class ElevationHead(nn.Module):
    """
    Predict per-pixel elevation relative to bed surface.

    Convention:
      - Positive values = sand bed surface above reference
      - Negative values = scour pit below bed surface
      - Zero = bed surface reference plane
    """

    def __init__(self, in_channels: int = 128, hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            ConvBlock(in_channels, hidden, activation='gelu'),
            ResBlock(hidden, use_se=True),
            ConvBlock(hidden, hidden // 2, activation='gelu'),
            ResBlock(hidden // 2),
            nn.Conv2d(hidden // 2, 1, 1),
        )
        # Learnable scaling factor (mm range)
        self.scale = nn.Parameter(torch.tensor(100.0), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            elevation: (B, 1, H, W) elevation in mm
        """
        raw = self.head(x)
        return raw * self.scale.abs()


class ScourMaskHead(nn.Module):
    """Predict scour pit binary mask."""

    def __init__(self, in_channels: int = 128, hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            ConvBlock(in_channels, hidden, activation='gelu'),
            ResBlock(hidden, use_se=True),
            ConvBlock(hidden, hidden // 2, activation='gelu'),
            nn.Conv2d(hidden // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            mask: (B, 1, H, W) in [0, 1]
        """
        return torch.sigmoid(self.head(x))


# ======================================================================
# Main Model
# ======================================================================

class ScourDepthFormer(nn.Module):
    """
    ScourDepthFormer: Terrain-Aware Scour Pit Depth Estimation.

    Input:
        rgb: (B, 3, H, W) RGB image of pile + scour bed

    Output:
        elevation:    (B, 1, H, W) elevation map (mm)
                       positive = sand bed, negative = scour pit
        scour_mask:   (B, 1, H, W) scour pit mask [0,1]
        camera_pose:  dict with rotation, translation, focal_length
        features:     intermediate decoder features (for analysis)

    Usage:
        model = ScourDepthFormer(encoder='vitl', img_size=518)
        output = model(rgb_tensor)
        elevation_mm = output['elevation']  # (B, 1, H, W)
    """

    # ViT layer indices for intermediate features
    LAYER_INDICES = {
        'vits': [2, 5, 8, 11],
        'vitb': [2, 5, 8, 11],
        'vitl': [4, 11, 17, 23],
        'vitg': [9, 19, 29, 39],
    }

    def __init__(self,
                 encoder: str = 'vitl',
                 img_size: int = 518,
                 feature_dims: Optional[List[int]] = None,
                 decoder_dim: int = 256,
                 out_dim: int = 128,
                 n_terrain_queries: int = 32):
        super().__init__()

        # Validate
        assert encoder in self.LAYER_INDICES, \
            f"Encoder must be one of {list(self.LAYER_INDICES.keys())}"
        assert img_size % 14 == 0, "img_size must be divisible by 14 (ViT patch size)"

        self.encoder_type = encoder
        self.img_size = img_size

        # ── DINOv2 Backbone (frozen) ──
        logger.info(f"Loading DINOv2-{encoder} backbone...")
        self.backbone = DINOv2(model_name=encoder)
        vit_dim = self.backbone.embed_dim  # 384(vits), 768(vitb), 1024(vitl), 1536(vitg)

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # ── Feature dimensions ──
        if feature_dims is None:
            feature_dims = [256, 256, 256, 256]

        # ── Camera Pose Estimator ──
        self.pose_estimator = CameraPoseEstimator(
            feat_dim=vit_dim, hidden_dim=512, num_layers=3)

        # ── Pose-to-feature injection ──
        pose_enc_dim = 4 + 3 + 1  # quat + trans + focal
        self.pose_injector = nn.Sequential(
            nn.Linear(pose_enc_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        # ── Multi-Scale Decoder ──
        self.decoder = MultiScaleDecoder(
            vit_dim=vit_dim,
            feature_dims=feature_dims,
            decoder_dim=decoder_dim,
            out_dim=out_dim,
        )

        # ── Terrain Attention ──
        self.terrain_attn = TerrainAttention(
            dim=out_dim, n_queries=n_terrain_queries)

        # ── Output Heads ──
        self.elevation_head = ElevationHead(in_channels=out_dim)
        self.scour_head = ScourMaskHead(in_channels=out_dim)

        # ── Logging ──
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"ScourDepthFormer: {total/1e6:.1f}M total, "
                    f"{trainable/1e6:.1f}M trainable "
                    f"(backbone={vit_dim/1e6:.1f}M frozen)")

    def forward(self, rgb: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            rgb: (B, 3, H, W) normalized to [0, 1]
            return_features: if True, return intermediate features

        Returns:
            dict with keys: elevation, scour_mask, camera_pose
        """
        B, C, H_in, W_in = rgb.shape
        patch_h, patch_w = H_in // 14, W_in // 14

        # ── 1. DINOv2 multi-scale features ──
        layer_indices = self.LAYER_INDICES[self.encoder_type]
        vit_features = self.backbone.get_intermediate_layers(
            rgb, layer_indices, return_class_token=False)
        # List of (B, N, vit_dim) tensors

        # ── 2. Camera pose estimation ──
        camera_pose = self.pose_estimator(vit_features[-1])

        # ── 3. Multi-scale decoding ──
        decoder_feat = self.decoder(vit_features, patch_h, patch_w)

        # ── 4. Inject pose encoding ──
        pose_code = torch.cat([
            camera_pose['rotation'],
            camera_pose['translation'],
            camera_pose['focal_length'],
        ], dim=-1)
        pose_feat = self.pose_injector(pose_code)
        pose_feat = pose_feat.unsqueeze(-1).unsqueeze(-1)
        pose_feat = pose_feat.expand(-1, -1,
                                     decoder_feat.shape[2],
                                     decoder_feat.shape[3])
        decoder_feat = decoder_feat + 0.05 * pose_feat

        # ── 5. Terrain attention ──
        decoder_feat = self.terrain_attn(decoder_feat)

        # ── 6. Output heads ──
        elevation = self.elevation_head(decoder_feat)
        scour_mask = self.scour_head(decoder_feat)

        # ── 7. Resize to input resolution ──
        if elevation.shape[2:] != (H_in, W_in):
            elevation = F.interpolate(
                elevation, (H_in, W_in),
                mode='bilinear', align_corners=False)
            scour_mask = F.interpolate(
                scour_mask, (H_in, W_in),
                mode='bilinear', align_corners=False)

        result = {
            'elevation': elevation,
            'scour_mask': scour_mask,
            'camera_pose': camera_pose,
        }
        if return_features:
            result['features'] = decoder_feat

        return result

    @torch.no_grad()
    def infer(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience wrapper for inference (no grad)."""
        self.eval()
        return self.forward(rgb)
