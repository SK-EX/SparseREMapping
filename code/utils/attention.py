"""
Attention Modules for Terrain-Aware Depth Estimation
=====================================================

Specialized attention mechanisms for pile-surrounding scour
pit feature extraction:
  1. TerrainCrossAttention — Cross-attention with learnable terrain queries
  2. RadialPositionEncoding — Distance-from-pile positional bias
  3. MultiScaleSpatialAttention — Spatial attention at multiple scales
  4. PileGuidedAttention — Pile-mask-guided feature enhancement

Author: ScourDepth Project, 2026
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Radial Position Encoding
# ======================================================================

class RadialPositionEncoding(nn.Module):
    """
    Encode distance from pile center as sinusoidal position bias.

    The scour pit has radial symmetry around the pile. This encoding
    provides a strong geometric prior for the model.
    """

    def __init__(self, dim: int = 64, max_radius: float = 200.0):
        super().__init__()
        self.dim = dim
        self.max_radius = max_radius

    def forward(self, h: int, w: int,
                center: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h, w: feature map height/width
            center: (B, 2) pile center (y, x) in normalized [-1,1] coords,
                    or None for image center
        Returns:
            encoding: (1, dim, h, w)
        """
        yy = torch.linspace(-1, 1, h)
        xx = torch.linspace(-1, 1, w)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')

        if center is not None:
            cy, cx = center[:, 0].mean(), center[:, 1].mean()
            dist = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        else:
            dist = torch.sqrt(gx ** 2 + gy ** 2)

        dist = dist.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)

        # Sinusoidal encoding of distance
        encodings = []
        for i in range(self.dim // 2):
            scale = 2 ** (i / (self.dim // 4))
            encodings.append(torch.sin(dist * scale * math.pi))
            encodings.append(torch.cos(dist * scale * math.pi))

        return torch.cat(encodings, dim=1)  # (1, dim, h, w)


# ======================================================================
# Terrain Cross-Attention
# ======================================================================

class TerrainCrossAttention(nn.Module):
    """
    Cross-attention module for terrain feature enhancement.

    A set of learnable 'terrain tokens' attend to image feature maps,
    capturing morphological patterns of the scour pit (horseshoe shape,
    asymmetry, deposition patterns, etc.).

    The enhanced queries are projected back to the feature map space
    via attention-weighted fusion.
    """

    def __init__(self, dim: int = 256, n_tokens: int = 32,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.dim = dim

        # Learnable terrain tokens
        self.terrain_tokens = nn.Parameter(
            torch.randn(1, n_tokens, dim) * 0.02)

        # Cross-attention: terrain tokens ← image features
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)

        # Self-attention: inter-token refinement
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: (B, C, H, W) decoder feature map
        Returns:
            enhanced: (B, C, H, W) terrain-enhanced features
        """
        B, C, H, W = feat_map.shape
        assert C == self.dim, f"Channel mismatch: {C} vs {self.dim}"

        # Spatial → sequence: (B, C, H, W) → (B, H*W, C)
        x = feat_map.flatten(2).permute(0, 2, 1)

        # Expand terrain tokens
        tokens = self.terrain_tokens.expand(B, -1, -1)

        # Cross-attention: tokens query image features
        tokens = self.norm_cross(
            tokens + self.cross_attn(tokens, x, x)[0])

        # Self-attention among tokens
        tokens = self.norm_self(
            tokens + self.self_attn(tokens, tokens, tokens)[0])

        # FFN
        tokens = self.norm_ffn(tokens + self.ffn(tokens))

        # Reverse attention: project tokens → feature map
        attn = torch.bmm(x, tokens.transpose(1, 2)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        enhanced = torch.bmm(attn, tokens)
        enhanced = enhanced.permute(0, 2, 1).reshape(B, C, H, W)

        return feat_map + 0.1 * enhanced


# ======================================================================
# Multi-Scale Spatial Attention
# ======================================================================

class MultiScaleSpatialAttention(nn.Module):
    """
    Spatial attention at multiple receptive field sizes.

    Uses dilated convolutions to capture scour features at different
    scales — from fine sediment texture to global scour morphology.
    """

    def __init__(self, channels: int, dilations=[1, 2, 4, 8]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, 3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(channels // 4),
                nn.GELU(),
            ) for d in dilations
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([b(x) for b in self.branches], dim=1)
        attn = self.fusion(multi_scale)
        return x * attn


# ======================================================================
# Pile-Guided Attention
# ======================================================================

class PileGuidedAttention(nn.Module):
    """
    Attention guided by pile mask to focus on pile-surrounding regions.

    Uses the pile location as a spatial prior to weight features,
    emphasizing the annular region where scour typically occurs.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels + 1, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor,
                pile_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) features
            pile_mask: (B, 1, H, W) binary pile region mask, optional
        """
        if pile_mask is None:
            return x
        if pile_mask.shape[2:] != x.shape[2:]:
            pile_mask = F.interpolate(pile_mask, x.shape[2:], mode='nearest')

        # Concatenate pile mask for attention
        pooled = F.adaptive_avg_pool2d(
            torch.cat([x, pile_mask], dim=1) if pile_mask.shape[1] == 1
            else torch.cat([x, pile_mask[:, :1]], dim=1), 1)
        attn = self.channel_attn(pooled)
        return x * attn


# ======================================================================
# Full Attention Module
# ======================================================================

class ScourAttentionModule(nn.Module):
    """
    Combined attention module for scour depth estimation.

    Pipeline:
      Feature Map → Radial PE → Terrain Cross-Attn → Multi-Scale Spatial Attn
                             → Pile-Guided Attn (if pile_mask available)

    This module can be inserted at any stage of the decoder.
    """

    def __init__(self, channels: int = 256,
                 n_terrain_tokens: int = 32,
                 n_heads: int = 8):
        super().__init__()
        self.radial_pe = RadialPositionEncoding(dim=channels)
        self.terrain_attn = TerrainCrossAttention(
            dim=channels, n_tokens=n_terrain_tokens, n_heads=n_heads)
        self.spatial_attn = MultiScaleSpatialAttention(channels)
        self.pile_attn = PileGuidedAttention(channels)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor,
                pile_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
            pile_mask: (B, 1, H, W) optional pile mask
        Returns:
            enhanced: (B, C, H, W)
        """
        # Radial position encoding
        pe = self.radial_pe(x.shape[2], x.shape[3]).to(x.device)
        x = x + 0.02 * pe.expand(x.shape[0], -1, -1, -1)

        # Terrain cross-attention
        x = self.terrain_attn(x)

        # Multi-scale spatial attention
        x = self.spatial_attn(x)

        # Pile-guided attention
        x = self.pile_attn(x, pile_mask)

        return self.norm(x)






