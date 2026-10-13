
"""
Image Transform Utilities
==========================

Data augmentation and preprocessing transforms for scour pit images.

- Geometric: random flip, rotation (small angle), perspective warp
- Photometric: brightness, contrast, saturation jitter
- Normalization: ImageNet stats for DINOv2

Author: ScourDepth Project, 2026
"""

import random
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance


# ======================================================================
# Normalization
# ======================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_imagenet(img: torch.Tensor) -> torch.Tensor:
    """Normalize (B,3,H,W) tensor with ImageNet stats."""
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std

def denormalize_imagenet(img: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization."""
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(1, 3, 1, 1)
    return img * std + mean


# ======================================================================
# Geometric Transforms
# ======================================================================

def random_horizontal_flip(image: np.ndarray, mask: np.ndarray,
                           p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random horizontal flip."""
    if random.random() < p:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
    return image, mask


def random_vertical_flip(image: np.ndarray, mask: np.ndarray,
                         p: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random vertical flip (less likely to preserve flow direction)."""
    if random.random() < p:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
    return image, mask


def random_rotation(image: np.ndarray, mask: np.ndarray,
                    max_angle: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """Apply small random rotation (scour pit is view-invariant)."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def random_resized_crop(image: np.ndarray, mask: np.ndarray,
                        scale: Tuple[float, float] = (0.8, 1.0),
                        size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Random resized crop to simulate different camera distances."""
    h, w = image.shape[:2]
    s = random.uniform(*scale)
    new_h, new_w = int(h * s), int(w * s)
    top = random.randint(0, max(0, h - new_h))
    left = random.randint(0, max(0, w - new_w))
    image = image[top:top + new_h, left:left + new_w]
    mask = mask[top:top + new_h, left:left + new_w]
    if size is not None:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


# ======================================================================
# Photometric Transforms
# ======================================================================

def random_brightness(image: np.ndarray, factor_range=(0.85, 1.15)) -> np.ndarray:
    """Adjust brightness."""
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_contrast(image: np.ndarray, factor_range=(0.9, 1.1)) -> np.ndarray:
    """Adjust contrast."""
    factor = random.uniform(*factor_range)
    mean = image.mean(axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def random_saturation(image: np.ndarray, factor_range=(0.9, 1.1)) -> np.ndarray:
    """Adjust saturation."""
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ======================================================================
# Combined Augmentation Pipeline
# ======================================================================

class ScourAugmentation:
    """
    Combined augmentation pipeline for scour pit training.

    Order: geometric → resize → photometric → tensor → normalize
    """

    def __init__(self,
                 img_size: int = 518,
                 flip_h_prob: float = 0.5,
                 flip_v_prob: float = 0.2,
                 rotation_deg: float = 10.0,
                 brightness_range=(0.85, 1.15),
                 contrast_range=(0.9, 1.1)):
        self.img_size = img_size
        self.flip_h_prob = flip_h_prob
        self.flip_v_prob = flip_v_prob
        self.rotation_deg = rotation_deg
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, image: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: (H, W, 3) uint8 RGB
            mask:  (H, W) uint8/float mask (optional)
        Returns:
            dict with 'image' (3, H, W) float tensor and optional 'mask'
        """
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        # Geometric
        image, mask = random_horizontal_flip(image, mask, self.flip_h_prob)
        image, mask = random_vertical_flip(image, mask, self.flip_v_prob)
        image, mask = random_rotation(image, mask, self.rotation_deg)

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # Photometric
        image = random_brightness(image, self.brightness_range)
        image = random_contrast(image, self.contrast_range)

        # To tensor
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # [0,1]
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return {'image': image_t, 'mask': mask_t}


def resize_and_normalize(image: np.ndarray,
                         img_size: int = 518) -> torch.Tensor:
    """Simple resize + normalize for inference."""
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image_t
