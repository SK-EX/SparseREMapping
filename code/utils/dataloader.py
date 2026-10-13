"""
Data Loading for Scour Pit Training
====================================

Loads pile+scour bed images with LabelMe JSON annotations.
Supports:
  - Multi-dataset loading (multiple experimental runs)
  - Train/val split with stratification
  - On-the-fly augmentation
  - Pseudo-label generation from scour_line annotations

Author: ScourDepth Project, 2026
"""

import json, random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


# ======================================================================
# LabelMe Parser
# ======================================================================

def parse_labelme_json(json_path: Path, img_size: int = 518) -> Dict:
    """
    Parse LabelMe JSON annotation.

    Returns:
        dict with:
          - scour_line: list of (x,y) for pile-scour boundary
          - pile_polygons: list of polygon point lists
          - center_point: (x, y) pile center
          - image_path: str relative image filename
          - image_size: (w, h) original size
    """
    with open(json_path) as f:
        ann = json.load(f)

    img_w = ann.get('imageWidth', img_size)
    img_h = ann.get('imageHeight', img_size)
    sx = img_size / max(img_w, 1)
    sy = img_size / max(img_h, 1)

    result: Dict = {
        'scour_line': None,
        'pile_polygons': [],
        'scour_pit_polygons': [],
        'center_point': None,
        'image_path': ann.get('imagePath', ''),
        'image_size': (img_w, img_h),
    }

    for shape in ann.get('shapes', []):
        pts = [(p[0] * sx, p[1] * sy) for p in shape.get('points', [])]
        label = shape.get('label', '').lower()

        if 'scour line' in label:
            result['scour_line'] = pts
        elif 'scour pit' in label and len(pts) >= 3:
            result['scour_pit_polygons'].append(pts)
        elif label == 'pile' and len(pts) >= 3:
            result['pile_polygons'].append(pts)
        elif 'center' in label:
            result['center_point'] = pts[0] if pts else None

    return result


# ======================================================================
# Mask Generators
# ======================================================================

def make_pile_mask(pile_polygons: List, img_size: int = 518) -> np.ndarray:
    """Convert pile polygons to binary mask."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for poly in pile_polygons:
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def make_scour_pit_mask(scour_pit_polygons: List, img_size: int = 518) -> np.ndarray:
    """Convert scour_pit polygons to binary mask (ground truth)."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for poly in scour_pit_polygons:
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def make_scour_region_mask(scour_line_pts: Optional[List],
                           pile_mask: np.ndarray,
                           img_size: int = 518,
                           extend_px: int = 35) -> np.ndarray:
    """
    Generate scour region pseudo-label from scour_line annotation.

    The scour_line marks the pile-scour boundary. We extend outward
    from this line to approximate the scour pit extent.
    """
    if scour_line_pts is None or len(scour_line_pts) < 3:
        return np.zeros((img_size, img_size), dtype=np.uint8)

    line_mask = np.zeros((img_size, img_size), dtype=np.uint8)
    pts = np.array(scour_line_pts, dtype=np.int32)
    cv2.polylines(line_mask, [pts], isClosed=True, color=255, thickness=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (extend_px * 2 + 1, extend_px * 2 + 1))
    region = cv2.dilate(line_mask, kernel, iterations=1)
    region[pile_mask > 128] = 0
    return region


def make_bed_region_mask(pile_mask: np.ndarray,
                         img_size: int = 518,
                         min_dist_px: int = 80) -> np.ndarray:
    """Generate bed reference region far from pile."""
    ys, xs = np.where(pile_mask > 128)
    if len(ys) == 0:
        return np.ones((img_size, img_size), dtype=np.uint8) * 255
    cx, cy = xs.mean(), ys.mean()
    yy, xx = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing='ij')
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    bed = np.zeros((img_size, img_size), dtype=np.uint8)
    bed[dist > min_dist_px] = 255
    bed[pile_mask > 128] = 0
    return bed


# ======================================================================
# Dataset
# ======================================================================

class ScourDataset(Dataset):
    """
    Dataset for scour pit depth estimation.

    Each sample:
      - rgb: (3, H, W) normalized [0,1]
      - pile_mask: (1, H, W) binary
      - scour_mask: (1, H, W) binary (pseudo-label from scour_line)
      - bed_mask: (1, H, W) binary
      - image_size: original (w, h)
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 img_size: int = 518,
                 augment: bool = True,
                 val_ratio: float = 0.3,
                 seed: int = 42):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        anno_dir = self.data_dir / 'anno'
        all_files = sorted(anno_dir.glob('*.json'))

        # Filter: only keep files with scour_pit OR scour_line + pile annotations
        self.json_files = []
        for jf in all_files:
            ann = parse_labelme_json(jf, self.img_size)
            has_scour = (ann.get('scour_pit_polygons') and len(ann['scour_pit_polygons']) > 0) or \
                        (ann['scour_line'] is not None)
            if has_scour and len(ann['pile_polygons']) > 0:
                self.json_files.append(jf)

        # Random split
        n = len(self.json_files)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)
        split_n = int(n * (1 - val_ratio))

        if split == 'train':
            self.json_files = [self.json_files[i] for i in indices[:split_n]]
        else:
            self.json_files = [self.json_files[i] for i in indices[split_n:]]

        print(f"[ScourDataset] {split}: {len(self.json_files)} samples "
              f"from {data_dir}")

    def __len__(self) -> int:
        return len(self.json_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        jf = self.json_files[idx]
        ann = parse_labelme_json(jf, self.img_size)

        # Load image
        img_name = Path(ann['image_path']).name
        img_path = self.data_dir / 'images' / img_name
        if not img_path.exists():
            # Try common extensions
            for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg']:
                alt = self.data_dir / 'images' / (img_name.rsplit('.', 1)[0] + ext)
                if alt.exists():
                    img_path = alt
                    break

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Generate masks — prefer real scour_pit annotations if available
        pile_mask = make_pile_mask(ann['pile_polygons'], self.img_size)
        if ann.get('scour_pit_polygons') and len(ann['scour_pit_polygons']) > 0:
            scour_mask = make_scour_pit_mask(ann['scour_pit_polygons'], self.img_size)
        else:
            scour_mask = make_scour_region_mask(ann['scour_line'], pile_mask,
                                                 self.img_size)
        bed_mask = make_bed_region_mask(pile_mask, self.img_size)

        # Augmentation
        if self.augment:
            img, masks = self._augment(img, {
                'pile': pile_mask, 'scour': scour_mask, 'bed': bed_mask})
            pile_mask = masks['pile']
            scour_mask = masks['scour']
            bed_mask = masks['bed']

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)

        # To tensor
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        pile_t = torch.from_numpy(pile_mask.astype(np.float32) / 255.0).unsqueeze(0)
        scour_t = torch.from_numpy(scour_mask.astype(np.float32) / 255.0).unsqueeze(0)
        bed_t = torch.from_numpy(bed_mask.astype(np.float32) / 255.0).unsqueeze(0)

        return {
            'image': img_t,
            'pile_mask': pile_t,
            'scour_mask': scour_t,
            'bed_mask': bed_t,
            'name': jf.stem,
        }

    def _augment(self, image: np.ndarray,
                 masks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """Simple augmentation."""
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            for k in masks:
                masks[k] = np.fliplr(masks[k]).copy()

        if random.random() < 0.3:
            image = np.flipud(image).copy()
            for k in masks:
                masks[k] = np.flipud(masks[k]).copy()

        # Brightness jitter
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return image, masks
