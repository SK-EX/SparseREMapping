#!/usr/bin/env python3
"""
冲刷坑体积计算 — Delaunay 三角化 + 柱体法
==========================================
用法: python scour_volume.py --input output/aligned_ply/combined_aligned.ply --scale 90
"""

import numpy as np, argparse, struct, sys
from pathlib import Path
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scour_slope import read_ply


def compute_volume_and_area(xyz, bed_level=None):
    """
    Delaunay 三角化 XY 投影, 计算床面以下的体积 + 3D 表面积.
    bed_level=None 时自动检测 (取 Z 的 P95 作为床面)
    返回 (volume, surface_area_3d, xy_area)
    """
    if bed_level is None:
        bed_level = np.percentile(xyz[:, 2], 95)
        print(f"  Auto bed level (P95): {bed_level:.4f}")

    below = xyz[:, 2] < bed_level
    pts = xyz[below]
    if len(pts) < 10:
        print(f"  Not enough points below bed!")
        return 0, 0, 0

    print(f"  Points below bed: {len(pts):,} / {len(xyz):,}")

    xy = pts[:, :2]
    tri = Delaunay(xy)
    print(f"  Triangles: {len(tri.simplices):,}")

    z = pts[:, 2]
    dz = bed_level - z

    total_vol = 0.0
    total_area_3d = 0.0
    total_area_xy = 0.0

    for simplex in tri.simplices:
        p = pts[simplex]  # 3×3: (x,y,z)
        a_xy = 0.5 * abs(np.cross(p[1,:2] - p[0,:2], p[2,:2] - p[0,:2]))

        # 3D 三角形面积 (海伦公式或向量法)
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        a_3d = 0.5 * np.linalg.norm(np.cross(v1, v2))

        # 柱体体积
        d = dz[simplex]
        vol = a_xy * (d[0] + d[1] + d[2]) / 3.0

        total_vol += vol
        total_area_3d += a_3d
        total_area_xy += a_xy

    return total_vol, total_area_3d, total_area_xy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--bed-level', type=float, default=None,
                        help='Bed Z level (auto-detect P95 if not set)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor (e.g. 90 for mm)')
    args = parser.parse_args()

    xyz, _ = read_ply(args.input)
    xyz = xyz * args.scale
    bed = args.bed_level * args.scale if args.bed_level is not None else None
    print(f"Loaded {len(xyz):,} pts (scale={args.scale})")

    vol, area_3d, area_xy = compute_volume_and_area(xyz, bed)

    print(f"\nScour volume: {vol:.4f} cubic units")
    print(f"Surface area (3D): {area_3d:.4f} square units")
    print(f"XY projected area: {area_xy:.4f} square units")

    if args.scale > 1:
        print(f"\n  Volume:")
        print(f"    = {vol:.2f} mm³")
        print(f"    = {vol/1e6:.4f} L")
        print(f"    = {vol/1e9:.8f} m³")
        print(f"  Surface area (3D):")
        print(f"    = {area_3d:.2f} mm²")
        print(f"    = {area_3d/1e6:.6f} m²")


if __name__ == '__main__':
    main()
