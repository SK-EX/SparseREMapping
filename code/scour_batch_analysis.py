#!/usr/bin/env python3
"""
批量冲刷坑体积+表面积分析
==========================
遍历 output/aligned_ply/ 下所有工况, 合并单视角 PLY, 计算体积+表面积, 生成对比图

用法:
  python scour_batch_analysis.py --scale 90
"""

import os, sys, argparse, struct
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scour_slope import read_ply

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


def compute_volume_and_area(xyz, bed_level=None):
    """Delaunay 三角化计算体积+表面积"""
    from scipy.spatial import Delaunay
    if bed_level is None:
        bed_level = np.percentile(xyz[:, 2], 95)
    below = xyz[:, 2] < bed_level
    pts = xyz[below]
    if len(pts) < 10:
        return 0, 0, 0, bed_level
    xy = pts[:, :2]
    tri = Delaunay(xy)
    z = pts[:, 2]
    dz = bed_level - z
    vol, a3d, axy = 0.0, 0.0, 0.0
    for s in tri.simplices:
        p = pts[s]
        v1, v2 = p[1]-p[0], p[2]-p[0]
        axy += 0.5 * abs(np.cross(v1[:2], v2[:2]))
        a3d += 0.5 * np.linalg.norm(np.cross(v1, v2))
        d = dz[s]
        vol += 0.5*abs(np.cross(v1[:2], v2[:2])) * (d[0]+d[1]+d[2])/3.0
    return vol, a3d, axy, bed_level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='output/aligned_ply')
    parser.add_argument('--output-dir', type=str, default='output/batch_analysis')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--bed-level', type=float, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有工况
    conditions = defaultdict(list)
    for ply_path in sorted(input_dir.rglob('*_aligned.ply')):
        if 'combined' in ply_path.name:
            continue
        cond_name = ply_path.parent.name
        conditions[cond_name].append(ply_path)

    print(f"Found {len(conditions)} conditions, {sum(len(v) for v in conditions.values())} PLY files")

    results = []
    for cond_name in sorted(conditions.keys()):
        print(f"\n─── {cond_name} ───")
        # 合并所有视角
        xyz_parts = []
        for pp in conditions[cond_name]:
            x, _ = read_ply(str(pp))
            if len(x) > 5:
                xyz_parts.append(x * args.scale)
        if not xyz_parts:
            print(f"  No valid PLYs, skipping")
            continue
        xyz_all = np.concatenate(xyz_parts, axis=0)
        print(f"  Merged: {len(xyz_all):,} pts from {len(xyz_parts)} views")

        # 计算
        bed = args.bed_level * args.scale if args.bed_level is not None else None
        vol, a3d, axy, bed_auto = compute_volume_and_area(xyz_all, bed)
        results.append({
            'name': cond_name,
            'vol': vol,
            'area_3d': a3d,
            'area_xy': axy,
            'bed': bed_auto,
            'n_views': len(xyz_parts),
            'n_pts': len(xyz_all),
            'depth': bed_auto - xyz_all[xyz_all[:, 2] < bed_auto, 2].min() if (xyz_all[:, 2] < bed_auto).any() else 0,
        })
        print(f"  Volume: {vol:.4f}  Area(3D): {a3d:.4f}  Depth: {results[-1]['depth']:.4f}")

    if not results:
        print("No results!")
        return

    # ── 保存数值表 ──
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f"{'Condition':<45} {'Vol':>12} {'Area3D':>12} {'XYArea':>12} {'Depth':>10} {'Views':>6} {'Pts':>10}\n")
        f.write("-" * 110 + "\n")
        for r in results:
            f.write(f"{r['name']:<45} {r['vol']:12.4f} {r['area_3d']:12.4f} {r['area_xy']:12.4f} {r['depth']:10.4f} {r['n_views']:6d} {r['n_pts']:10,d}\n")

    # ── 对比柱状图 ──
    names = [r['name'][:30] for r in results]
    vols = [r['vol'] for r in results]
    areas = [r['area_3d'] for r in results]
    depths = [r['depth'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))

    ax = axes[0]
    bars = ax.bar(range(len(results)), vols, color=colors, edgecolor='black')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Volume (VGGT units)' if args.scale == 1 else 'Volume (mm³)', fontsize=12)
    ax.set_title('Scour Volume', fontsize=14, fontweight='bold')
    for i, v in enumerate(vols):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    ax.bar(range(len(results)), areas, color=colors, edgecolor='black')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Surface Area (VGGT units)' if args.scale == 1 else 'Area (mm²)', fontsize=12)
    ax.set_title('3D Surface Area', fontsize=14, fontweight='bold')

    ax = axes[2]
    ax.bar(range(len(results)), depths, color=colors, edgecolor='black')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Depth (VGGT units)' if args.scale == 1 else 'Depth (mm)', fontsize=12)
    ax.set_title('Max Scour Depth', fontsize=14, fontweight='bold')

    fig.suptitle('Scour Pit Analysis — Batch Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  results.txt — numerical table")
    print(f"  comparison.png — bar charts")


if __name__ == '__main__':
    main()
