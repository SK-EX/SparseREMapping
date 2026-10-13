#!/usr/bin/env python3
"""
PLY 点云对齐: 全局统一变换 — 桩中心→原点, 床面法向→+Z
====================================================
从 combined.ply 计算一个全局 4×4 变换矩阵，
然后对所有 per-view PLY 应用同一个矩阵，保证视图间一致。

用法:
  python align_ply.py
"""

import os, sys, argparse, struct
import numpy as np
from pathlib import Path


def read_ply(filepath):
    with open(filepath, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            header.append(line)
            if line == 'end_header':
                break
        n_verts = 0
        props = []
        is_ascii = False
        for h in header:
            if h.startswith('format ascii'):
                is_ascii = True
            if h.startswith('element vertex'):
                n_verts = int(h.split()[-1])
            elif h.startswith('property'):
                props.append(h.split()[-1])
        xyz = np.zeros((n_verts, 3), dtype=np.float32)
        rgb = np.zeros((n_verts, 3), dtype=np.uint8)
        x_idx = props.index('x') if 'x' in props else 0
        ri = props.index('red') if 'red' in props else None
        if is_ascii:
            for i in range(n_verts):
                vals = f.readline().decode('utf-8', errors='ignore').strip().split()
                xyz[i] = [float(vals[x_idx]), float(vals[x_idx+1]), float(vals[x_idx+2])]
                if ri is not None:
                    rgb[i] = [int(vals[ri]), int(vals[ri+1]), int(vals[ri+2])]
            return xyz, rgb
        for i in range(n_verts):
            vals = []
            for p in props:
                if p in ('red', 'green', 'blue'):
                    vals.append(struct.unpack('B', f.read(1))[0])
                else:
                    vals.append(struct.unpack('f', f.read(4))[0])
            xyz[i] = [vals[x_idx], vals[x_idx+1], vals[x_idx+2]]
            ri = props.index('red') if 'red' in props else None
            if ri is not None:
                rgb[i] = [vals[ri], vals[ri+1], vals[ri+2]]
    return xyz, rgb


def write_ply(filepath, xyz, rgb):
    n = len(xyz)
    with open(filepath, 'wb') as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        for i in range(n):
            f.write(xyz[i].astype(np.float32).tobytes())
            f.write(rgb[i].astype(np.uint8).tobytes())


def compute_global_transform(xyz_all):
    """
    从全部点云计算统一变换矩阵 (R, t)
    1. 质心 → 原点
    2. 床面法向 (PCA最小主成分) → +Z
    """
    centroid = xyz_all.mean(axis=0)
    xyz_c = xyz_all - centroid

    cov = np.cov(xyz_c.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # 升序: λ0<λ1<λ2

    # 最小特征值对应的特征向量 = 床面法向量
    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal = -normal

    # Rodrigues: rotate normal → (0,0,1)
    z_target = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(normal, z_target)
    nrm = np.linalg.norm(rot_axis)

    if nrm < 1e-10:
        R = np.eye(3)
    else:
        rot_axis /= nrm
        theta = np.arccos(np.clip(np.dot(normal, z_target), -1.0, 1.0))
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                       [rot_axis[2], 0, -rot_axis[0]],
                       [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    # 构建 4×4 变换: 先平移后旋转
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ centroid

    return T, centroid, normal


def apply_transform(xyz, T):
    """对 N×3 点云应用 4×4 变换矩阵"""
    ones = np.ones((len(xyz), 1), dtype=np.float32)
    xyz_h = np.hstack([xyz, ones])
    xyz_t = (T @ xyz_h.T).T[:, :3]
    return xyz_t


def main():
    parser = argparse.ArgumentParser(description='Global align scour pit PLYs')
    parser.add_argument('--input-dir', type=str,
                        default='output/vggt_scour_random147_4_h8t1_4/ply')
    parser.add_argument('--output-dir', type=str, default='output/aligned_ply/vggt_scour_random147_4_h8t1_4')
    parser.add_argument('--ref', type=str, default='combined.ply',
                        help='Reference PLY to compute global transform')
    parser.add_argument('--files', type=str, default=None,
                        help='Comma-separated .ply names to transform')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 确定文件列表 + 获取参考点云 ──
    if args.files:
        ply_files = [input_dir / f.strip() for f in args.files.split(',')]
    else:
        ply_files = sorted([p for p in input_dir.glob('*.ply')
                            if 'combined' not in p.name
                            and 'acc_' not in p.name
                            and 'aligned' not in p.name])

    # 参考点云: 优先用 --ref, 否则自动合并所有文件
    ref_path = input_dir / args.ref
    if ref_path.exists():
        print(f"Computing global transform from: {ref_path.name}")
        xyz_ref, _ = read_ply(str(ref_path))
    else:
        print(f"Reference '{args.ref}' not found, merging {len(ply_files)} PLYs as reference...")
        xyz_parts = []
        for fp in ply_files:
            if fp.exists():
                x, _ = read_ply(str(fp))
                xyz_parts.append(x)
                print(f"  + {fp.name}: {len(x):,} pts")
        if not xyz_parts:
            print("No PLY files found!"); return
        xyz_ref = np.concatenate(xyz_parts, axis=0)
        print(f"  Merged reference: {len(xyz_ref):,} pts")
    # 过滤离群点 (Z方向 5×MAD)
    z_med = np.median(xyz_ref[:, 2])
    z_mad = np.median(np.abs(xyz_ref[:, 2] - z_med))
    inlier = np.abs(xyz_ref[:, 2] - z_med) < 5 * z_mad
    xyz_clean = xyz_ref[inlier]
    print(f"  {len(xyz_ref):,} pts → {len(xyz_clean):,} inliers")

    T, centroid, normal = compute_global_transform(xyz_clean)
    print(f"  Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
    print(f"  Bed normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
    print(f"  Transform matrix (4×4):")
    for row in T:
        print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}]")

    # 保存变换矩阵
    np.savetxt(output_dir / 'transform_matrix.txt', T, fmt='%.8f',
               header='Global 4x4 transform: R[:3,:3] | t[:3]')

    # ── 2. 对齐参考点云 ──
    xyz_ref_aligned = apply_transform(xyz_ref, T)
    print(f"\n  reference_aligned: {len(xyz_ref_aligned):,} pts")

    # ── 3. 对每个 per-view PLY 应用相同变换 ──
    print(f"\nApplying same transform to {len(ply_files)} views:")
    for ply_path in ply_files:
        if not ply_path.exists():
            print(f"  [SKIP] {ply_path.name} — not found")
            continue

        xyz, rgb = read_ply(str(ply_path))
        if len(xyz) < 5:
            continue

        xyz_t = apply_transform(xyz, T)
        out_path = output_dir / f'{ply_path.stem}_aligned.ply'
        write_ply(str(out_path), xyz_t, rgb)
        z_range = xyz_t[:, 2].ptp()
        z_mean = xyz_t[:, 2].mean()
        print(f"  {ply_path.name} → {out_path.name}  ({len(xyz_t):,} pts, z∈[{z_mean:.3f}±{z_range/2:.3f}])")

    print(f"\nDone! → {output_dir}/")
    print(f"All PLYs share the same transform matrix.")


if __name__ == '__main__':
    main()
