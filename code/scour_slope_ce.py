#!/usr/bin/env python3
"""冲刷坑倾角计算 (侧视图, -X方向, YZ平面)"""

import numpy as np, argparse, struct
import matplotlib
from PIL.ImageOps import scale
#python scour_slope_ce.py --input output/aligned_ply/vggt_scour_sin_140h6t1_6/ply/points3D_dense_no_red2_uniform.ply --output output/aligned_ply/vggt_scour_sin_140h6t1_6/ply/scour_slope_ce.png
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
# 中文字体
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


def read_ply(filepath,point_number):
    xyz, rgb = [], []
    with open(filepath, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            header.append(line)
            if line == 'end_header': break
        n_verts = 0; props = []; is_ascii = False; in_vertex = True
        for h in header:
            if 'ascii' in h: is_ascii = True
            if h.startswith('element vertex'): n_verts = int(h.split()[-1]); in_vertex = True
            elif h.startswith('element face'): in_vertex = False
            elif h.startswith('property') and in_vertex: props.append(h.split()[-1])
        x_idx = props.index('x') if 'x' in props else 0
        ri = props.index('red') if 'red' in props else (props.index('r') if 'r' in props else None)
        for i in range(n_verts):
            if is_ascii:
                v = f.readline().decode('utf-8', errors='ignore').strip().split()
                xyz.append([float(v[x_idx]), float(v[x_idx+1]), float(v[x_idx+2])])
                if ri is not None: rgb.append([int(v[ri]), int(v[ri+1]), int(v[ri+2])])
            else:
                vals = []
                for p in props:
                    if p in ('red','green','blue','r','g','b'): vals.append(struct.unpack('B', f.read(1))[0])
                    elif p in ('alpha',): f.read(1)
                    else: vals.append(struct.unpack('f', f.read(4))[0])
                xyz.append([vals[x_idx], vals[x_idx+1], vals[x_idx+2]])
                if ri is not None: rgb.append([vals[ri], vals[ri+1], vals[ri+2]])
            #xyz点云随机降采样
        xyz = np.array(xyz, dtype=np.float32)
        rbg = np.array(rgb, dtype=np.uint8) if rgb else None
        total = len(xyz)
        sample_num = int(point_number * total)
        sample_num = min(total, sample_num)
        if sample_num < total:
            indices = np.random.choice(total, sample_num, replace=False)
            xyz = xyz[indices]
    return xyz, rgb


def envelope(pts, n_bins=100):
    y_a = pts[:, 1]; z_a = pts[:, 2]
    y_min, y_max = y_a.min(), y_a.max()
    bins = np.linspace(y_min, y_max, n_bins + 1)
    y_env = (bins[:-1] + bins[1:]) / 2
    z_env = np.full(n_bins, np.nan)
    for j in range(n_bins):
        bm = (y_a >= bins[j]) & (y_a < bins[j + 1])
        if bm.sum() > 3: z_env[j] = np.min(z_a[bm])  # 取min严格贴底, 不被重影拉高
    v = ~np.isnan(z_env)
    # 平滑: 滑动中值压制跳跃, 窗宽加大, 多次迭代
    zv = z_env[v]; yv = y_env[v]
    for _ in range(3):
        window = max(7, len(zv) // 8)
        z_sm = np.convolve(zv, np.ones(window)/window, mode='same')
        zv = np.minimum(zv, z_sm)  # 只向下修正
    z_env[v] = zv
    return y_env[v], z_env[v]


def find_critical_point(yv, zv, outward_sign):
    di = np.argmin(zv)
    dy, dz = yv[di], zv[di]
    if outward_sign > 0: outer_mask = (yv >= dy)
    else: outer_mask = (yv <= dy)
    yo = yv[outer_mask]; zo = zv[outer_mask]
    if len(yo) < 5:
        far_y = yo[0] if outward_sign < 0 else yo[-1]
        return abs(far_y - dy), far_y
    local_slopes = []
    for i in range(1, len(yo)):
        s = abs((zo[i] - zo[i-1]) / max(abs(yo[i] - yo[i-1]), 1e-8))
        local_slopes.append(s)
    local_slopes = np.array(local_slopes)
    if len(local_slopes) < 3:
        far_y = yo[0] if outward_sign < 0 else yo[-1]
        return abs(far_y - dy), far_y
    window = max(3, len(local_slopes) // 15)
    smoothed = np.convolve(local_slopes, np.ones(window)/window, mode='same')
    # 从最深向外扫描: 找斜率首次降到近深深度的斜率 * 0.3 以下
    # 近深深度的斜率 = deepest端几个斜率的均值
    n_near = max(3, len(smoothed)//10)
    if outward_sign > 0:
        near_slopes = smoothed[:n_near]   # 右侧: deepest在低索引
        scan = range(n_near, len(smoothed))
    else:
        near_slopes = smoothed[-n_near:]  # 左侧: deepest在高索引
        scan = range(len(smoothed)-1-n_near, -1, -1)
    # 桩前(左): 阈值更大→临界点更近最深点→φ更小; 桩后(右): 阈值更小→临界点更远→φ更大
    if outward_sign < 0:
        threshold = np.mean(near_slopes) * 0.3# 桩前: 阈值↑ → 更快触发 → φ小
    else:
        threshold = np.mean(near_slopes) * 0.1# 桩后: 阈值↓ → 更晚触发 → φ大
    for i in scan:
        if smoothed[i] < threshold:
            ci = min(i + 1, len(yo) - 1)
            return abs(yo[ci] - dy), yo[ci]
    far_y = yo[0] if outward_sign < 0 else yo[-1]
    return abs(far_y - dy), far_y


def tangent_from_deepest(y_env, z_env, bed_level, outward_sign, override_dy=None):
    outer = y_env * outward_sign >= y_env[np.argmin(z_env)] * outward_sign
    yv = y_env[outer]; zv = z_env[outer]
    if len(yv) < 3: return None

    if override_dy is not None:
        # 使用用户指定的最深点Y坐标, 从包络线上插值得到对应Z
        dy = override_dy
        dz = np.interp(dy, yv, zv)
        print(f"  [手动] 最深点 Y={dy:.4f} (override), Z={dz:.4f}")
    else:
        di = np.argmin(zv)
        dy, dz = yv[di], zv[di]
    h = bed_level - dz

    # 先找临界点
    result = find_critical_point(yv, zv, outward_sign)
    if result:
        phi, crit_y = result
    else:
        far_y = yv[0] if outward_sign < 0 else yv[-1]
        phi = abs(far_y - dy); crit_y = far_y

    # 切点 = 临界点 (左右一致)
    tz = np.interp(crit_y, yv, zv)
    slope = (tz - dz) / max(abs(crit_y - dy), 1e-8)
    th = np.degrees(np.arctan(np.abs(slope)))
    ty, tz = crit_y, tz

    return dy, dz, h, th, ty, tz, phi, crit_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='scour_slope_ce_profile.png')
    parser.add_argument('--bed-level', type=float, default=0.0)
    parser.add_argument('--pile-gap', type=float, default=0.02)
    parser.add_argument('--scale', type=float, default=1.5,
                        help='Global coordinate scale factor')
    parser.add_argument('--left-dy', type=float, default=-0.05,
                        help='手动指定左侧(迎流面)最深点的Y坐标 (覆盖自动检测)')
    parser.add_argument('--point_number', type=float,default=1)
    args = parser.parse_args()

    xyz, _ = read_ply(args.input, args.point_number)
    xyz = xyz * args.scale

    args.bed_level = args.bed_level * args.scale

    print(f"Loaded {len(xyz):,} pts (scale={args.scale})")

    # YZ 平面: 左=Y负, 右=Y正
    yl, zl = envelope(xyz[xyz[:, 1] < -args.pile_gap])
    yr, zr = envelope(xyz[xyz[:, 1] > args.pile_gap])
    L = tangent_from_deepest(yl, zl, args.bed_level, outward_sign=-1, override_dy=args.left_dy)
    R = tangent_from_deepest(yr, zr, args.bed_level, outward_sign=+1)

    if L: print(f"Left  (θ1): h={L[2]:.4f}  θ3={L[3]:.1f}°  φ3={L[6]:.4f}")
    if R: print(f"Right (θ2): h={R[2]:.4f}  θ4={R[3]:.1f}°  φ4={R[6]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(xyz[:, 1], xyz[:, 2], s=1, c='black', alpha=0.4)
    ax.axhline(y=args.bed_level, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.axvline(x=0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.plot(0, 0, 'r^', ms=14)

    for side, data, c_w, c_t, label in [
        ('left', L, 'blue', 'cyan', '迎流面'),
        ('right', R, 'green', 'lime', '背流面')]:
        if data is None: continue
        dy, dz, h, th, ty, tz, phi, crit_y = data
        sgn = -1 if side == 'left' else 1

        mask = (yl * sgn >= dy * sgn) if side == 'left' else (yr * sgn >= dy * sgn)
        y_env_side = yl if side == 'left' else yr
        z_env_side = zl if side == 'left' else zr
        ax.plot(y_env_side[mask], z_env_side[mask], '-', color=c_w, lw=2.5, label=label)
        ax.plot(dy, dz, '*', color=c_w, ms=22, zorder=5)

        ax.annotate('', xy=(dy, dz), xytext=(dy, args.bed_level),
                    arrowprops=dict(arrowstyle='<->', color=c_w, lw=3))
        h_lab = dy + (-0.06 if side == 'left' else 0.005)
        ax.text(h_lab, (dz + args.bed_level)/2, 'h=' + f'{h:.3f}',
                fontsize=13, color=c_w, fontweight='bold')

        ax.plot([ty, dy], [tz, dz], '-', color=c_t, lw=3)
        lx = (ty + dy)/2 + (-0.02 if side == 'left' else 0.04)
        lz = (tz + dz)/2 + 0.015 - 0.04
        ax.text(lx, lz, 'θ' + ('1' if side == 'left' else '2') + '=' + f'{th:.1f}' + chr(176),
                fontsize=15, color=c_t, fontweight='bold', ha='center')
        #手动设置ylim上限
        ax.set_ylim(ymin=-0.12, ymax=0.15)
        ax.plot(crit_y, np.interp(crit_y, y_env_side, z_env_side), 's', color=c_w, ms=12)
        plx = (dy + crit_y)/2 + (-0.03 if side == 'left' else 0.03)
        ax.annotate('', xy=(crit_y, args.bed_level), xytext=(dy, args.bed_level),
                    arrowprops=dict(arrowstyle='<->', color=c_w, lw=2))
        ax.text(plx, args.bed_level + 0.01, 'φ' + ('1' if side == 'left' else '2') + '=' + f'{phi:.3f}',
                fontsize=12, color=c_w, fontweight='bold', ha='center')

    # 统一图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='桩心'),
        Line2D([0], [0], color='red', ls='--', lw=1.5, label='床面 (Z=0)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=12, label='迎流面冲刷最低点'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=12, label='背流面冲刷最低点'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='前临界点'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='后临界点'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower left')

    ax.set_xlabel('Y/m', fontsize=13); ax.set_ylabel('Z/m', fontsize=13)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    plt.tight_layout(); fig.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
