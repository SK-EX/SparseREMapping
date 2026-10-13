#!/usr/bin/env python3
"""冲刷坑倾角计算 (正视图, -Y方向, XZ平面)"""

import numpy as np, argparse, struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


def read_ply(filepath):
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
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.uint8) if rgb else None


def envelope(pts, n_bins=100):
    x_a = pts[:, 0]; z_a = pts[:, 2]  # front view: XZ plane
    x_min, x_max = x_a.min(), x_a.max()
    bins = np.linspace(x_min, x_max, n_bins + 1)
    x_env = (bins[:-1] + bins[1:]) / 2
    z_env = np.full(n_bins, np.nan)
    for j in range(n_bins):
        bm = (x_a >= bins[j]) & (x_a < bins[j + 1])
        if bm.sum() > 3: z_env[j] = np.min(z_a[bm])
    v = ~np.isnan(z_env)
    zv = z_env[v]; xv = x_env[v]
    for _ in range(3):
        window = max(7, len(zv) // 8)
        z_sm = np.convolve(zv, np.ones(window)/window, mode='same')
        zv = np.minimum(zv, z_sm)
    z_env[v] = zv
    return x_env[v], z_env[v]


def find_critical_point(xv, zv, outward_sign):
    di = np.argmin(zv)
    dx, dz = xv[di], zv[di]
    if outward_sign > 0: outer_mask = (xv >= dx)
    else: outer_mask = (xv <= dx)
    xo = xv[outer_mask]; zo = zv[outer_mask]
    if len(xo) < 5:
        far_x = xo[0] if outward_sign < 0 else xo[-1]
        return abs(far_x - dx), far_x
    local_slopes = []
    for i in range(1, len(xo)):
        s = abs((zo[i] - zo[i-1]) / max(abs(xo[i] - xo[i-1]), 1e-8))
        local_slopes.append(s)
    local_slopes = np.array(local_slopes)
    if len(local_slopes) < 3:
        far_x = xo[0] if outward_sign < 0 else xo[-1]
        return abs(far_x - dx), far_x
    window = max(3, len(local_slopes) // 15)
    smoothed = np.convolve(local_slopes, np.ones(window)/window, mode='same')
    n_near = max(3, len(smoothed)//10)
    if outward_sign > 0:
        near_slopes = smoothed[:n_near]
        scan = range(n_near, len(smoothed))
    else:
        near_slopes = smoothed[-n_near:]
        scan = range(len(smoothed)-1-n_near, -1, -1)
    if outward_sign < 0:
        threshold = np.mean(near_slopes) * 0.15
    else:
        threshold = np.mean(near_slopes) * 0.16
    for i in scan:
        if smoothed[i] < threshold:
            ci = min(i + 1, len(xo) - 1)
            return abs(xo[ci] - dx), xo[ci]
    far_x = xo[0] if outward_sign < 0 else xo[-1]
    return abs(far_x - dx), far_x


def tangent_from_deepest(x_env, z_env, bed_level, outward_sign):
    outer = x_env * outward_sign >= x_env[np.argmin(z_env)] * outward_sign
    xv = x_env[outer]; zv = z_env[outer]
    if len(xv) < 3: return None
    di = np.argmin(zv)
    dx, dz = xv[di], zv[di]
    h = bed_level - dz
    result = find_critical_point(xv, zv, outward_sign)
    if result: phi, crit_x = result
    else:
        far_x = xv[0] if outward_sign < 0 else xv[-1]
        phi = abs(far_x - dx); crit_x = far_x
    tz = np.interp(crit_x, xv, zv)
    slope = (tz - dz) / max(abs(crit_x - dx), 1e-8)
    th = np.degrees(np.arctan(np.abs(slope)))
    return dx, dz, h, th, crit_x, tz, phi, crit_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='scour_slope_profile.png')
    parser.add_argument('--bed-level', type=float, default=0.0)
    parser.add_argument('--pile-gap', type=float, default=0.02)
    parser.add_argument('--scale', type=float, default=0.7,
                        help='Global coordinate scale factor (e.g. 90 for mm conversion)')
    args = parser.parse_args()

    xyz, _ = read_ply(args.input)
    xyz = xyz * args.scale
    args.bed_level = args.bed_level * args.scale
    print(f"Loaded {len(xyz):,} pts (scale={args.scale})")

    xl, zl = envelope(xyz[xyz[:, 0] < -args.pile_gap])
    xr, zr = envelope(xyz[xyz[:, 0] > args.pile_gap])
    L = tangent_from_deepest(xl, zl, args.bed_level, outward_sign=-1)
    R = tangent_from_deepest(xr, zr, args.bed_level, outward_sign=+1)

    if L: print(f"Left  (θ3): h={L[2]:.4f}  θ3={L[3]:.1f}°  φ3={L[6]:.4f}")
    if R: print(f"Right (θ4): h={R[2]:.4f}  θ4={R[3]:.1f}°  φ4={R[6]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(xyz[:, 0], xyz[:, 2], s=1, c='black', alpha=0.4)
    ax.axhline(y=args.bed_level, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.axvline(x=0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.plot(0, 0, 'r^', ms=14)

    for side, data, c_w, c_t, label in [
        ('left', L, 'blue', 'cyan', '迎流面左侧'),
        ('right', R, 'green', 'lime', '迎流面右侧')]:
        if data is None: continue
        dx, dz, h, th, tx, tz, phi, crit_x = data
        sgn = -1 if side == 'left' else 1

        mask = (xl * sgn >= dx * sgn) if side == 'left' else (xr * sgn >= dx * sgn)
        x_env_side = xl if side == 'left' else xr
        z_env_side = zl if side == 'left' else zr
        ax.plot(x_env_side[mask], z_env_side[mask], '-', color=c_w, lw=2.5, label=label)
        ax.plot(dx, dz, '*', color=c_w, ms=22, zorder=5)

        ax.annotate('', xy=(dx, dz), xytext=(dx, args.bed_level),
                    arrowprops=dict(arrowstyle='<->', color=c_w, lw=3))
        h_lab = dx + (-0.08 if side == 'left' else 0.005)
        ax.text(h_lab, (dz + args.bed_level)/2, 'h=' + f'{h:.3f}',
                fontsize=13, color=c_w, fontweight='bold')

        ax.plot([tx, dx], [tz, dz], '-', color=c_t, lw=3)
        lx = (tx + dx)/2 + (-0.06 if side == 'left' else 0.06)
        lz = (tz + dz)/2 + 0.015 - 0.04
        ax.text(lx, lz, 'θ' + ('3' if side == 'left' else '4') + '=' + f'{th:.1f}' + chr(176),
                fontsize=15, color=c_t, fontweight='bold', ha='center')

        ax.plot(crit_x, np.interp(crit_x, x_env_side, z_env_side), 's', color=c_w, ms=12)
        plx = (dx + crit_x)/2 + (-0.03 if side == 'left' else 0.03)
        ax.annotate('', xy=(crit_x, args.bed_level), xytext=(dx, args.bed_level),
                    arrowprops=dict(arrowstyle='<->', color=c_w, lw=2))
        ax.text(plx, args.bed_level + 0.01, 'φ' + ('3' if side == 'left' else '4') + '=' + f'{phi:.3f}',
                fontsize=12, color=c_w, fontweight='bold', ha='center')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='桩心'),
        Line2D([0], [0], color='red', ls='--', lw=1.5, label='床面 (Z=0)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=12, label='桩左侧冲刷最低点'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=12, label='桩右侧冲刷最低点'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='左临界点'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='右临界点'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower left')
    ax.set_xlabel('X/m', fontsize=13); ax.set_ylabel('Z/m', fontsize=13)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    plt.tight_layout(); fig.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
