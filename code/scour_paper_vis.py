#!/usr/bin/env python3
"""
冲刷坑论文级可视化 — 3D表面 + 深度热力图
=========================================
用法: python scour_paper_vis.py --scale 90
"""

import os, sys, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scour_slope import read_ply
from scour_volume import compute_volume_and_area  # reuse

# 字体: 英文/数字 Times New Roman, 中文宋体 (Noto Serif CJK) 逐字回退
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
plt.rcParams['font.family'] = ['Times New Roman', 'Noto Serif CJK JP']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
# 新增这一行，关键！让\mathrm{} 使用rc设置的文本字体
plt.rcParams['mathtext.rm'] = 'Noto Serif CJK JP'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Noto Serif CJK JP'
plt.rcParams['mathtext.it'] = 'Noto Serif CJK JP:italic'
plt.rcParams['mathtext.bf'] = 'Noto Serif CJK JP:bold'
# ═══════════════════════════════════════════
# 每个工况的坐标缩放系数 (VGGT归一化→实际mm)
# 桩径90mm在不同工况中的VGGT尺寸不同
# ═══════════════════════════════════════════
CONDITION_SCALES = {
    'vggt_scour_purewave_current6_1_2': 0.861, # 浪高h=6cm,周期1.2s,振动级=0,流速=0.25
    'vggt_scour_random_145_0_25_pure_currents2': 0.71,  # 浪高h=0cm,周期=0s,振动级145,流速=0.25
    'vggt_scour_random145_h4_t1_4': 1.327,#浪高h=4cm,周期1.4s,振动级145,流速=0.25
    'vggt_scour_random145_h6T1_4_0_25':0.691,#浪高h=6cm,周期1.4s,振动级145,流速=0.25
    'vggt_scour_random_145_h8t1_4': 0.935,#浪高h=8cm,周期=1.4s,振动级145,流速=0.25
    'vggt_scour_random_140_H6_T1_6': 0.76,
    'vggt_scour_random147_4_h8t1_4': 0.595,
    'vggt_scour_150_h6t1_2': 0.825,
    # 新增工况在此添加
}

# ═══════════════════════════════════════════
# 手动指定基准线 (VGGT归一化坐标)
# Z < bed_level 的为坑面区域
# ═══════════════════════════════════════════
CONDITION_BED_LEVELS = {
    'vggt_scour_purewave_current6_1_2': -0.039,# 浪高h=6cm,周期1.2s,振动级=0,流速=0.25
    'vggt_scour_random_145_0_25_pure_currents2': -0.0545,  # 浪高h=0cm,周期=0s,振动级145,流速=0.25 (纯流)
    'vggt_scour_random145_h4_t1_4': -0.027,#浪高h=4cm,周期1.4s,振动级145,流速=0.25
    'vggt_scour_random145_h6T1_4_0_25': -0.051,#浪高h=6cm,周期1.4s,振动级145,流速=0.25
    'vggt_scour_random_145_h8t1_4': -0.0378,#浪高h=8cm,周期=1.4s,振动级145,流速=0.25
    'vggt_scour_random_140_H6_T1_6': -0.043,
    'vggt_scour_random147_4_h8t1_4': -0.0555,
    'vggt_scour_150_h6t1_2': -0.04,
    # 新增工况在此添加
}

# ═══════════════════════════════════════════
# 实验水力参数 (来自 vggt.txt 34-42行)
# D = 桩径 9cm = 0.09m
# H = 浪高 (m), T = 周期 (s), V = 振动级 (0或145), Uc = 流速 (m/s)
# ═══════════════════════════════════════════
EXP_PARAMS = {
    'vggt_scour_purewave_current6_1_2':        {'H': 0.06, 'T': 1.2, 'V': 0,   'Uc': 0.25},
    'vggt_scour_random_145_0_25_pure_currents2': {'H': 0.0, 'T': 0.0, 'V': 145, 'Uc': 0.25},
    'vggt_scour_random145_h4_t1_4':           {'H': 0.04, 'T': 1.4, 'V': 145, 'Uc': 0.25},
    'vggt_scour_random145_h6T1_4_0_25':       {'H': 0.06, 'T': 1.4, 'V': 145, 'Uc': 0.25},
    'vggt_scour_random_145_h8t1_4':           {'H': 0.08, 'T': 1.4, 'V': 145, 'Uc': 0.25},
    'vggt_scour_random_140_H6_T1_6': {'H': 0.06, 'T': 1.2, 'V': 140, 'Uc': 0.25},
    'vggt_scour_random147_4_h8t1_4': {'H': 0.06, 'T': 1.2, 'V': 147.4, 'Uc': 0.25},
    'vggt_scour_150_h6t1_2': {'H': 0.06, 'T': 1.2, 'V': 150, 'Uc': 0.25},
}

# ═══════════════════════════════════════════
# 自定义x轴标签 (右子图折线图, 可修改为任意名称)
# ═══════════════════════════════════════════
CUSTOM_XLABELS = {
    'vggt_scour_purewave_current6_1_2': 'A1',
    'vggt_scour_random_140_H6_T1_6': 'A2',
    'vggt_scour_random147_4_h8t1_4': 'A3',
    'vggt_scour_150_h6t1_2':'A4',
    'vggt_scour_random_145_0_25_pure_currents2': 'B1',
    'vggt_scour_random145_h4_t1_4': 'B2',
    'vggt_scour_random145_h6T1_4_0_25': 'B3',
    'vggt_scour_random_145_h8t1_4': 'B4',
}

# ═══════════════════════════════════════════
# 自定义图例标签 (左子图散点图, 可修改为任意名称)
# ═══════════════════════════════════════════
CUSTOM_LEGEND_LABELS = {
    'vggt_scour_purewave_current6_1_2': 'A1: $R_{\\mathit{v}}$=0, $H_{\\mathit{1/3}}$=6cm T=1.2s',
    'vggt_scour_random_140_H6_T1_6': 'A2: $R_{\\mathit{v}}$=0.0619, $H_{\\mathit{1/3}}$=6cm T=1.2s',
    'vggt_scour_random147_4_h8t1_4': 'A3: $R_{\\mathit{v}}$=0.1281, $H_{\\mathit{1/3}}$=6cm T=1.2s',
    'vggt_scour_150_h6t1_2':'A4: $R_{\\mathit{v}}$=0.1578, $H_{\\mathit{1/3}}$=6cm T=1.2s',
    'vggt_scour_random_145_0_25_pure_currents2': 'B1: $R_{\\mathit{v}}$=0.0995',
    'vggt_scour_random145_h4_t1_4': 'B2: $R_{\\mathit{v}}$=0.0995, $H_{\\mathit{1/3}}$=4cm T=1.4s',
    'vggt_scour_random145_h6T1_4_0_25': 'B3: $R_{\\mathit{v}}$=0.0995, $H_{\\mathit{1/3}}$=6cm T=1.4s',
    'vggt_scour_random_145_h8t1_4': 'B4: $R_{\\mathit{v}}$=0.0995, $H_{\\mathit{1/3}}$=8cm T=1.4s',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='output/aligned_ply')
    parser.add_argument('--output-dir', type=str, default='output/paper_vis')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集工况 (只包含 CONDITION_SCALES 中定义的)
    conditions = defaultdict(list)
    for ply_path in sorted(input_dir.rglob('*_aligned.ply')):
        if 'combined' in ply_path.name: continue
        cond = ply_path.parent.name
        if cond not in CONDITION_SCALES: continue
        conditions[cond].append(ply_path)

    CONDITION_ORDER = [
        'vggt_scour_purewave_current6_1_2',  # 2: 纯波, H=6, V=0
        'vggt_scour_random_140_H6_T1_6',
        'vggt_scour_random147_4_h8t1_4',
        'vggt_scour_150_h6t1_2',
        'vggt_scour_random_145_0_25_pure_currents2',  # 1: 纯流, H=0, V=145
        'vggt_scour_random145_h4_t1_4',               # 3: H=4, V=145
        'vggt_scour_random145_h6T1_4_0_25',           # 4: H=6, V=145
        'vggt_scour_random_145_h8t1_4',               # 5: H=8, V=145
    ]
    cond_names = [c for c in CONDITION_ORDER if c in conditions]
    n = len(cond_names)
    print(f"{n} conditions\n")

    # 先批量计算并输出数值
    print(f"{'Condition':<45} {'Scale':>6} {'Z_min':>10} {'Volume':>12} {'Area3D':>12}")
    print("-" * 90)
    results_cache = {}
    for cond_name in cond_names:
        s = CONDITION_SCALES.get(cond_name, 1.0)
        xyz_parts = []
        for pp in conditions[cond_name]:
            x, _ = read_ply(str(pp))
            if len(x) > 5: xyz_parts.append(x * s)
        if not xyz_parts: continue
        xyz = np.concatenate(xyz_parts, axis=0)
        bed = CONDITION_BED_LEVELS.get(cond_name, 0.0) * s
        vol, a3d, _ = compute_volume_and_area(xyz, bed)
        vol = vol * 4.0   # 体积修正 ×4（缩放系数偏小约4倍）
        below = xyz[xyz[:,2] < bed, 2] if (xyz[:,2] < bed).any() else np.array([bed])
        d = below.min()  # 最深点Z坐标
        results_cache[cond_name] = {'xyz': xyz, 'bed': bed, 'vol': vol, 'a3d': a3d, 'd': d}
        print(f"{cond_name:<45} {s:6.3f} {d:10.6f} {vol:12.6f} {a3d:12.6f}")
    print()

    # ============================================================
    # Figure 1: 深度热力图网格 (top-down view, Z as color)
    # ============================================================
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4.5))
    if rows == 1 and cols == 1: axes = np.array([axes])
    axes_flat = axes.flatten()

    cmap = plt.cm.inferno

    for idx, cond_name in enumerate(cond_names):
        ax = axes_flat[idx]
        if cond_name not in results_cache: continue
        rc = results_cache[cond_name]
        xyz, bed = rc['xyz'], rc['bed']
        below = xyz[xyz[:, 2] < bed]

        sc = ax.scatter(below[:, 0], below[:, 1], c=below[:, 2],
                        cmap=cmap, s=2, alpha=0.8, vmin=below[:, 2].min(), vmax=bed)
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_aspect('equal')
        ax.set_title(cond_name.replace('vggt_scour_', ''), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax, label='Z')
        v_str = f'{rc["vol"]:.4f}'
        d_str = f'{rc["d"]:.4f}'
        ax.text(0.02, 0.98, 'V=' + v_str + '\nZ_min=' + d_str,
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle('Scour Pit Depth Heatmaps (Top-down View)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / 'depth_heatmaps.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: depth_heatmaps.png")

    # ============================================================
    # Figure 2: 3D 表面渲染 (最好的一个工况)
    # ============================================================
    # 选体积最大的工况
    best_cond = cond_names[0]
    best_vol = 0
    for cond_name in cond_names:
        if cond_name in results_cache:
            if results_cache[cond_name]['vol'] > best_vol:
                best_vol = results_cache[cond_name]['vol']
                best_cond = cond_name

    # 最佳工况 3D 图
    rc_best = results_cache[best_cond]
    xyz = rc_best['xyz']
    bed = rc_best['bed']
    below = xyz[xyz[:, 2] < bed]
    # 降采样
    if len(below) > 30000:
        below = below[np.random.choice(len(below), 30000, replace=False)]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(below[:, 0], below[:, 1], below[:, 2],
                    c=below[:, 2], cmap=cmap, s=3, alpha=0.7, vmin=below[:, 2].min(), vmax=bed)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'{best_cond.replace("vggt_scour_","")}\nV={best_vol:.0f} mm³',
                 fontsize=14, fontweight='bold')
    ax.view_init(elev=30, azim=-60)
    fig.colorbar(sc, ax=ax, shrink=0.5, label='Z (depth)')
    plt.tight_layout()
    fig.savefig(output_dir / '3d_best.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: 3d_best.png ({best_cond})")

    # ============================================================
    # Figure 3: 汇总对比 (优雅版)
    # ============================================================
    names_short = [c.replace('vggt_scour_', '').replace('random_', 'R') for c in cond_names]
    vols = []
    areas = []
    depths_list = []
    for cond_name in cond_names:
        if cond_name in results_cache:
            rc = results_cache[cond_name]
            vols.append(rc['vol']); areas.append(rc['a3d']); depths_list.append(rc['d'])
        else:
            vols.append(0); areas.append(0); depths_list.append(0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, n))

    for ax, data, ylabel, title in [
        (axes[0], vols, '冲刷坑体积(m³)', ''),
        (axes[1], areas, 'Surface Area (mm²)', '冲刷坑表面积'),
        (axes[2], depths_list, '最大冲刷深度-Z/m ','')]:
        ax.plot(range(n), data, 'o-', color='#2c3e50', linewidth=2, markersize=8,
                markerfacecolor='#e74c3c', markeredgewidth=1, markeredgecolor='#2c3e50')
        ax.fill_between(range(n), 0, data, alpha=0.1, color='#3498db')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names_short, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # for i, val in enumerate(data):
        #     if val > 0:
        #         ax.text(i, val, f'{val:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    fig.suptitle('Scour Pit Analysis — Multi-Condition Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'summary_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: summary_comparison.png")

    # ============================================================
    # Figure 4: XZ 投影 + P95 基准线 (验证体积计算)
    # ============================================================
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    if rows == 1 and cols == 1: axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, cond_name in enumerate(cond_names):
        ax = axes_flat[idx]
        if cond_name not in results_cache: continue
        rc = results_cache[cond_name]
        xyz, bed = rc['xyz'], rc['bed']

        # XZ 投影
        ax.scatter(xyz[:, 0], xyz[:, 2], s=1, c='black', alpha=0.4)
        # P95 基准线
        ax.axhline(y=bed, color='red', ls='--', lw=2, label=f'Bed={bed:.3f}')
        # 标记坑内区域 (Z < bed)
        ax.fill_between([xyz[:,0].min(), xyz[:,0].max()], bed, xyz[:,2].min(),
                        alpha=0.08, color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(cond_name.replace('vggt_scour_', ''), fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle('XZ Projection + Manual Bed Level', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / 'xz_bed_check.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: xz_bed_check.png")

    # ============================================================
    # Figure 5: 经验公式拟合 — 冲刷坑体积预测
    # ============================================================
    from scipy.optimize import curve_fit, brentq

    D = 0.09  # 桩径 (m)

    # 收集实测体积与参数
    vols_meas = []
    H_list, T_list, V_list, Uc_list = [], [], [], []
    fit_cond_names = []
    for cond_name in cond_names:
        if cond_name in results_cache and cond_name in EXP_PARAMS:
            vols_meas.append(results_cache[cond_name]['vol'])
            p = EXP_PARAMS[cond_name]
            H_list.append(p['H']); T_list.append(p['T'])
            V_list.append(p['V']); Uc_list.append(p['Uc'])
            fit_cond_names.append(cond_name)
    vols_meas = np.array(vols_meas)
    H_arr = np.array(H_list); T_arr = np.array(T_list)
    V_arr = np.array(V_list); Uc_arr = np.array(Uc_list)

    # KC数: KC = πH/D (波浪), 纯流时KC=0
    # ── 拟合：V = A · (S/D)^a · exp(B·KC) · exp(E·Rv)（冲刷深度+波高+振动）──
    sqrt2 = np.sqrt(2)
    amp_map = {0: 0.0, 140: 3.94, 145: 3.94 + (8.15 - 3.94) * 5 / 7.4, 147.4: 8.15, 150: 10.1}
    def Rv_of(fv):
        return sqrt2 * (amp_map[fv] / 1000.0) / D   # 无量纲振动级 √2·振幅/D
    Rv_arr = np.array([Rv_of(v) for v in V_arr])
    # KC = Umax·T/D
    g = 9.81
    h_w = 0.3   # 水深 (m)
    def solve_k(T):
        if T <= 0: return 0.0
        w = 2.0 * np.pi / T
        return brentq(lambda k: w**2 - g * k * np.tanh(k * h_w), 1e-6, 100.0)
    k_arr = np.array([solve_k(T) for T in T_arr])
    sinh_kh = np.sinh(k_arr * h_w)
    KC_arr = np.array([0.343 * 1.2, 0.343* 1.2, 0.343 * 1.2, 0.343 * 1.2,0.25 * 1.4,0.319 * 1.4,0.343 * 1.4,0.402 * 1.4])
    KC_arr = np.pi * KC_arr / D
    S_cm = np.array([8.8, 8.6, 8.0, 8.0, 8.7, 8.3, 8.4, 8.2])   # 冲刷深度 (cm)
    SD_arr = S_cm / 9.0                              # S/D（桩径D=9cm）

    def model_vol(X, A, a, B, E):
        SD, KC, Rv = X
        return A * (SD ** a) * np.exp(B * KC) * np.exp(E * Rv)

    X_data = (SD_arr, KC_arr, Rv_arr)

    # 全拟合（带符号约束：a>0 冲刷深度正，B>0 波高正，E<0 振动负）
    best_params, _ = curve_fit(model_vol, X_data, vols_meas,
                               p0=[1e-6, 2.0, 0.1, -2.0],
                               bounds=([0, 0, 0, -10], [np.inf, 4, 1, 0]), maxfev=50000)
    A, a, B, E = best_params
    best_model = model_vol
    best_name = 'model_SD_KC_Rv'
    preds = best_model(X_data, *best_params)
    ss_res = np.sum((vols_meas - preds) ** 2)
    ss_tot = np.sum((vols_meas - np.mean(vols_meas)) ** 2)
    best_r2 = 1 - ss_res / ss_tot

    print(f"\n✓ 拟合: A={A:.6e}, a(S/D)={a:+.4f}, B(KC)={B:+.4f}, E(Rv)={E:+.4f}")
    print(f"  R²={best_r2:.4f}")

    # 打印对比表
    preds = best_model(X_data, *best_params)
    print(f"\n{'Condition':<45} {'V_meas':>10} {'V_pred':>10} {'Err%':>8}")
    print("-" * 75)
    for i, cn in enumerate(fit_cond_names):
        err_pct = (preds[i] - vols_meas[i]) / vols_meas[i] * 100 if vols_meas[i] > 0 else 0
        print(f"{cn:<45} {vols_meas[i]:10.6f} {preds[i]:10.6f} {err_pct:8.1f}")

    # 保存公式
    with open(output_dir / 'empirical_formula.txt', 'w') as f:
        f.write(f"Best model: {best_name} (R²={best_r2:.4f})\n")
        f.write(f"D = 0.09 m (pile diameter)\n\n")
        f.write(f"Formula: V = {A:.6e} × (S/D)^{a:.4f} × exp({B:.4f}×KC) × exp({E:.4f}×R_v)\n")
        f.write(f"where D=0.09 m, KC=πUmax*T/D\n\n")
        f.write(f"Parameters: H (wave height, m), T (period, s), V (vibration level, 0 or 145)\n\n")
        f.write("Comparison:\n")
        f.write(f"{'Condition':<45} {'V_meas':>10} {'V_pred':>10} {'Err%':>8}\n")
        f.write("-" * 75 + "\n")
        for i, cn in enumerate(fit_cond_names):
            err_pct = (preds[i] - vols_meas[i]) / vols_meas[i] * 100 if vols_meas[i] > 0 else 0
            f.write(f"{cn:<45} {vols_meas[i]:10.6f} {preds[i]:10.6f} {err_pct:8.1f}\n")

    # ── 预测 vs 实测 散点图 ──
    x_labels = [CUSTOM_XLABELS.get(c, c.replace('vggt_scour_', '')) for c in fit_cond_names]
    legend_labels = [CUSTOM_LEGEND_LABELS.get(c, c.replace('vggt_scour_', '')) for c in fit_cond_names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 预测 vs 实测
    ax = axes[0]
    ax.plot([0, vols_meas.max()*1.1], [0, vols_meas.max()*1.1], '--', color='gray', lw=1, alpha=0.5)
    scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(fit_cond_names)))
    for i, cond_name in enumerate(fit_cond_names):
        ax.scatter(vols_meas[i], preds[i], c=[scatter_colors[i]], s=80, edgecolors='black', linewidth=0.5, zorder=3,
                   label=f'{legend_labels[i]}')
    ax.set_xlabel('实测体积(m³)', fontsize=14)
    ax.set_ylabel('预测体积 (m³)', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='upper left', title='工况名称', title_fontsize=10)

    # 公式文本框
    formula_str = (f'$V = {A:.4f} \\cdot (S/D)^{{{a:.2f}}} \\cdot \\exp({B:.4f}\\overline{{KC}}) \\times \\exp({E:+.4f}R_v)$\n'
                   f'$\\overline{{KC}}=\\pi U_mT/D$, $R_v=\\sqrt{{2}}A/D$, $R^2={best_r2:.3f}$')
    ax.text(0.95, 0.15, formula_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # 右: 散点图对比 (按工况)，含宫氏公式(重新拟合)对比
    ax = axes[1]
    x_idx = np.arange(len(fit_cond_names))
    V_gong = D**3 * (10.506 * 0.13 * 4.0) * SD_arr**1.785   # 宫氏公式(系数×0.116×4)
    ax.scatter(x_idx, vols_meas, color='#e74c3c', s=90, marker='o',
               edgecolors='black', linewidth=0.8, label='实测值', zorder=3)
    ax.scatter(x_idx, preds, color='#3498db', s=90, marker='s',
               edgecolors='black', linewidth=0.8, label='式(9)预测值', zorder=3)
    ax.scatter(x_idx, V_gong, color='#2ecc71', s=90, marker='^',
           edgecolors='black', linewidth=0.8,
           label=r'$\mathrm{宫恩宇等预测值}^{[27,28]}$', zorder=3)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=12)
    ax.set_xlabel('工况名称', fontsize=14)
    ax.set_ylabel('冲刷坑体积(m³)', fontsize=14)
    ax.legend(fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / 'empirical_fit.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: empirical_fit.png")
    print(f"Saved: empirical_formula.txt")

    # ============================================================
    # Figure 6: 冲刷坑表面分色可视化 (验证基准线)
    # 床上点=灰色, 坑内点=按深度着色, 红色虚线=基准线
    # ============================================================
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5.5, rows*4.5))
    if rows == 1 and cols == 1: axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, cond_name in enumerate(cond_names):
        ax = axes_flat[idx]
        if cond_name not in results_cache: continue
        rc = results_cache[cond_name]
        xyz, bed = rc['xyz'], rc['bed']

        below = xyz[:, 2] < bed
        above = ~below

        # 床面以上点 (灰色)
        ax.scatter(xyz[above, 0], xyz[above, 1], s=1, c='lightgray',
                   alpha=0.35, label='床面 (Z ≥ 基准线)')
        # 坑内点 (按深度着色)
        if below.any():
            sc = ax.scatter(xyz[below, 0], xyz[below, 1], c=xyz[below, 2],
                            cmap='inferno', s=2, alpha=0.9, vmin=xyz[below, 2].min(), vmax=bed)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(sc, cax=cax, label='坑内Z (越深越亮)')

        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_aspect('equal')
        ax.set_title(f'{cond_name.replace("vggt_scour_", "")}\nbed={bed:.3f}',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.15)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle('冲刷坑表面分色 — 验证基准线 (灰=床面, 彩色=坑内)', fontsize=15,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / 'surface_bed_check.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: surface_bed_check.png")

    print(f"\n✓ All figures → {output_dir}/")


if __name__ == '__main__':
    main()
