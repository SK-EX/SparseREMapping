#!/usr/bin/env python3
"""
SCI论文算法框架图 — 单桩冲刷坑三维重建全流程
==============================================
用法: python network_framework.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Rectangle
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import numpy as np

# 宋体
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
plt.rcParams['font.sans-serif'] = ['Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_aspect('equal')

# ═══════════════════════════ Colors ═══════════════════════════
C_INPUT   = '#E8F5E9'  # 浅绿 - 输入
C_SEG     = '#FFF3E0'  # 浅橙 - 分割
C_VGGT    = '#E3F2FD'  # 浅蓝 - VGGT
C_MERGE   = '#F3E5F5'  # 浅紫 - 合并对齐
C_POST    = '#FFEBEE'  # 浅红 - 后处理
C_OUTPUT  = '#E0F2F1'  # 浅青 - 输出
C_ARROW   = '#37474F'  # 深灰 - 箭头
C_BORDER  = '#455A64'  # 边框

BOX_STYLE = dict(boxstyle='round,pad=0.15', edgecolor=C_BORDER, facecolor='white', linewidth=1.8)

def draw_box(ax, x, y, w, h, text, fc, title=None, fontsize=9):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                          facecolor=fc, edgecolor=C_BORDER, linewidth=1.8, alpha=0.95, zorder=2)
    ax.add_patch(box)
    if title:
        ax.text(x + w/2, y + h + 0.12, title, ha='center', va='bottom',
                fontsize=fontsize-1, fontweight='bold', color=C_BORDER, zorder=3)
    if isinstance(text, list):
        for i, line in enumerate(text):
            ax.text(x + w/2, y + h/2 - (len(text)-1)*fontsize*0.15 + i*fontsize*0.33,
                    line, ha='center', va='center', fontsize=fontsize, color='#212121', zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='#212121', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, style='solid'):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.5,
                                connectionstyle='arc3,rad=0' if style=='solid' else f'arc3,rad={style}'))

def draw_sub_box(ax, x, y, w, h, text, fc, fontsize=7.5):
    """绘制子模块小框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                          facecolor=fc, edgecolor=C_BORDER, linewidth=1.2, alpha=0.9, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color='#212121', zorder=3)

# ═══════════════════════════ Phase labels ═══════════════════════════
PHASE_FS = 10
ax.text(1.2, 11.3, '(a) 图像分割阶段', fontsize=PHASE_FS, fontweight='bold', color='#E65100')
ax.text(6.8, 11.3, '(b) 多视角三维重建阶段', fontsize=PHASE_FS, fontweight='bold', color='#0D47A1')
ax.text(13.5, 11.3, '(c) 后处理与参数提取', fontsize=PHASE_FS, fontweight='bold', color='#B71C1C')

# ═══════════════════════════ Row heights ═══════════════════════════
# Input row
Y_IN  = 9.8
# Architecture row
Y_ARC = 7.8
# Feature row (VGGT)
Y_VGGT = 5.5
# Output
Y_OUT = 3.5
# Post results
Y_RES = 1.5

# ══════════════════ STAGE (a): Segmentation ══════════════════

# Input: multi-view images
draw_box(ax, 0.3, Y_IN, 3.0, 1.0,
         ['多视角RGB图像', '3000×4000, 绕桩拍摄'], C_INPUT,
         fontsize=9)

# Backbone detail
draw_arrow(ax, 1.8, Y_IN, 1.8, Y_ARC + 1.4)

# UNet + ResNet34 architecture
draw_box(ax, 0.3, Y_ARC, 3.0, 1.35,
         ['ScourUNet', '(UNet编码-解码 + ResNet34骨干)'], C_SEG,
         fontsize=9)

# Internal sub-blocks of UNet
draw_sub_box(ax, 0.5, Y_ARC - 0.55, 1.25, 0.45, 'ResNet34\n编码器', fc='#FFCC80', fontsize=6.5)
draw_sub_box(ax, 1.85, Y_ARC - 0.55, 1.25, 0.45, '跳跃连接\n+ 上采样解码器', fc='#FFE0B2', fontsize=6.5)

# Loss
draw_box(ax, 0.6, Y_ARC - 1.1, 2.4, 0.55, 'BCE + Dice Loss (有监督)', fc='#FFF8E1', fontsize=8)

draw_arrow(ax, 1.8, Y_ARC - 0.55, 1.8, Y_ARC - 1.1)

# Output: masks
draw_box(ax, 0.3, Y_ARC - 2.0, 3.0, 0.75,
         ['冲刷坑二值掩模 (Binary Mask)', 'mask × 原图 → 坑面注意力增强'], fc='#FFECB3',
         fontsize=8)

draw_arrow(ax, 1.8, Y_ARC, 1.8, Y_ARC - 1.35)

# ══════════════════ STAGE (b): VGGT ══════════════════

# Arrow from masks to VGGT
draw_arrow(ax, 3.3, Y_ARC - 1.6, 4.5, Y_VGGT + 1.0)

# VGGT main box
draw_box(ax, 4.5, Y_VGGT + 0.1, 4.8, 1.6,
         ['VGGT-1B 多视角三维重建',
          'Mask Injection + 全局注意力融合'], C_VGGT, fontsize=9)

# VGGT internals
draw_sub_box(ax, 4.75, Y_VGGT - 0.5, 1.35, 0.5, 'ViT编码器\n多图全局注意力', fc='#90CAF9', fontsize=6)
draw_sub_box(ax, 6.25, Y_VGGT - 0.5, 1.35, 0.5, '深度图+相机位姿\n联合预测', fc='#64B5F6', fontsize=6)
draw_sub_box(ax, 7.75, Y_VGGT - 0.5, 1.35, 0.5, '逐像素反投影\n→ 3D世界坐标点云', fc='#42A5F5', fontsize=6)

# Arrow to per-view PLY
draw_arrow(ax, 6.9, Y_VGGT - 0.4, 6.9, Y_VGGT - 1.3)

# Per-view point clouds
draw_box(ax, 4.5, Y_VGGT - 2.0, 4.8, 0.65,
         ['单视角点云 (.ply) × N → 所有视角合并 ∈ 公共世界坐标系'], fc='#BBDEFB',
         fontsize=8)

# ══════════════════ STAGE (c): Post-processing ══════════════════

# Arrow to alignment
draw_arrow(ax, 9.3, Y_VGGT - 1.7, 10.8, Y_IN - 0.8)

# PCA Alignment
draw_box(ax, 10.5, Y_IN - 0.1, 2.8, 0.8,
         ['PCA 全局对齐',
          '质心平移 + 床面法向旋转'], C_MERGE, fontsize=9)

draw_arrow(ax, 11.9, Y_IN + 0.7, 11.9, Y_IN + 1.2)

# Combined point cloud
draw_box(ax, 10.5, Y_IN + 1.3, 2.8, 0.85,
         ['融合点云 (Combined PLY)',
          'PCA对齐后所有视角叠加'], fc='#E1BEE7', fontsize=8.5)

# Aligned per-view PLYs
draw_box(ax, 14.5, Y_IN + 1.3, 2.8, 0.85,
         ['对齐单视角点云',
          '保留逐帧几何细节'], fc='#CE93D8', fontsize=8.5)

draw_arrow(ax, 13.3, Y_IN + 1.7, 14.5, Y_IN + 1.7)

# Down from alignment to processing
draw_box(ax, 10.5, Y_ARC - 0.0, 2.8, 0.7,
         ['手动标定床面基准线', 'Z < Bed → 坑面区域'], fc='#F8BBD0', fontsize=8)
draw_arrow(ax, 11.9, Y_IN - 0.1, 11.9, Y_ARC - 0.0)

# Quantify — branch into 3
for i, (xpos, title, items) in enumerate([
    (9.2, '体积计算', ['Delaunay三角化',
                        '柱体法求和',
                        'V = Σ(Aᵢ × d̄ᵢ/3)']),
    (11.9, '表面积计算', ['3D Delaunay三角网',
                         '三角形面积累加',
                         'A₃D = Σ 0.5|v₁×v₂|']),
    (14.6, '坡度参数提取', ['YZ/XZ下包络线',
                          '最深点+临界点',
                          'θ₁ θ₂ φ₁ φ₂ h']),
]):
    draw_box(ax, xpos, Y_ARC - 1.4, 2.4, 1.3, items, C_POST, fontsize=7.5)
    draw_arrow(ax, 11.9, Y_ARC - 0.0, xpos + 1.2, Y_ARC - 1.4)
    ax.text(xpos + 1.2, Y_ARC - 1.5, title, fontsize=7.5, fontweight='bold',
            color='#B71C1C', ha='center', zorder=3)

# Final output
draw_arrow(ax, 11.9, Y_ARC - 2.7, 11.9, Y_ARC - 3.2)

draw_box(ax, 9.0, Y_ARC - 4.0, 5.8, 0.65,
         ['输出: 冲刷体积 V, 3D表面积 A, 冲刷深度 h, 坡度角 θ₁ θ₂, 冲刷半径 φ₁ φ₂'], fc=C_OUTPUT, fontsize=8.5)

# ══════════════════ Data icons ══════════════════
ICON_Y = 0.3
for x, icon, label in [
    (1.8, '📷', '输入'),
    (4.0, '⬜', '掩模'),
    (6.9, '☁️', '点云'),
    (11.9, '📐', '参数'),
]:
    ax.text(x, ICON_Y, label, fontsize=8, ha='center', color='#78909C')

# ══════════════════ Legend at bottom ══════════════════
legend_elements = [
    mpatches.Patch(facecolor=C_SEG,   edgecolor=C_BORDER, label='图像分割 (ScourUNet)'),
    mpatches.Patch(facecolor=C_VGGT,  edgecolor=C_BORDER, label='VGGT-1B 三维重建'),
    mpatches.Patch(facecolor=C_MERGE, edgecolor=C_BORDER, label='点云对齐与融合'),
    mpatches.Patch(facecolor=C_POST,  edgecolor=C_BORDER, label='后处理与参数提取'),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor=C_BORDER, label='输出结果'),
]
leg = ax.legend(handles=legend_elements, loc='lower center', ncol=5,
                fontsize=8, framealpha=0.9, edgecolor='#78909C')
leg.set_bbox_to_anchor((0.5, -0.02))

plt.tight_layout(pad=0.5)
fig.savefig('output/paper_vis/network_framework.png', dpi=250, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("Saved: output/paper_vis/network_framework.png")
