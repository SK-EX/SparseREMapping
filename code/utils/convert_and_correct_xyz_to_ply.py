# -*- coding: utf-8 -*-
"""
convert_and_correct_xyz_to_ply.py
=================================
一步完成两项工作：
  1) 读取地形三维点云 *.xyz（逗号分隔，取前三列作为 x,y,z），写成 ASCII .ply；
  2) 用 TopoMultiFile1xiuzheng.m（长沙理工大学 Dr.Lv 的水槽扫描数据拼装校正逻辑）
     对坐标做同样的修正，输出修正后的 .ply。

用法：
    python convert_and_correct_xyz_to_ply.py            # 处理当前目录所有 *.xyz
    python convert_and_correct_xyz_to_ply.py a.xyz b.xyz # 只处理指定文件

输入文件顺序很重要：第 i 个文件对应 deltax/deltay/deltaz 的第 i 个值
（与原 MATLAB 中 dir() 按文件名排序、然后顺序赋值的规则一致）。
"""

import os
import glob
import sys
import math

import numpy as np


# =====================================================================
# 校正参数（取自 TopoMultiFile1xiuzheng.m）
# =====================================================================
A = [-1, 2]                # x 范围 / m（轴交换后的新坐标）
B = [-1, 2]                # y 范围 / m
C = [-1, 0.3]              # z 高度范围 / m（高度过滤）
Angle = [217, 290]         # 航向角范围 / deg
H = 0.5                    # 探头离水底高度 / m
Gama = 1                   # 探头倾斜校正 / deg
Pile = [0.81, 0.80]        # 桩位置 / m（最后整体平移）

# 拼装校准量 / m —— 下标 i 与输入文件的处理顺序一一对应
deltax = [0, -0.49, -0.543, -0.277, 2.416, 2.424]
deltay = [0, -0.349, -0.033, -0.4265, 0.111, 1.375]
deltaz = [0, 0.172, 0.009, 0.184, 0.004, 0.007]

# 输出文件命名
RAW_SUFFIX = '.ply'              # 原始坐标 -> 1.2Gs.ply
CORR_SUFFIX = '_corrected.ply'   # 修正坐标 -> 1.2Gs_corrected.ply


# =====================================================================
# 工具函数
# =====================================================================
def rot2d(deg):
    """2x2 逆时针旋转矩阵，等价于 MATLAB 的 [cosd, -sind; sind, cosd]."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s], [s, c]])


def read_xyz(fn):
    """读取逗号分隔 .xyz，取前三列 (x, y, z)。"""
    return np.loadtxt(fn, delimiter=',')[:, :3]


def read_ply_xyz(fn):
    """读取 ASCII .ply 中 end_header 之后的顶点 x,y,z。"""
    with open(fn) as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip() == 'end_header')
    return np.loadtxt(lines[idx + 1:])            # N x 3


def write_ply_xyz(fn, pts):
    """把 N x 3 数组写成仅含顶点的 ASCII .ply。"""
    pts = np.atleast_2d(pts)
    with open(fn, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % pts.shape[0])
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f, pts, fmt='%.10f')


# =====================================================================
# 坐标修正（与 TopoMultiFile1xiuzheng.m 逐行对应）
# =====================================================================
def correct(xyz, i):
    """
    xyz : N x 3，存放原始扫描坐标 Data(:,1:3)。
    i   : 文件序号（0 起），用于取第 i 组拼装校准量 deltax/deltay/deltaz。
    返回修正后的 N' x 3 点云（经过三次范围裁剪，点数可能减少）。
    """
    x = xyz[:, 0].astype(float).copy()
    y = xyz[:, 1].astype(float).copy()
    z = xyz[:, 2].astype(float).copy()

    # 1) 原始扫描坐标系换算/转置：x=-z, y=-y, z=-x
    nx, ny, nz = -z, -y, -x

    # 2) 按 x、y 范围裁剪
    mask = (nx >= A[0]) & (nx <= A[1]) & (ny >= B[0]) & (ny <= B[1])
    nx, ny, nz = nx[mask], ny[mask], nz[mask]

    # 3) 原点还原（减各自最小值）
    nx = nx - nx.min()
    ny = ny - ny.min()

    # 4) 航向校正：yz 平面旋转 theta = -(Angle(2)-Angle(1))/5 = -14.6°，再加探头高度 H
    theta = -(Angle[1] - Angle[0]) / 5.0
    yz = rot2d(theta) @ np.vstack([ny, nz])
    ny, nz = yz[0], yz[1]
    nz = nz + H

    # 5) 倾斜校正 Gama：xz 平面旋转 1°
    xz = rot2d(Gama) @ np.vstack([nx, nz])
    nx, nz = xz[0], xz[1]

    # 6) 高度过滤（z 在 [C0, C1] 之内保留）
    mask = (nz >= C[0]) & (nz <= C[1])
    nx, ny, nz = nx[mask], ny[mask], nz[mask]

    # 7) 原点校正：加各自最大值 + 第 i 组拼装校准量
    nx = nx + nx.max() + deltax[i]
    ny = ny + ny.max() + deltay[i]
    nz = nz + deltaz[i]

    # 8) 桩位置整体平移（原代码是对合并后点云统一做，常数平移逐文件等价）
    nx = nx - Pile[0]
    ny = ny - Pile[1]

    return np.column_stack([nx, ny, nz])


# =====================================================================
# 主流程
# =====================================================================
def main():
    args = sys.argv[1:]
    if args:
        files = args
    else:
        files = sorted(glob.glob('*.xyz'))
    if not files:
        print('未找到 *.xyz 文件。')
        return

    n_files = len(files)
    if n_files > len(deltax):
        print('警告：输入文件数 %d 超过 deltax/deltay/deltaz 长度 %d，'
              '超出部分将没有对应的拼装校准量。' % (n_files, len(deltax)))

    for i, xyz_file in enumerate(files):
        stem = os.path.splitext(os.path.basename(xyz_file))[0]

        # --- 1) .xyz -> 原始 .ply（取前三列） ---
        raw = read_xyz(xyz_file)
        raw_ply = stem + RAW_SUFFIX
        write_ply_xyz(raw_ply, raw)

        # --- 2) 坐标修正 -> 修正 .ply ---
        if i < len(deltax):
            corr = correct(raw, i)
        else:
            corr = correct(raw, len(deltax) - 1)  # 无对应校准时复用最后一组
        corr_ply = stem + CORR_SUFFIX
        write_ply_xyz(corr_ply, corr)

        print('%-8s  raw: %6d -> %-16s  |  corrected: %6d -> %s'
              % (stem, raw.shape[0], raw_ply, corr.shape[0], corr_ply))

    print('完成。')


if __name__ == '__main__':
    main()
