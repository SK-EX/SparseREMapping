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


# 输出文件命名
RAW_SUFFIX = '.ply'              # 原始坐标 -> 1.2Gs.ply

# =====================================================================
# 工具函数
# =====================================================================

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
# 主流程
# =====================================================================
def main():
    args = sys.argv[1:]
    if args:
        files = args
    else:
        files = sorted(glob.glob('*.txt'))
    if not files:
        print('未找到 *.txt 文件。')
        return

    n_files = len(files)

    for i, xyz_file in enumerate(files):
        stem = os.path.splitext(os.path.basename(xyz_file))[0]

        # --- 1) .txt -> 原始 .ply（取前三列） ---
        raw = read_xyz(xyz_file)
        raw_ply = stem + RAW_SUFFIX
        write_ply_xyz(raw_ply, raw)

    print('完成。')


if __name__ == '__main__':
    main()
