# -*- coding: utf-8 -*-
"""
merge_corrected_ply_to_data.py
==============================
把 6 个修正过的 *.ply（MeshLab / VCGLIB 风格，每点带 x,y,z,nx,ny,nz）
按顺序合并成一个 Data.ply。

用法：
    python merge_corrected_ply_to_data.py                       # 合并当前目录全部 *_corrected.ply
    python merge_corrected_ply_to_data.py a.ply b.ply c.ply     # 只合并指定文件

约定：
    - 输入为 ASCII PLY，头部含 element vertex（点数）与若干 property float/double；
    - 顶点数据为定长行，每行浮点列数与 property 个数一致；
    - 假定每个文件的顶点列数相同、均为定长（面 count 为 0，无真正的面数据行）；
    - 合并仅把所有文件的顶点按给定顺序纵向拼接，头部总点数更新；
    - 输出保留输入的全部浮点 property（x,y,z,nx,ny,nz 等）与 element face 0 结构。

依赖：numpy
"""
import os
import sys
import glob

import numpy as np

# 输出文件名
OUTPUT = 'Data.ply'
# 写入顶点时用的浮点格式
FMT = '%.10g'


def parse_vcg_ply(fn):
    """
    读取一个 VCGLIB 风格 ASCII PLY。

    返回 (verts, float_names)：
        verts       : N x K 的 ndarray，N = 顶点数，K = 浮点 property 个数
        float_names : K 个 property 的名称（如 ['x','y','z','nx','ny','nz']）
    """
    with open(fn) as f:
        lines = f.readlines()

    # 定位头部结束行
    try:
        idx = next(i for i, l in enumerate(lines) if l.strip() == 'end_header')
    except StopIteration:
        raise ValueError('%s: 找不到 end_header，可能不是 ASCII PLY。' % fn)

    header = lines[:idx]

    # 顶点数
    try:
        vcount = int([l for l in header if l.startswith('element vertex')][0].split()[2])
    except IndexError:
        raise ValueError('%s: 头部缺少 element vertex 声明。' % fn)

    # 顶点浮点 property 名称（按出现顺序）
    float_names = [l.split()[2] for l in header
                   if l.startswith('property float') or l.startswith('property double')]
    if not float_names:
        raise ValueError('%s: 头部没有浮点 property。' % fn)

    # 数据体
    body = lines[idx + 1:]
    nbody = len(body)
    if nbody < vcount:
        raise ValueError('%s: 数据行数 %d < 头部顶点数 %d。' % (fn, nbody, vcount))
    if nbody > vcount:
        print('  注意: %s 数据行数 %d > 头部顶点数 %d，将只取前 %d 行。'
              % (os.path.basename(fn), nbody, vcount, vcount))
        body = body[:vcount]

    verts = np.loadtxt(body)
    verts = np.atleast_2d(verts)          # 保证形状为 N x K
    if verts.ndim != 2 or verts.shape[1] != len(float_names):
        raise ValueError('%s: 数据列数 %s 与头部 property 数 %d 不一致。'
                         % (fn, verts.shape, len(float_names)))

    return verts, float_names


def write_vcg_ply(fn, verts, float_names):
    """把 N x K 顶点数组写成保留 x,y,z,... 与 element face 0 结构的 ASCII PLY。"""
    n, k = verts.shape
    assert k == len(float_names), 'property 名称数(%d)与列数(%d)不一致' % (len(float_names), k)
    with open(fn, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment VCGLIB generated\n')
        f.write('element vertex %d\n' % n)
        for name in float_names:
            f.write('property float %s\n' % name)
        f.write('element face 0\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        np.savetxt(f, verts, fmt=FMT)


def main():
    args = sys.argv[1:]
    files = args if args else sorted(glob.glob('*_corrected.ply'))
    if not files:
        print('未找到要合并的 *.ply 文件。')
        return

    print('合并顺序:')
    parts = []
    common_names = None
    for f in files:
        verts, names = parse_vcg_ply(f)
        if common_names is None:
            common_names = names
        elif names != common_names:
            print('警告: %s 的 property 与首个文件不一致，将沿用首个文件的结构。'
                  % os.path.basename(f))
        parts.append(verts)
        print('  %-22s %8d 点  (列: %s)' % (os.path.basename(f), verts.shape[0], ','.join(names)))

    merged = np.vstack(parts)
    write_vcg_ply(OUTPUT, merged, common_names)

    print('\n%s  合并总点数 = %d (= %s)' %
          (OUTPUT, merged.shape[0], ' + '.join(str(p.shape[0]) for p in parts)))
    print('包围盒: x[%.4f,%.4f]  y[%.4f,%.4f]  z[%.4f,%.4f]' %
          (merged[:, 0].min(), merged[:, 0].max(),
           merged[:, 1].min(), merged[:, 1].max(),
           merged[:, 2].min(), merged[:, 2].max()))


if __name__ == '__main__':
    main()
