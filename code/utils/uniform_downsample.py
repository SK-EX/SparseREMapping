#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY 点云「均匀(空间)降采样」—— 体素栅格降采样,不是随机采样。

原理:把三维空间切成边长 = voxel_size 的立方体网格(voxel)。
每个非空体素只保留一个代表点 —— 默认取"离该体素中心最近的真实点"
(完全保留原始坐标/法向/颜色,不产生新值)。
这样降采样后点的最小间距约为 voxel_size,空间分布均匀,
不会像随机降采样那样出现局部密/局部疏。

用法示例:
    # 方式一:按固定体素大小(最直观的空间稀疏程度)
    python uniform_downsample.py -v 0.01

    # 方式二:按保留比例(自动反推体素大小,使点数约为原来的 20%)
    python uniform_downsample.py -r 0.2

    # 方式三:按目标点数
    python uniform_downsample.py -n 40000

输出:默认在输入文件名后加 _uniform,如 points3D_dense_no_red2_uniform.ply
"""
import argparse
import numpy as np
import plyfile

# PLY 属性类型 -> numpy 编码(输出仍写成 binary_little_endian)
_DTYPE = {
    "float": "<f4", "double": "<f8",
    "uchar": "u1", "char": "i1",
    "ushort": "<u2", "short": "<i2",
    "uint": "<u4", "int": "<i4",
}


def read_vertex(path):
    """读入 vertex 全部属性列。
    返回 (名称列表, {名称: numpy数组}, {名称: 源PLY属性numpy类型})。"""
    ply = plyfile.PlyData.read(path)
    v = ply["vertex"]
    names = [p.name for p in v.properties]
    cols = {name: np.asarray(v[name]) for name in names}
    dtypes = {p.name: np.dtype(p.val_dtype).newbyteorder("<")
              for p in v.properties}
    return names, cols, dtypes


def _group_info(points, mins, voxel):
    """把点量化到体素,返回排序后的分组信息(起点 / 各组大小 / 各点体素中心)。"""
    idx = np.floor((points - mins) / voxel).astype(np.int64)
    mx = idx.max(axis=0) + 1                      # 每轴体素数
    key = (idx[:, 0] * mx[1] + idx[:, 1]) * mx[2] + idx[:, 2]  # 一维线性体素号
    centers = mins + (idx + 0.5) * voxel           # 每个点自己所在体素的中心
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.diff(ks) != 0) + 1  # 组起点
    starts = np.concatenate([[0], starts])
    sizes = np.diff(np.concatenate([starts, [len(ks)]]))
    return order, starts, sizes, centers


def downsample_cols(points, cols, voxel, mode="nearest"):
    """
    返回降采样后的 (points, cols),保持各列 dtype 不变。
    mode = "nearest" : 每体素取离中心最近的真实点(默认,保真)
           = "average": 每体素取平均(质心 + 均值颜色/法向)
    """
    n = len(points)
    mins = points.min(axis=0)
    order, starts, sizes, centers = _group_info(points, mins, voxel)

    if mode == "nearest":
        d2 = ((points - centers) ** 2).sum(axis=1)      # 到自身体素中心距离^2
        d2s = d2[order]
        gmin = np.minimum.reduceat(d2s, starts)          # 每组最小距离
        is_min = d2s == np.repeat(gmin, sizes)           # 每组取第一个最小值
        pos = np.arange(n)
        pick_pos = np.minimum.reduceat(np.where(is_min, pos, n), starts)
        orig = order[pick_pos]                           # 原数组中的下标
        out_points = points[orig]
        out_cols = {name: cols[name][orig] for name in cols}
    else:  # average:每体素质心 + 均值法向/颜色
        s = np.cumsum(np.concatenate([[0], sizes]))
        out_points = np.empty((len(starts), 3), dtype=points.dtype)
        out_cols = {}
        for name, c in cols.items():
            cs = c[order]
            if c.dtype.kind == "f":                      # float:直接均值
                out_cols[name] = np.add.reduceat(cs, starts) / sizes[:, None] \
                    if cs.ndim == 2 else np.add.reduceat(cs, starts) / sizes
            else:                                        # uchar:均值后取整
                avg = np.add.reduceat(cs.astype(np.float64), starts) / sizes
                out_cols[name] = np.rint(avg).astype(c.dtype)
        out_points = np.column_stack(
            [np.add.reduceat(points[order, k], starts) / sizes
             for k in range(3)]).astype(points.dtype)
    return out_points, out_cols


def count_for_voxel(points, voxel):
    """给定体素大小 -> 非空体素数(即降采样后的点数)。"""
    mins = points.min(axis=0)
    idx = np.floor((points - mins) / voxel).astype(np.int64)
    return len(np.unique(idx, axis=0))


def pick_voxel_for_target(points, target, diag):
    """二分搜索体素大小,使降采样后点数尽量接近 target。"""
    lo, hi = diag / 1e4, diag * 0.5          # lo≈保留全部, hi≈剩 1 点
    best_v, best_err = None, np.inf
    for _ in range(50):
        v = np.sqrt(lo * hi)                  # 对数区间折半
        c = count_for_voxel(points, float(v))
        err = abs(c - target) / target
        if err < best_err:
            best_v, best_err = v, err
        if c > target:
            lo = v                            # 点太多 -> 体素放大
        else:
            hi = v
    return float(best_v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default="points3D_dense_no_red2.ply",
                    help="输入 PLY(默认 points3D_dense_no_red2.ply)")
    ap.add_argument("-o", "--out", help="输出 PLY(默认 输入名_uniform.ply)")
    ap.add_argument("-v", "--voxel", type=float, help="固定体素边长(米/单位),最直接的稀疏控制")
    ap.add_argument("-r", "--keep-ratio", type=float,
                    help="保留比例(0,1),自动反推体素,如 0.2=保留约20%")
    ap.add_argument("-n", "--num-points", type=int, help="目标点数,自动反推体素")
    ap.add_argument("--mode", choices=["nearest", "average"], default="nearest",
                    help="代表点取法:nearest=体素中心最近的真实点(默认,保真);"
                         "average=质心/均值")
    args = ap.parse_args()

    names, cols, dtypes = read_vertex(args.input)
    n_in = len(cols["x"])
    pts = cols["x"], cols["y"], cols["z"]
    points = np.column_stack([cols[c] for c in ("x", "y", "z")]).astype(np.float64)
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    if args.voxel:
        voxel = args.voxel
    elif args.num_points:
        voxel = pick_voxel_for_target(points, args.num_points, diag)
    else:
        ratio = args.keep_ratio if args.keep_ratio else 0.2
        if not 0 < ratio < 1:
            raise SystemExit(f"keep-ratio 需在 (0,1) 内,收到 {ratio}")
        voxel = pick_voxel_for_target(points, round(n_in * ratio), diag)

    out_points, out_cols = downsample_cols(points, cols, voxel, mode=args.mode)
    n_out = len(out_points)

    # 颜色/nx.. 全部原样保留;只让 x,y,z 用降采样后的坐标
    out_cols = dict(out_cols)
    out_cols["x"], out_cols["y"], out_cols["z"] = \
        out_points[:, 0], out_points[:, 1], out_points[:, 2]

    # 重建结构化数组,严格按源 PLY 的属性类型写回(不把坐标升级成 double)
    rec = np.empty(n_out, dtype=[(nm, dtypes[nm]) for nm in names])
    for nm in names:
        rec[nm] = out_cols[nm].astype(dtypes[nm])

    el = plyfile.PlyElement.describe(rec, "vertex")
    plyfile.PlyData([el], text=False).write(args.out or
        args.input.replace(".ply", "_uniform.ply"))

    print(f"输入   : {n_in:>8d} 点")
    print(f"体素   : {voxel:.5f}  (单位同坐标)")
    print(f"输出   : {n_out:>8d} 点  (保持 {n_out / n_in:.1%})")
    print(f"稀疏度 : 降采样后点间距约 ≥ {voxel:.5f}")


if __name__ == "__main__":
    main()
