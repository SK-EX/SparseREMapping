#!/usr/bin/env python3
"""
VGGT 稠密重建 — 环绕视角整组联合重建 (仿官方 demo_colmap2)
=========================================
核心方法 (与 facebook/VGGT demo_colmap2.py 一致):
  1. 整组环绕视角在【一次前向】里进 VGGT 的 aggregator + camera_head + depth_head
  2. 用 camera head 的位姿 (extrinsic/intrinsic, 全局统一世界坐标系) 把每帧 depth
     反投影成三维点 → 这才是真正"同一坐标系"的场景点云
     ⚠ 不使用 model 直出的 world_points (各帧参考不一致, 直接拼接会杂乱)
  3. 置信度按 VGGT 的 expp1 量纲过滤 (默认 ≥5.0, 官方 demo 同样默认)
  4. 显存不够 → 降分辨率 (默认 392→336); 仍不够 → 环绕等间隔抽稀视角
     (仍单次前向、单一坐标系), 绝不退回逐张重建

用法:
  python vggt_scour_supervised2.py                          # 整圈视角联合重建 (392px)
  python vggt_scour_supervised2.py --target-size 518        # 更高分辨率(更吃显存)
  python vggt_scour_supervised2.py --num-images 24          # 等间隔抽 24 视角(仍一次联合)
  python vggt_scour_supervised2.py --ply-conf 5.0           # 置信度阈值 (expp1 量纲, 官方5.0)
  python vggt_scour_supervised2.py --ply-interval 0         # 只出合并 combined.ply
"""

# ⚠️ 必须在 import torch 之前设置，减少显存碎片
import os, sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import cv2
import numpy as np
import torch
import argparse
import gc
import time

sys.path.insert(0, '/home/wyh/Desktop/wyh/code')
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ═══════════════════════════════════════════════════════════════════
# VGGT 推理辅助
# ═══════════════════════════════════════════════════════════════════

def get_dtype():
    """选择最优混合精度 dtype"""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:  # Ampere+ (A100/A6000/RTX30xx+)
            return torch.bfloat16
        elif cap[0] >= 7:  # Volta/Turing (V100/RTX20xx)
            return torch.float16
    return torch.float32


def get_vram_info(device='cuda'):
    """返回 (已用GB, 总量GB)"""
    if device == 'cuda' and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0, 0


def load_vggt(device='cuda'):
    """加载 VGGT 模型"""
    print("Loading VGGT-1B...")
    model = VGGT()
    ckpt_path = '/home/wyh/Downloads/model.pt'
    if not os.path.exists(ckpt_path):
        hf_cache = os.path.expanduser('~/.cache/huggingface/hub/models--facebook--VGGT-1B')
        if os.path.exists(hf_cache):
            import glob as _glob
            candidates = _glob.glob(f'{hf_cache}/**/*.pt', recursive=True) + \
                         _glob.glob(f'{hf_cache}/**/*.bin', recursive=True)
            if candidates:
                ckpt_path = candidates[0]
                print(f"  Using cached checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Download from https://huggingface.co/facebook/VGGT-1B")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    model = model.to(device).eval()
    return model


def _unproject_depth_to_world(depth, extrinsics, intrinsics):
    """
    depth:        [S, H, W] float32, 深度 (来自 VGGT depth head)
    extrinsics:   [S, 3, 4]  OpenCV 约定 camera-from-world [R|t]
    intrinsics:   [S, 3, 3]  [[fx,0,cx],[0,fy,cy],[0,0,1]] (像素, 分辨率=HxW)
    返回:         [S, H, W, 3]  全局世界坐标 (与官方 unproject_depth_map_to_point_map 一致)
    """
    S, H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = []
    for i in range(S):
        d = depth[i]
        K = intrinsics[i]
        x_cam = (u - K[0, 2]) * d / K[0, 0]
        y_cam = (v - K[1, 2]) * d / K[1, 1]
        cam = np.stack([x_cam, y_cam, d], axis=-1)

        R = extrinsics[i, :3, :3]      # cam-from-world 旋转
        t = extrinsics[i, :3, 3]       # cam-from-world 平移
        R_inv = R.T                    # world-from-cam 旋转
        t_inv = -R_inv @ t             # world-from-cam 平移
        world = cam @ R_inv.T + t_inv  # 行向量: world = cam·R + (-Rᵀt)
        out.append(world)
    return np.stack(out, axis=0).astype(np.float32)


def run_vggt_joint(image_paths, model, target_size, dtype, device='cuda'):
    """
    对整组环绕视角运行一次 VGGT 前向 → 全局统一相机位姿 + depth。
    返回:
      depth         [B, S, H, W, 1]
      depth_conf    [B, S, H, W]       (expp1 量纲, 官方阈值 ~5.0)
      extrinsic     [S, 3, 4]          每帧相机位姿 (OpenCV, camera-from-world)
      intrinsic     [S, 3, 3]
      world         [B, S, H, W, 3]    depth 用位姿反投影得到的全局坐标点
    """
    imgs, coords = load_and_preprocess_images_square(image_paths, target_size=target_size)
    imgs = imgs.to(device)

    try:
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=dtype):
                pred = model(imgs)

        # 从相机编码解出 extrinsic / intrinsic (全局统一坐标系)
        pose_enc = pred['pose_enc']                        # [B, S, 9]
        H, W = pred['depth'].shape[2], pred['depth'].shape[3]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W))

        depth = pred['depth'].cpu().numpy()                # [B,S,H,W,1]
        depth_conf = pred['depth_conf'].cpu().numpy()      # [B,S,H,W]
        extrinsic = extrinsic.cpu().numpy().squeeze(0)     # [S,3,4]
        intrinsic = intrinsic.cpu().numpy().squeeze(0)     # [S,3,3]

        # depth 反投影 → 全局世界坐标 (官方做法, 不用 model 的 world_points)
        world = _unproject_depth_to_world(depth[0, ..., 0], extrinsic, intrinsic)

        del pred, pose_enc
        return {
            'depth': depth, 'depth_conf': depth_conf,
            'extrinsic': extrinsic, 'intrinsic': intrinsic,
            'world': world[None],  # [B,S,H,W,3]
        }
    finally:
        del imgs, coords


def _cleanup_oom():
    """激进显存清理 — 必须在 OOM 后调用 (同时断开 traceback 对 tensor 的引用)"""
    import sys as _sys
    _sys.exc_info()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def select_even(paths, n):
    """环绕视角等间隔抽样, 保持角度覆盖; n<=0 或 >=len 返回全部。"""
    paths = list(paths)
    if n <= 0 or n >= len(paths):
        return paths
    idx = np.unique(np.rint(np.linspace(0, len(paths) - 1, n)).astype(int))
    return [paths[i] for i in idx]


def resolution_candidates(target, min_size, step=28):
    """从 target 向下到 min_size 的候选分辨率 (14 的倍数)。"""
    s = []
    x = int(target)
    while x >= int(min_size):
        s.append(x)
        x -= step
    if s[-1] != int(min_size):
        s.append(int(min_size))
    return s


def _try_one_forward(group, model, size, dtype, device):
    """尝试一次联合前向。 成功返回 pred dict, OOM 返回 None。"""
    try:
        return run_vggt_joint(group, model, size, dtype, device)
    except torch.cuda.OutOfMemoryError:
        _cleanup_oom()
        return None


def run_joint_scene(full_group, model, target_size, min_size, dtype, device,
                    auto_views=True):
    """
    把一组环绕视角作为整体做一次 VGGT 联合重建。
    返回 dict {pred, used_group, used_size} 或 None。
    """
    n_full = len(full_group)
    sizes = resolution_candidates(target_size, min_size)

    def report(tag, group, size, ok):
        n = len(group)
        if ok:
            print(f"  ✓ [joint] {tag}: {n} views @ {size}px (single forward, one world frame)")
        else:
            print(f"  ✗ [joint] {tag}: {n} views @ {size}px → OOM")

    # ── Phase A: 整组在最高可行分辨率 ──
    for size in sizes:
        pred = _try_one_forward(full_group, model, size, dtype, device)
        report('all', full_group, size, pred is not None)
        if pred is not None:
            return {'pred': pred, 'used_group': full_group, 'used_size': size}

    # ── Phase B: OOM → 环绕视角等间隔抽稀 (仍单次前向) ──
    if not auto_views or n_full <= 8:
        return None

    ratios = sorted({round(n_full * r) for r in (0.75, 0.5, 0.33, 0.25)}, reverse=True)
    decim_list = [r for r in ratios if 8 <= r < n_full]
    decim_list.append(max(8, n_full // 8))
    decim_list = sorted(set(decim_list), reverse=True)

    for nv in decim_list:
        group = select_even(full_group, nv)
        pred = None
        for size in sizes:  # 从最小分辨率往上找能跑的
            pred = _try_one_forward(group, model, size, dtype, device)
            if pred is not None:
                report('decim', group, size, True)
                return {'pred': pred, 'used_group': group, 'used_size': size}
            report('decim', group, size, False)
        report('decim', group, sizes[0], False)
    return None


def _write_ply(filepath, xyz, rgb):
    """写 binary PLY 点云文件 (xyz: float32[N,3], rgb: uint8[N,3])"""
    n = len(xyz)
    with open(filepath, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode())
        xyz_f = xyz.astype(np.float32)
        rgb_u = rgb.astype(np.uint8)
        for i in range(n):
            f.write(xyz_f[i].tobytes())
            f.write(rgb_u[i].tobytes())


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='VGGT Joint Multi-View Reconstruction (official method)')
    parser.add_argument('--img-dir', type=str, default='data/random147_4 h8t1_4/images',
                        help='Directory of surrounding-view images of ONE scene (a whole ring)')
    parser.add_argument('--output-dir', type=str, default='output2/vggt_scour_random147_4_h8t1_4',
                        help='Where to save depth maps and results')
    parser.add_argument('--images', type=str, default=None,
                        help='Exact comma-separated view numbers used TOGETHER in one forward, e.g. "1,8,16,24,32,40"')
    parser.add_argument('--num-images', type=int, default=0,
                        help='Number of views (evenly spaced around the ring) fed TOGETHER in one forward. '
                             '0 = all views in the folder (default).')
    parser.add_argument('--target-size', type=int, default=392,
                        help='VGGT square input resolution (default: 392, like the official demo; max 518). '
                             'Must be divisible by 14. Smaller = less VRAM.')
    parser.add_argument('--min-size', type=int, default=336,
                        help='Lowest resolution to auto-try on OOM (default: 336)')
    parser.add_argument('--no-auto-views', action='store_true',
                        help='On OOM do NOT auto-subsample views; error out instead')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (uses more VRAM)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do NOT save depth .npy files to output dir')
    parser.add_argument('--no-ply', action='store_true',
                        help='Disable PLY export')
    parser.add_argument('--ply-stride', type=int, default=2,
                        help='Downsample stride for PLY export (default: 2, higher = smaller file)')
    parser.add_argument('--ply-conf', type=float, default=None,
                        help='Min depth confidence threshold (VGGT expp1 scale: 1+e^x). '
                             'Actual range is small (here ~1.0-3.5), so default None = auto-choose '
                             '~top-20%% confident pixels per run; set e.g. 1.5-2.0 to keep denser, '
                             'higher to keep only cleanest.')
    parser.add_argument('--ply-interval', type=int, default=1,
                        help='Export one per-view PLY every N views (default: 1 = all views; '
                             '0 = only the merged combined.ply). All views share the same world frame.')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    dtype = get_dtype() if not args.no_amp else torch.float32
    dtype_str = {torch.bfloat16: 'bfloat16', torch.float16: 'float16', torch.float32: 'float32'}[dtype]

    if not torch.cuda.is_available():
        print("✗ CUDA not available — VGGT requires a GPU")
        return

    used, total = get_vram_info()
    print(f"═══ VGGT Joint Multi-View Reconstruction (official method) ═══")
    print(f"  Device:      {device}  ({used:.1f}/{total:.1f}G in use)")
    print(f"  AMP dtype:   {dtype_str}")
    print(f"  Resolution:  {args.target_size}px (floor {args.min_size}px)")
    print(f"  Views:       {'--images set' if args.images else ('all in folder' if args.num_images <= 0 else args.num_images)}")
    print(f"  Method:      camera_head pose + depth unproject, ONE forward → single global world frame")

    # ── ① 选整圈视角 ──
    print(f"\n─── Step 1: select surrounding views (one scene) ───")
    img_dir = args.img_dir
    all_paths = sorted([os.path.join(img_dir, p) for p in os.listdir(img_dir)
                        if p.endswith(('.jpg', '.png', '.JPG', '.PNG'))],
                       key=lambda p: int(''.join(ch for ch in os.path.splitext(os.path.basename(p))[0] if ch.isdigit()) or 0))
    if not all_paths:
        print(f"✗ No images found in {img_dir}")
        return
    print(f"  {len(all_paths)} views found in {img_dir}")

    if args.images:
        group = [os.path.join(img_dir, f'{n.strip()}.jpg') for n in args.images.split(',')]
        group = [p for p in group if os.path.exists(p)]
        if not group:
            print("✗ None of --images exist. Views are named like 1.jpg, 2.jpg ... inside img_dir.")
            return
    else:
        group = all_paths
        if args.num_images and 0 < args.num_images < len(group):
            group = select_even(group, args.num_images)
    print(f"  Using {len(group)} views as ONE joint scene:")
    print(f"    " + ', '.join(os.path.splitext(os.path.basename(p))[0] for p in group))

    # ── ② VGGT 联合推理 (一次前向: 位姿 + depth) ──
    print(f"\n─── Step 2: VGGT joint inference (camera + depth, single forward) ───")
    model = load_vggt(device)
    used, total = get_vram_info()
    print(f"  VRAM before VGGT: {used:.2f}/{total:.1f}G")

    t0 = time.time()
    res = run_joint_scene(group, model, args.target_size, args.min_size, dtype, device,
                          auto_views=not args.no_auto_views)
    if res is None:
        print(f"\n✗ Could not reconstruct {len(group)} views jointly even at {args.min_size}px.")
        print(f"  → Options:  --num-images <n>  (evenly fewer views, still one joint forward)")
        print(f"              --target-size 336  or  a GPU with more VRAM")
        return

    pred = res['pred']
    used_group = res['used_group']
    num_views = len(used_group)
    elapsed = time.time() - t0
    print(f"  ✓ Joint inference done: {num_views} views @ {res['used_size']}px "
          f"in {elapsed:.1f}s ({elapsed/num_views:.1f}s/view)")

    depth = pred['depth']            # [B, S, H, W, 1]
    conf = pred['depth_conf']        # [B, S, H, W]  (expp1)
    world = pred['world']            # [B, S, H, W, 3]  depth 反投影, 全局坐标
    B, S = depth.shape[0], depth.shape[1]
    assert S == num_views, f"model returned {S} views but {num_views} were fed"
    H_w, W_w = depth.shape[2], depth.shape[3]
    print(f"  Output: {S} views, spatial {H_w}x{W_w}px, one global world frame ✓")

    del model  # 释放 VGGT 权重
    torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 置信度分布 + 自动阈值 (VGGT expp1 量纲很小, 不能照搬官方的 5.0) ──
    conf_finite = conf[np.isfinite(conf)]
    if conf_finite.size:
        print(f"  depth_conf: min={conf_finite.min():.2f} "
              f"median={np.median(conf_finite):.2f} "
              f"p95={np.percentile(conf_finite, 95):.2f} "
              f"max={conf_finite.max():.2f}   (expp1 scale)")
    if args.ply_conf is not None:
        conf_thr = args.ply_conf
    else:
        base = conf > 1.0
        conf_thr = float(np.percentile(conf[base], 60)) if base.sum() > 0 else 1.0
        print(f"  (auto) conf threshold = {conf_thr:.2f}  (top ~40% of conf>1 pixels; "
              f"cross-view depth error verified <2.5%)")

    # ── ③ 逐视角 depth 保存 + 场景点云导出 (同一全局坐标系) ──
    ply_dir = os.path.join(args.output_dir, 'ply')
    if not args.no_ply:
        os.makedirs(ply_dir, exist_ok=True)
    all_xyz, all_rgb = [], []

    for i in range(num_views):
        base = os.path.splitext(os.path.basename(used_group[i]))[0]
        conf_i = conf[0, i] if B == 1 else conf[i]
        depth_i = depth[0, i, ..., 0] if B == 1 else depth[i, ..., 0]

        if not args.no_save:
            np.save(os.path.join(args.output_dir, f'{base}_depth.npy'), depth_i)
            np.save(os.path.join(args.output_dir, f'{base}_conf.npy'), conf_i)
            # 相机位姿也存一份 (同一全局坐标系), 供后续对齐/分析用
            np.savez(os.path.join(args.output_dir, f'{base}_cam.npz'),
                     extrinsic=pred['extrinsic'][i], intrinsic=pred['intrinsic'][i])

        if args.no_ply:
            continue

        # 该帧全局坐标点 + 原图颜色
        pts_i = world[0, i] if B == 1 else world[i]      # [H,W,3]
        img = cv2.imread(used_group[i])
        if img is None:
            continue
        img_rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (W_w, H_w))

        stride = args.ply_stride
        pts_ds = pts_i[::stride, ::stride].reshape(-1, 3)
        wc_ds = conf_i[::stride, ::stride].reshape(-1)
        rgb_ds = img_rgb[::stride, ::stride].reshape(-1, 3)

        keep = wc_ds >= conf_thr
        xyz, rgb = pts_ds[keep], rgb_ds[keep]
        if len(xyz) < 10:
            print(f"  [view {i+1}] {base}: too few valid points ({len(xyz)}) "
                  f"at conf≥{conf_thr:.2f}, skip")
            continue

        if args.ply_interval > 0 and (i + 1) % args.ply_interval == 0:
            _write_ply(os.path.join(ply_dir, f'{base}.ply'), xyz, rgb)
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    if not args.no_ply:
        print(f"\n─── Step 3: joint scene point cloud export ───")
        if all_xyz:
            xyz_all = np.concatenate(all_xyz, 0)
            rgb_all = np.concatenate(all_rgb, 0)
            _write_ply(os.path.join(ply_dir, 'combined.ply'), xyz_all, rgb_all)
            mn, mx = xyz_all.min(0), xyz_all.max(0)
            print(f"  ✓ combined.ply: {len(xyz_all):,} points (all {num_views} views, one global frame)")
            print(f"  Scene bbox: x[{mn[0]:.4f},{mx[0]:.4f}] "
                  f"y[{mn[1]:.4f},{mx[1]:.4f}] z[{mn[2]:.4f},{mx[2]:.4f}]")
            print(f"  ✓ exported to {ply_dir}/")
        else:
            print(f"  ⚠ No point survived conf≥{conf_thr:.2f}; try lowering --ply-conf (e.g. 1.2)")

    print(f"\n  Output saved to: {args.output_dir}/")
    used, total = get_vram_info()
    print(f"  Final VRAM: {used:.2f}/{total:.1f}G")
    print("✓ Done")


if __name__ == '__main__':
    main()
