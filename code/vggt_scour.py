#!/usr/bin/env python3
"""
VGGT 稠密重建 — Mask引导输入 + 显存优化
=========================================
策略:
  1. AMP 混合精度 (bfloat16/fp16) → 显存 ~50%
  2. 降低 VGGT 输入分辨率 (square pad → target_size)
  3. 分批推理 (--batch-size) → 全局注意力 token 数骤降
  4. 单图回退 (--single) → OOM 的最后防线
  5. DPT head 内置 frames_chunk_size 分段

用法:
  python vggt_scour_random_145_h8t1_4.py                          # 默认: 全量推理, 434px, PLY+合并
  python vggt_scour_random_145_h8t1_4.py --target-size 336        # 更低分辨率 (更省显存)
  python vggt_scour_random_145_h8t1_4.py --batch-size 6           # 分批模式 (省显存, 但坐标系不一致)
  python vggt_scour_random_145_h8t1_4.py --single                 # 逐张推理 (最省显存)
  python vggt_scour_random_145_h8t1_4.py --num-images 8           # 只用8张图
  python vggt_scour_random_145_h8t1_4.py --no-ply                 # 不导出 PLY
"""

# ⚠️ 必须在 import torch 之前设置，减少显存碎片
import os, sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random, argparse, gc, time
from collections import defaultdict

sys.path.insert(0, '/home/wyh/Desktop/wyh/code')
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square


# ═══════════════════════════════════════════════════════════════════
# ScourUNet — 预测冲刷坑 mask (IoU 0.879)
# ═══════════════════════════════════════════════════════════════════

class ScourUNet(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet34(weights=None)
        if os.path.exists('resnet34-b627a593.pth'):
            resnet.load_state_dict(torch.load('resnet34-b627a593.pth', map_location='cpu', weights_only=True))
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
        self.up4 = self._up(768, 256)
        self.up3 = self._up(384, 128)
        self.up2 = self._up(192, 64)
        self.up1 = self._up(128, 64)
        self.out = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 1, 1))

    def _up(self, ic, oc):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d4 = self.up4(torch.cat([e5, F.interpolate(e4, e5.shape[2:], mode='bilinear', align_corners=False)], 1))
        d3 = self.up3(torch.cat([d4, F.interpolate(e3, d4.shape[2:], mode='bilinear', align_corners=False)], 1))
        d2 = self.up2(torch.cat([d3, F.interpolate(e2, d3.shape[2:], mode='bilinear', align_corners=False)], 1))
        d1 = self.up1(torch.cat([d2, F.interpolate(e1, d2.shape[2:], mode='bilinear', align_corners=False)], 1))
        o = self.out(d1)
        if o.shape[2:] != x.shape[2:]:
            o = F.interpolate(o, x.shape[2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(o)


# ═══════════════════════════════════════════════════════════════════
# Mask 注入 — 坑内原值, 床面降对比度
# ═══════════════════════════════════════════════════════════════════

def apply_mask(img_rgb, mask, bed_weight=0.4, mode='edge'):
    """mask 加权注入
       mode='edge':  坑外高斯模糊 (保留颜色结构→位姿匹配, 抑制纹理→聚焦坑内) [推荐]
       mode='blend': 坑外拉向均值 (更激进, 可能损害位姿)
       bed_weight: 注入强度, 1.0=无注入, 0.3-0.5=推荐"""
    img = np.array(img_rgb).astype(np.float32)
    w = np.clip(mask, bed_weight, 1.0)
    if mode == 'edge':
        # 坑外: 高斯模糊 → 保留大尺度颜色梯度(位姿用), 抹掉小纹理(减少干扰)
        blurred = cv2.GaussianBlur(img, (21, 21), 10)
        blended = img * w[:, :, None] + blurred * (1 - w[:, :, None])
    else:
        # 坑外: 拉向均值
        mean = img.mean(axis=(0, 1), keepdims=True)
        blended = img * w[:, :, None] + mean * (1 - w[:, :, None])
    return blended.clip(0, 255).astype(np.uint8)


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
        # 尝试 huggingface 缓存
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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device).eval()
    return model


def run_vggt_batch(image_paths, model, target_size, dtype, device='cuda'):
    """
    对一批图像运行 VGGT 推理。
    返回: dict with keys depth, depth_conf, world_points, world_points_conf
    """
    imgs, coords = load_and_preprocess_images_square(image_paths, target_size=target_size)
    imgs = imgs.to(device)

    try:
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=dtype):
                pred = model(imgs)

        result = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                  for k, v in pred.items()}
        del pred
        return result
    finally:
        # 确保即使 OOM 也释放输入 tensor
        del imgs, coords


def run_vggt_batched(all_paths, model, target_size, batch_size, dtype, device='cuda'):
    """
    分批运行 VGGT，返回合并后的预测结果。
    每批独立推理，深度尺度可能不同（每批有自己的坐标系）。
    """
    all_predictions = []
    num_batches = (len(all_paths) + batch_size - 1) // batch_size

    for bi in range(num_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(all_paths))
        batch_paths = all_paths[start:end]

        used, total = get_vram_info()
        print(f"  Batch {bi+1}/{num_batches}: {len(batch_paths)} images "
              f"[{start}:{end}] | VRAM: {used:.1f}/{total:.1f}G")

        try:
            pred = run_vggt_batch(batch_paths, model, target_size, dtype, device)
            all_predictions.append(pred)
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ OOM on batch {bi+1}! Falling back to single-image...")
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            for j, p in enumerate(batch_paths):
                try:
                    pred = run_vggt_batch([p], model, target_size, dtype, device)
                    all_predictions.append(pred)
                    gc.collect(); torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    print(f"  ✗ OOM on single image {os.path.basename(p)}, skipping")
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    return all_predictions


def merge_predictions(pred_list):
    """合并分批预测结果: 沿 S 维度拼接"""
    valid = [p for p in pred_list if p is not None]
    if not valid:
        return None

    keys = ['depth', 'depth_conf', 'world_points', 'world_points_conf']
    merged = {}
    for k in keys:
        if k in valid[0]:
            merged[k] = np.concatenate([p[k] for p in valid], axis=1)  # axis=1 = S dim
    return merged


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
    parser = argparse.ArgumentParser(description='VGGT Scour Pit Reconstruction (Memory-Optimized)')
    parser.add_argument('--target-size', type=int, default=434,
                        help='VGGT square input size (default: 434, original: 518, min: 336). '
                             'Must be divisible by 14. Smaller = less VRAM.')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Images per VGGT forward pass (default: 0 = all-at-once, globally consistent). '
                             'Set to 6 for batched mode (less VRAM, but batched views have different coordinate frames).')
    parser.add_argument('--single', action='store_true',
                        help='Process each image independently (minimal VRAM)')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-detect optimal batch size (starts small, increases if VRAM allows)')
    parser.add_argument('--num-images', type=int, default=16,
                        help='Number of random images to select (default: 16)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (uses more VRAM)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img-dir', type=str, default='data/random_140_H6_T1_6/images')
    parser.add_argument('--output-dir', type=str, default='output/vggt_scour_random_140_H6_T1_6',
                        help='Where to save depth maps and results')
    parser.add_argument('--no-save', action='store_true',
                        help='Do NOT save depth .npy files to output dir')
    parser.add_argument('--ply', action='store_true', default=True,
                        help='Export PLY point cloud from VGGT world_points (default: on)')
    parser.add_argument('--no-ply', action='store_true',
                        help='Disable PLY export')
    parser.add_argument('--ply-stride', type=int, default=2,
                        help='Downsample stride for PLY export (default: 2, higher = smaller file)')
    parser.add_argument('--ply-conf', type=float, default=0.5,
                        help='Min confidence for PLY points (default: 0.5)')
    parser.add_argument('--ply-mask', type=float, default=0.3,
                        help='Min mask value for PLY points (default: 0.3)')
    parser.add_argument('--no-per-view', action='store_true',
                        help='Do NOT export individual per-view PLY files')
    parser.add_argument('--ply-combined', action='store_true',
                        help='Also export a combined PLY (all views merged)')
    parser.add_argument('--mask-model', type=str, default='result/checkpoints/best_model_random_140_H6_T1_6.pth')
    parser.add_argument('--mask-weight', type=float, default=0.4,
                        help='Mask injection weight for non-pit areas (default: 0.4). '
                             '1.0=no injection, 0.4=moderate, 0.15=aggressive (may hurt pose). '
                             'Lower = more pit focus but worse camera pose estimation.')
    parser.add_argument('--no-mask', action='store_true',
                        help='Skip mask injection entirely (use raw RGB for VGGT, mask only for depth analysis)')
    parser.add_argument('--mask-mode', type=str, default='edge', choices=['edge', 'blend'],
                        help='Mask injection mode: edge=Gaussian blur non-pit (preserve color structure), '
                             'blend=pull non-pit toward mean (aggressive)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    dtype = get_dtype() if not args.no_amp else torch.float32
    dtype_str = {torch.bfloat16: 'bfloat16', torch.float16: 'float16', torch.float32: 'float32'}[dtype]

    print(f"═══ VGGT Scour Reconstruction ═══")
    print(f"  Device:      {device}")
    print(f"  AMP dtype:   {dtype_str}")
    print(f"  Target size: {args.target_size}px")
    print(f"  Batch size:  {'single' if args.single else args.batch_size}")
    print(f"  Num images:  {args.num_images}")
    print(f"  Mask weight: {'off' if args.no_mask else args.mask_weight} "
          f"({'raw RGB' if args.no_mask else '1.0=no injection, <0.3=aggressive}'})")

    # ── ① ScourUNet mask prediction ──
    print("\n─── Step 1: ScourUNet mask prediction ───")
    seg = ScourUNet().to(device).eval()
    seg.load_state_dict(torch.load(args.mask_model, map_location=device, weights_only=True))
    print("  ScourUNet loaded (IoU 0.879)")

    img_dir = args.img_dir
    all_paths = sorted([os.path.join(img_dir, p) for p in os.listdir(img_dir) if p.endswith(('.jpg', '.png', '.JPG', '.PNG'))])
    if len(all_paths) < args.num_images:
        print(f"  Warning: only {len(all_paths)} images found (requested {args.num_images})")
        args.num_images = len(all_paths)

    random.seed(args.seed)
    paths = random.sample(all_paths, args.num_images)
    print(f"  Selected {len(paths)} images")

    # 预测 mask + 注入
    tmp_dir = '/tmp/vggt_masked'
    os.makedirs(tmp_dir, exist_ok=True)
    masks = {}

    for i, p in enumerate(paths):
        img = cv2.imread(p)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_s = cv2.resize(rgb, (512, 512))
        t = torch.from_numpy(rgb_s).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        with torch.no_grad():
            mask = seg(t).squeeze().cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        masks[os.path.basename(p)] = mask

        # Mask 注入 (或跳过)
        if args.no_mask:
            masked = rgb  # 原始 RGB，不做注入
        else:
            masked = apply_mask(rgb, mask, bed_weight=args.mask_weight, mode=args.mask_mode)
        # 用 PNG 无损保存
        out_name = os.path.splitext(os.path.basename(p))[0] + '.png'
        cv2.imwrite(os.path.join(tmp_dir, out_name),
                    cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
        if (i + 1) % 8 == 0:
            print(f"  Mask prediction: {i+1}/{len(paths)}")

    print(f"  ✓ All {len(paths)} masks predicted + injected")

    # 释放 ScourUNet (后续只用 VGGT)
    del seg
    torch.cuda.empty_cache()

    # ── ② VGGT 推理 ──
    print(f"\n─── Step 2: VGGT inference ───")
    used, total = get_vram_info()
    print(f"  VRAM before VGGT: {used:.2f}/{total:.1f}G")

    model = load_vggt(device)
    tmp_paths = [os.path.join(tmp_dir, os.path.splitext(os.path.basename(p))[0] + '.png') for p in paths]

    t0 = time.time()

    # ── Auto-detect batch size ──
    if args.auto and not args.single:
        print(f"  Auto-detecting optimal batch size...")
        candidate_sizes = [8, 6, 4, 2, 1]
        best_bs = 1
        for bs in candidate_sizes:
            if bs > len(tmp_paths):
                continue
            try:
                test_paths = tmp_paths[:bs]
                _ = run_vggt_batch(test_paths, model, args.target_size, dtype, device)
                torch.cuda.empty_cache()
                best_bs = bs
                print(f"    ✓ batch_size={bs} OK")
                break  # 找到能跑的就行
            except torch.cuda.OutOfMemoryError:
                print(f"    ✗ batch_size={bs} OOM, trying smaller...")
                torch.cuda.empty_cache()
                gc.collect()
        args.batch_size = best_bs
        print(f"  → Using batch_size={args.batch_size}")

    def _cleanup_oom():
        """激进显存清理 — 必须在 OOM 后调用"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if args.single or args.batch_size == 1:
        # 逐张推理
        print(f"  Mode: single-image inference")
        pred_list = []
        for i, tp in enumerate(tmp_paths):
            used, total = get_vram_info()
            print(f"  [{i+1}/{len(tmp_paths)}] {os.path.basename(tp)} | VRAM: {used:.2f}/{total:.1f}G")
            try:
                pred = run_vggt_batch([tp], model, args.target_size, dtype, device)
                pred_list.append(pred)
                _cleanup_oom()
            except torch.cuda.OutOfMemoryError:
                print(f"  ✗ OOM, skipping {os.path.basename(tp)}")
                pred_list.append(None)
                _cleanup_oom()
    elif args.batch_size > 0:
        # 分批推理
        print(f"  Mode: batched ({args.batch_size} images/batch)")
        pred_list = run_vggt_batched(tmp_paths, model, args.target_size, args.batch_size, dtype, device)
    else:
        # 全量推理 — 所有图一次性送入，确保全局一致的坐标系
        print(f"  Mode: all-at-once ({len(tmp_paths)} images, {args.target_size}px)")

        def _try_all_at_once(size):
            """尝试全量推理，成功返回 pred_list，OOM 返回 None"""
            nonlocal model
            try:
                pred = run_vggt_batch(tmp_paths, model, size, dtype, device)
                return [pred]
            except torch.cuda.OutOfMemoryError:
                # 彻底清理: 删模型 + 清 traceback (防止 Python 持有 tensor 引用)
                import sys as _sys
                _sys.exc_info()  # 消费当前异常，断开 traceback → tensor 引用链
                if 'model' in dir():
                    del model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return None

        pred_list = _try_all_at_once(args.target_size)

        if pred_list is None:
            # 降分辨率，找到能跑的最大分辨率
            candidates = [s for s in [392, 378, 364, 336]
                          if s < args.target_size and s % 14 == 0]
            for fs in candidates:
                used, total = get_vram_info()
                print(f"  ⚠ OOM at previous size → reloading model + trying {fs}px "
                      f"(VRAM before reload: {used:.2f}/{total:.1f}G)")
                model = load_vggt(device)
                pred_list = _try_all_at_once(fs)
                if pred_list is not None:
                    print(f"  ✓ All {len(tmp_paths)} images globally aligned at {fs}px!")
                    break
                else:
                    print(f"  ✗ {fs}px still OOM")

            if pred_list is None:
                print(f"  ✗ All-at-once not possible on this GPU (11.6 GB) with {len(tmp_paths)} images.")
                print(f"  → Use --batch-size 6 (batched) or fewer images: --num-images 8")
                pred_list = []

    elapsed = time.time() - t0
    print(f"  ✓ VGGT inference done in {elapsed:.1f}s ({elapsed/len(paths):.1f}s/img)")

    # 合并结果
    predictions = merge_predictions(pred_list)
    if predictions is None:
        print("✗ No valid predictions!")
        return

    depth = predictions['depth']       # [B, S, H, W, 1]  — S = 成功的图像数
    conf = predictions['depth_conf']   # [B, S, H, W]
    num_successful = depth.shape[1]
    print(f"  Depth shape: {depth.shape}, Conf shape: {conf.shape}")
    print(f"  Successful images: {num_successful}/{len(paths)}")

    if num_successful == 0:
        print("✗ All images failed!")
        return

    if num_successful < len(paths):
        print(f"  ⚠ {len(paths) - num_successful} images failed due to OOM — skipped in analysis")

    if not args.single and args.batch_size > 1 and args.batch_size < len(paths):
        print(f"  ⚠ Note: batched mode → depth SCALES differ between batches "
              f"(each batch has own coordinate frame). Per-image statistics are valid, "
              f"but absolute depth values are NOT comparable across batches.")

    # ── ③ 提取坑内深度统计 ──
    print(f"\n─── Step 3: Scour pit depth analysis ───")
    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(num_successful):
        name = os.path.basename(paths[i])
        mask = masks.get(name)
        if mask is None:
            continue

        # 对齐 mask 到 depth 分辨率
        mask_resized = cv2.resize(mask, (depth.shape[3], depth.shape[2]))
        conf_i = conf[0, i] if depth.shape[0] == 1 else conf[i]
        depth_i = depth[0, i, ..., 0] if depth.shape[0] == 1 else depth[i, ..., 0]

        valid = (mask_resized > 0.3) & (conf_i > 0.3)
        if valid.sum() > 10:
            sd = depth_i[valid]
            results.append({
                'name': name,
                'mean': float(sd.mean()),
                'std': float(sd.std()),
                'median': float(np.median(sd)),
                'min': float(sd.min()),
                'max': float(sd.max()),
                'valid_pixels': int(valid.sum()),
            })
            print(f"  {name}: mean={sd.mean():.4f} ± {sd.std():.4f} "
                  f"(median={np.median(sd):.4f}, N={valid.sum()})")
        else:
            print(f"  {name}: insufficient valid pixels ({valid.sum()})")

        # 保存 depth map
        if not args.no_save:
            np.save(os.path.join(args.output_dir, f'{os.path.splitext(name)[0]}_depth.npy'), depth_i)
            np.save(os.path.join(args.output_dir, f'{os.path.splitext(name)[0]}_conf.npy'), conf_i)
            np.save(os.path.join(args.output_dir, f'{os.path.splitext(name)[0]}_mask.npy'), mask_resized)

    # ── ④ PLY 导出 ──
    if not args.no_ply:
        print(f"\n─── Step 4: PLY point cloud export ───")
        wp = predictions.get('world_points')       # [B, S, H, W, 3]
        wp_conf = predictions.get('world_points_conf')  # [B, S, H, W]

        if wp is None:
            print("  ⚠ No world_points in predictions! VGGT may not have point_head enabled.")
        else:
            ply_dir = os.path.join(args.output_dir, 'ply')
            os.makedirs(ply_dir, exist_ok=True)

            all_verts = []
            total_points = 0

            for i in range(num_successful):
                name = os.path.splitext(os.path.basename(paths[i]))[0]
                mask = masks.get(os.path.basename(paths[i]))
                if mask is None:
                    continue

                H_wp, W_wp = wp.shape[2], wp.shape[3]
                mask_wp = cv2.resize(mask, (W_wp, H_wp))

                pts_i = wp[0, i] if wp.shape[0] == 1 else wp[i]
                conf_i = wp_conf[0, i] if wp_conf.shape[0] == 1 else wp_conf[i]

                img = cv2.imread(paths[i])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (W_wp, H_wp))

                stride = args.ply_stride
                pts_ds = pts_i[::stride, ::stride].reshape(-1, 3)
                conf_ds = conf_i[::stride, ::stride].reshape(-1)
                mask_ds = mask_wp[::stride, ::stride].reshape(-1)
                color_ds = img_rgb[::stride, ::stride].reshape(-1, 3)

                valid = (conf_ds > args.ply_conf) & (mask_ds > args.ply_mask)
                xyz = pts_ds[valid]
                rgb = color_ds[valid]

                if len(xyz) < 10:
                    print(f"  {name}: too few valid points ({len(xyz)}), skipping")
                    continue

                # 每张视图单独 PLY
                if not args.no_per_view:
                    ply_path = os.path.join(ply_dir, f'{name}.ply')
                    _write_ply(ply_path, xyz, rgb)

                total_points += len(xyz)
                all_verts.append((xyz, rgb))

            # 可选合并版
            if args.ply_combined and all_verts:
                xyz_all = np.concatenate([v[0] for v in all_verts], axis=0)
                rgb_all = np.concatenate([v[1] for v in all_verts], axis=0)
                combined_path = os.path.join(ply_dir, 'combined.ply')
                _write_ply(combined_path, xyz_all, rgb_all)

            if not args.no_per_view:
                print(f"  ✓ {num_successful} per-view PLY files exported to {ply_dir}/")
            if args.ply_combined:
                print(f"  ✓ combined.ply: {sum(len(v[0]) for v in all_verts):,} points")

            print(f"  ✓ {total_points:,} total points exported to {ply_dir}/")

    # ── ⑤ 汇总 ──
    print(f"\n─── Summary ───")
    if results:
        means = [r['mean'] for r in results]
        print(f"  Global scour depth: {np.mean(means):.4f} ± {np.std(means):.4f}")
        print(f"  Range: [{min(means):.4f}, {max(means):.4f}]")
        print(f"  Output saved to: {args.output_dir}/")

    used, total = get_vram_info()
    print(f"  Final VRAM: {used:.2f}/{total:.1f}G")
    print("✓ Done")


if __name__ == '__main__':
    main()
