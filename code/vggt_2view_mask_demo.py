#!/usr/bin/env python3
"""VGGT 2-view mask demo — 1.jpg + 41.jpg 冲刷坑掩模重建"""
import os, cv2, numpy as np, torch, gc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys; sys.path.insert(0, '/home/wyh/Desktop/wyh/code')
from train.train_supervised import ScourUNet
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square

device = 'cuda'; gc.collect(); torch.cuda.empty_cache()
img_nums = [1, 41]
paths = [f'data/random_145_0_25_pure_currents/images/{n}.jpg' for n in img_nums]
print(f'Images: {img_nums}')

# Step 1: Mask
seg = ScourUNet().to(device).eval()
seg.load_state_dict(torch.load('result/checkpoints_supervised/best_seg_model.pth', map_location=device, weights_only=True))
tmp_dir = '/tmp/vggt_2mask'
os.makedirs(tmp_dir, exist_ok=True)
masks = {}
for p in paths:
    img = cv2.imread(p); h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(cv2.resize(rgb, (512, 512))).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad(): m = torch.sigmoid(seg(t)).squeeze().cpu().numpy()
    masks[os.path.basename(p)] = cv2.resize(m, (w, h))
    cv2.imwrite(os.path.join(tmp_dir, f'{os.path.basename(p).replace(".jpg",".png")}'),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
del seg; torch.cuda.empty_cache()

# Step 2: VGGT
model = VGGT()
model.load_state_dict(torch.load('/home/wyh/Downloads/model.pt', map_location=device, weights_only=True), strict=True)
model = model.to(device).eval()
tmp_paths = [os.path.join(tmp_dir, f'{n}.png') for n in img_nums]
for tsize in [434, 392, 364, 336]:
    try:
        imgs, _ = load_and_preprocess_images_square(tmp_paths, target_size=tsize)
        imgs = imgs.to(device)
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(imgs)
        print(f'VGGT OK at {tsize}px'); break
    except torch.cuda.OutOfMemoryError:
        del imgs; gc.collect(); torch.cuda.empty_cache()

wp = pred['world_points'].cpu().numpy()
wp_conf = pred['world_points_conf'].cpu().numpy()
H, W = wp.shape[2], wp.shape[3]
del model, imgs, pred; torch.cuda.empty_cache()

# Step 3: Extract + merge into single point cloud
out_dir = 'output/vggt_2view_mask_demo'
os.makedirs(out_dir, exist_ok=True)

# Collect all valid points from both views
all_xyz, all_rgb = [], []
for i, p in enumerate(paths):
    n = os.path.splitext(os.path.basename(p))[0]
    mask = masks[os.path.basename(p)]
    mask_wp = cv2.resize(mask, (W, H))
    pts_i = wp[0, i]
    img_rgb = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (W, H))
    pts_ds = pts_i[::2, ::2].reshape(-1, 3)
    mask_ds = mask_wp[::2, ::2].reshape(-1)
    color_ds = img_rgb[::2, ::2].reshape(-1, 3)
    keep = mask_ds > 0.5

    # Color-code by view: view 0 = red, view 1 = cyan
    colors = color_ds.copy()
    if i == 0:
        colors[:, 0] = 255  # red for view 1 (1.jpg)
    else:
        colors[:, 1:] = 255  # cyan for view 2 (41.jpg)

    all_xyz.append(pts_ds[keep])
    all_rgb.append(colors[keep])
    print(f'{n}: {keep.sum():,} pts (color={"red" if i==0 else "cyan"})')

xyz = np.concatenate(all_xyz).astype(np.float32)
rgb = np.concatenate(all_rgb).astype(np.uint8)

# Write one PLY — red=1.jpg view, cyan=41.jpg view
with open(f'{out_dir}/combined.ply', 'wb') as f:
    n = len(xyz)
    f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
    for i in range(n):
        f.write(xyz[i].tobytes())
        f.write(rgb[i].tobytes())

print(f'combined.ply: {n:,} pts (red=1.jpg, cyan=41.jpg)')
print(f'Done -> {out_dir}/')
