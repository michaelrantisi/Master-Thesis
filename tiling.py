import os
import numpy as np
import cv2

def list_images(folder, exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(root, f))
    return sorted(out)

def read_rgb01(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)

def write_rgb01(path, rgb01):
    rgb = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def tissue_fraction(rgb01, bg_thresh=0.95):
    gray = rgb01.mean(axis=2)
    return float((gray < bg_thresh).mean())

def tile_image(rgb01, tile=256, stride=128):
    H, W, _ = rgb01.shape
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            patch = rgb01[y:y+tile, x:x+tile]
            if patch.shape[0] == tile and patch.shape[1] == tile:
                yield patch, (y, x)

def aug_geometric(rgb01, rng):
    # random flip
    if rng.random() < 0.5:
        rgb01 = rgb01[:, ::-1, :]
    if rng.random() < 0.5:
        rgb01 = rgb01[::-1, :, :]

    # random rotation 0/90/180/270
    k = rng.integers(0, 4)
    if k:
        rgb01 = np.rot90(rgb01, k, axes=(0, 1)).copy()

    return rgb01

def aug_photometric(rgb01, rng):
    # mild brightness/contrast jitter (keep realistic)
    a = float(rng.uniform(0.9, 1.1))  # contrast
    b = float(rng.uniform(-0.05, 0.05))  # brightness shift
    out = np.clip(a * rgb01 + b, 0.0, 1.0)

    # mild gaussian noise
    if rng.random() < 0.5:
        noise = rng.normal(0.0, 0.01, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)

    return out
