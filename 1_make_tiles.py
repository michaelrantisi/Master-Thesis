import os
import csv
import numpy as np
from tqdm import tqdm

from tiling import (
    list_images, read_rgb01, write_rgb01,
    tile_image, tissue_fraction,
    aug_geometric, aug_photometric
)

# INPUTS: original images
HEALTHY_DIR = r"C:\Users\User\Desktop\thesis\Healthy choroid"
UNHEALTHY_DIR = r"C:\Users\User\Desktop\thesis\Unhealthy choroid"

# OUTPUTS
OUT_ROOT = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out"
OUT_H = os.path.join(OUT_ROOT, "healthy")
OUT_U = os.path.join(OUT_ROOT, "unhealthy")
MANIFEST = os.path.join(OUT_ROOT, "manifest.csv")

TILE = 256
STRIDE = 128
MIN_TISSUE = 0.10
N_AUG = 6 

os.makedirs(OUT_H, exist_ok=True)
os.makedirs(OUT_U, exist_ok=True)

def aug_microscopy(rgb01, rng):
    """
    Microscopy-friendly extra aug:
    - gamma
    - mild blur (focus)
    - mild brightness/contrast already in aug_photometric
    """
    x = rgb01

    # gamma correction
    gamma = float(rng.uniform(0.85, 1.15))
    x = np.clip(x, 0.0, 1.0) ** gamma

    # mild blur sometimes
    if rng.random() < 0.35:
        import cv2
        sigma = float(rng.uniform(0.0, 1.0))
        if sigma > 1e-6:
            x = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return np.clip(x, 0.0, 1.0).astype(np.float32)

def process_class(src_dir, label_name, out_dir, writer, rng):
    paths = list_images(src_dir)
    if len(paths) == 0:
        raise RuntimeError(f"No images found in: {src_dir}")

    saved = 0
    for p in tqdm(paths, desc=f"Tiling {label_name}"):
        src_id = os.path.splitext(os.path.basename(p))[0]
        img = read_rgb01(p)

        for patch, (y, x) in tile_image(img, tile=TILE, stride=STRIDE):
            if tissue_fraction(patch) < MIN_TISSUE:
                continue

            for aug_i in range(N_AUG):
                rr = np.random.default_rng(rng.integers(0, 2**31 - 1))
                t = aug_geometric(patch.copy(), rr)
                t = aug_photometric(t, rr)
                t = aug_microscopy(t, rr)

                fname = f"{label_name}__src-{src_id}__y{y:04d}_x{x:04d}__aug{aug_i:02d}.jpg"
                out_path = os.path.join(out_dir, fname)
                write_rgb01(out_path, t)

                writer.writerow([label_name, src_id, y, x, aug_i, out_path])
                saved += 1

    return saved

def main():
    rng = np.random.default_rng(42)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "src_id", "y", "x", "aug", "path"])

        n_h = process_class(HEALTHY_DIR, "healthy", OUT_H, writer, rng)
        n_u = process_class(UNHEALTHY_DIR, "unhealthy", OUT_U, writer, rng)

    print("Done.")
    print("Saved healthy tiles:", n_h)
    print("Saved unhealthy tiles:", n_u)
    print("Manifest:", MANIFEST)

if __name__ == "__main__":
    main()
