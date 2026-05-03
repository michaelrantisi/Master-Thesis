import os
import csv
import numpy as np
from tqdm import tqdm

from tiling import list_images, read_rgb01, write_rgb01, tile_image, tissue_fraction, aug_geometric, aug_photometric

HEALTHY_DIR = r"C:\Users\User\Desktop\thesis\Healthy choroid"
UNHEALTHY_DIR = r"C:\Users\User\Desktop\thesis\Unhealthy choroid"

OUT_ROOT = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out"
OUT_H = os.path.join(OUT_ROOT, "healthy")
OUT_U = os.path.join(OUT_ROOT, "unhealthy")
MANIFEST = os.path.join(OUT_ROOT, "manifest.csv")

TILE = 256
STRIDE = 256  # non-overlapping by default to match thesis text
MIN_TISSUE = 0.10
N_AUG = 2     # keep moderate; source-aware split preserves integrity

os.makedirs(OUT_H, exist_ok=True)
os.makedirs(OUT_U, exist_ok=True)


def process_class(src_dir, label_name, out_dir, writer, rng):
    paths = list_images(src_dir)
    if not paths:
        raise RuntimeError(f"No images found in: {src_dir}")

    saved = 0
    for p in tqdm(paths, desc=f"Tiling {label_name}"):
        src_id = os.path.splitext(os.path.basename(p))[0]
        img = read_rgb01(p)

        for patch, (y, x) in tile_image(img, tile=TILE, stride=STRIDE):
            if tissue_fraction(patch) < MIN_TISSUE:
                continue

            # save original tile first
            base_name = f"{label_name}__src-{src_id}__y{y:04d}_x{x:04d}__orig.jpg"
            out_path = os.path.join(out_dir, base_name)
            write_rgb01(out_path, patch)
            writer.writerow([label_name, src_id, y, x, -1, out_path])
            saved += 1

            for aug_i in range(N_AUG):
                rr = np.random.default_rng(rng.integers(0, 2**31 - 1))
                t = aug_geometric(patch.copy(), rr)
                t = aug_photometric(t, rr)
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
