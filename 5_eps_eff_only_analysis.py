import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from tqdm import tqdm
import pandas as pd

from tiling import list_images
from mmf import eps_eff_map_from_rgb
HEALTHY_DIR = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\healthy"
UNHEALTHY_DIR = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\unhealthy"

OUT_DIR = r".\runs_eps_eff"
os.makedirs(OUT_DIR, exist_ok=True)
EPS_CLIP = (1.0, 20.0)
EPS_HOST = 2.0
EPS_CHANNELS = (3.0, 4.0, 5.0)


def read_rgb01(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def process_group(paths, label_name):
    rows = []

    for p in tqdm(paths, desc=f"Processing {label_name}"):
        rgb = read_rgb01(p)

        eps = eps_eff_map_from_rgb(
            rgb,
            eps_host=EPS_HOST,
            eps_channels=EPS_CHANNELS,
            clip=EPS_CLIP,
        )

        rows.append({
            "label": label_name,
            "eps_mean": float(eps.mean()),
            "eps_p95": float(np.percentile(eps, 95)),
            "eps_std": float(eps.std()),
        })

    return pd.DataFrame(rows)


def main():
    healthy_paths = list_images(HEALTHY_DIR, exts=(".jpg", ".jpeg", ".png"))
    unhealthy_paths = list_images(UNHEALTHY_DIR, exts=(".jpg", ".jpeg", ".png"))

    if len(healthy_paths) == 0 or len(unhealthy_paths) == 0:
        raise RuntimeError("No tiles found")

    df_h = process_group(healthy_paths, "healthy")
    df_u = process_group(unhealthy_paths, "unhealthy")

    df = pd.concat([df_h, df_u], ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, "eps_eff_tile_stats.csv"), index=False)

    print("\nSaved eps_eff_tile_stats.csv")

    # Statistical tests 
    for col in ["eps_mean", "eps_p95", "eps_std"]:
        h = df_h[col].values
        u = df_u[col].values

        ks = ks_2samp(h, u)

        print(f"\n=== ε_eff metric: {col} ===")
        print("Healthy mean ± std:", h.mean(), h.std())
        print("Unhealthy mean ± std:", u.mean(), u.std())
        print("KS D:", ks.statistic, "p:", ks.pvalue)

        # Histogram
        plt.figure()
        plt.hist(h, bins=60, alpha=0.7, label="Healthy")
        plt.hist(u, bins=60, alpha=0.7, label="Unhealthy")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{col}_hist.png"), dpi=200)
        plt.close()

    print("\nSaved ε_eff histograms in:", OUT_DIR)


if __name__ == "__main__":
    main()
