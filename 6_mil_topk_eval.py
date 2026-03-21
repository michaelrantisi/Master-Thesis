import os
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

from data_manifest import load_manifest, split_by_source
from models import SmallConvAE
from mmf import eps_feature_stack

MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
AE_WEIGHTS = r".\runs_ae\ae_best_6ch.pt"

def read_rgb01(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)

def make_x6(path):
    rgb = read_rgb01(path)
    eps3 = eps_feature_stack(rgb)
    x6 = np.concatenate([rgb, eps3], axis=2)
    x6 = torch.from_numpy(x6).permute(2,0,1).float()
    return x6

def score_tiles(df, ae, device):
    scores = []
    with torch.no_grad():
        for i in tqdm(range(len(df)), desc="Scoring tiles"):
            p = df.loc[i, "path"]
            x = make_x6(p).unsqueeze(0).to(device)
            recon = ae(x)
            mse = ((recon - x) ** 2).mean().item()
            scores.append(mse)
    return np.array(scores, dtype=np.float64)

def topk_mean(x, k=10):
    x = np.sort(np.asarray(x))
    k = min(k, len(x))
    return float(x[-k:].mean())

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = load_manifest(MANIFEST)
    tr_df, va_df = split_by_source(df, test_size=0.2, seed=42)

    # We will CALIBRATE on TRAIN HEALTHY only 
    tr_healthy = tr_df[tr_df["label"] == "healthy"].reset_index(drop=True)
    va_all = va_df.reset_index(drop=True)

    ae = SmallConvAE(in_ch=6).to(device)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=device))
    ae.eval()

    # 1) Score calibration set (train healthy)
    tr_h_scores = score_tiles(tr_healthy, ae, device)
    mu = float(tr_h_scores.mean())
    sd = float(tr_h_scores.std() + 1e-12)
    print(f"Calibration (train healthy) mu={mu:.6f} sd={sd:.6f}")

    # 2) Score validation tiles (healthy+unhealthy)
    va_scores = score_tiles(va_all, ae, device)

    # 3) Two-sided anomaly score: absolute z-distance from healthy mean
    va_anom = np.abs((va_scores - mu) / sd)

    va_all = va_all.copy()
    va_all["mse"] = va_scores
    va_all["anom"] = va_anom

    # Aggregate per source image (true MIL bag)
    agg = va_all.groupby(["label", "src_id"]).agg(
        mean=("anom", "mean"),
        p95=("anom", lambda x: np.percentile(x, 95)),
        top10=("anom", lambda x: topk_mean(x, k=10)),
        max=("anom", "max"),
        n=("anom", "count"),
    ).reset_index()

    y = (agg["label"] == "unhealthy").astype(int).values

    for col in ["mean", "p95", "top10", "max"]:
        s = agg[col].values
        auc = roc_auc_score(y, s)
        sh = s[y == 0]
        su = s[y == 1]
        ks = ks_2samp(sh, su)
        print(f"\n=== CALIBRATED MIL per-src using {col} ===")
        print("AUROC:", float(auc))
        print("KS D:", float(ks.statistic), "p:", float(ks.pvalue))
        print("Healthy mean/std:", float(sh.mean()), float(sh.std()))
        print("Unhealthy mean/std:", float(su.mean()), float(su.std()))

    out_dir = r".\runs_mil"
    os.makedirs(out_dir, exist_ok=True)
    agg.to_csv(os.path.join(out_dir, "val_src_aggregates_calibrated.csv"), index=False)
    va_all.to_csv(os.path.join(out_dir, "val_tile_scores_calibrated.csv"), index=False)
    print("\nSaved:",
          os.path.join(out_dir, "val_src_aggregates_calibrated.csv"),
          "and val_tile_scores_calibrated.csv")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
