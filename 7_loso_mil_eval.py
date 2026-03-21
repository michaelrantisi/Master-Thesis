import os
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from models import SmallConvAE
from mmf import eps_feature_stack
from data_manifest import load_manifest

MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
AE_WEIGHTS = r".\runs_ae\ae_best_6ch.pt"

OUT_DIR = r".\runs_loso"
os.makedirs(OUT_DIR, exist_ok=True)


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
    x6 = torch.from_numpy(x6).permute(2, 0, 1).float()
    return x6


@torch.no_grad()
def score_df(df, ae, device, desc):
    scores = []
    for i in tqdm(range(len(df)), desc=desc):
        p = df.iloc[i]["path"]
        x = make_x6(p).unsqueeze(0).to(device)
        recon = ae(x)
        mse = ((recon - x) ** 2).mean().item()
        scores.append(mse)
    return np.array(scores, dtype=np.float64)


def topk_mean(x, k=10):
    x = np.sort(np.asarray(x))
    k = min(k, len(x))
    return float(x[-k:].mean())


def aggregate_per_src(df, col="anom"):
    agg = df.groupby(["label", "src_id"]).agg(
        mean=(col, "mean"),
        p95=(col, lambda x: np.percentile(x, 95)),
        top10=(col, lambda x: topk_mean(x, k=10)),
        max=(col, "max"),
        n=(col, "count"),
    ).reset_index()
    agg["y"] = (agg["label"] == "unhealthy").astype(int)
    return agg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = load_manifest(MANIFEST)
    srcs = sorted(df["src_id"].unique())

    ae = SmallConvAE(in_ch=6).to(device)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=device))
    ae.eval()

    fold_rows = []

    for holdout in srcs:
        train_df = df[df["src_id"] != holdout].reset_index(drop=True)
        test_df  = df[df["src_id"] == holdout].reset_index(drop=True)

        # calibrate on TRAIN HEALTHY only
        tr_h = train_df[train_df["label"] == "healthy"].reset_index(drop=True)
        if len(tr_h) == 0:
            continue

        tr_h_scores = score_df(tr_h, ae, device, desc=f"Calib healthy (holdout={holdout})")
        mu = float(tr_h_scores.mean())
        sd = float(tr_h_scores.std() + 1e-12)

        te_scores = score_df(test_df, ae, device, desc=f"Test tiles (holdout={holdout})")
        te_anom = np.abs((te_scores - mu) / sd)

        test_df = test_df.copy()
        test_df["mse"] = te_scores
        test_df["anom"] = te_anom

        # aggregate only THIS holdout image => one row
        agg = aggregate_per_src(test_df, col="anom")
        agg["holdout"] = holdout
        fold_rows.append(agg)

    all_agg = pd.concat(fold_rows, ignore_index=True)

    # AUROC across all holdout images, for each aggregation
    y = all_agg["y"].values

    results = {}
    for col in ["mean", "p95", "top10", "max"]:
        s = all_agg[col].values
        auc = roc_auc_score(y, s)
        results[col] = float(auc)
        print(f"LOSO AUROC ({col}) = {auc:.4f}")

    all_agg.to_csv(os.path.join(OUT_DIR, "loso_src_scores.csv"), index=False)
    print("Saved:", os.path.join(OUT_DIR, "loso_src_scores.csv"))
    print("Done.")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
