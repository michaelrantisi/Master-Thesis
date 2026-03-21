import os
import re
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from tiling import list_images
from models import SmallConvAE
from mmf import eps_eff_map_from_rgb

HEALTHY_DIR = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\healthy"
UNHEALTHY_DIR = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\unhealthy"

AE_WEIGHTS = r".\runs_ae\ae_best.pt"
OUT_DIR = r".\runs_eval_image"
os.makedirs(OUT_DIR, exist_ok=True)
EPS_CLIP = (1.0, 20.0)
EPS_HOST = 2.0
EPS_CHANNELS = (3.0, 4.0, 5.0)
N_AUG = 6

def read_rgb01(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)


def tile_group_id_from_filename(path: str, n_aug: int = 6) -> int:
    base = os.path.basename(path)
    m = re.search(r"_(\d+)\.", base)
    if not m:
        return hash(base)
    k = int(m.group(1))
    return k // n_aug


class EvalTileDataset(torch.utils.data.Dataset):
    def __init__(self, paths, label: int):
        self.paths = list(paths)
        self.label = int(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        rgb = read_rgb01(p)

        eps = eps_eff_map_from_rgb(
            rgb,
            eps_host=EPS_HOST,
            eps_channels=EPS_CHANNELS,
            clip=EPS_CLIP,
        )
        eps01 = (eps - EPS_CLIP[0]) / (EPS_CLIP[1] - EPS_CLIP[0] + 1e-8)
        eps01 = np.clip(eps01, 0.0, 1.0).astype(np.float32)

        x4 = np.concatenate([rgb, eps01[..., None]], axis=2)   
        x4 = torch.from_numpy(x4).permute(2, 0, 1).float()     

        # group id for image-level aggregation
        gid = tile_group_id_from_filename(p, n_aug=N_AUG)

        y = self.label
        return x4, y, gid, p


def aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates tile scores per group id.
    Returns one row per group with mean/p95/max and label.
    """
    agg = df.groupby(["label", "gid"]).agg(
        mean_score=("score", "mean"),
        p95_score=("score", lambda x: np.percentile(x, 95)),
        max_score=("score", "max"),
        n_tiles=("score", "count"),
    ).reset_index()
    return agg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not os.path.exists(AE_WEIGHTS):
        raise RuntimeError(f"AE weights not found: {AE_WEIGHTS}")

    h = list_images(HEALTHY_DIR, exts=(".jpg", ".jpeg", ".png"))
    u = list_images(UNHEALTHY_DIR, exts=(".jpg", ".jpeg", ".png"))

    if len(h) == 0:
        raise RuntimeError(f"No images found in: {HEALTHY_DIR}")
    if len(u) == 0:
        raise RuntimeError(f"No images found in: {UNHEALTHY_DIR}")

    ds = torch.utils.data.ConcatDataset([
        EvalTileDataset(h, 0),
        EvalTileDataset(u, 1),
    ])

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    ae = SmallConvAE(in_ch=4).to(device)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=device))
    ae.eval()

    rows = []

    with torch.no_grad():
        for x4, y, gid, path in tqdm(loader, desc="Eval AE (tiles)"):
            x4 = x4.to(device, non_blocking=True)
            recon = ae(x4)
            mse = ((recon - x4) ** 2).mean(dim=(1, 2, 3)).detach().cpu().numpy()
            if torch.is_tensor(y):
                y = y.cpu().numpy()
            if torch.is_tensor(gid):
                gid = gid.cpu().numpy()

            for i in range(len(mse)):
                rows.append({
                    "label": int(y[i]),
                    "gid": int(gid[i]),
                    "score": float(mse[i]),
                    "path": str(path[i]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "tile_scores.csv"), index=False)
    print("Saved tile_scores.csv")

    agg = aggregate_scores(df)
    agg.to_csv(os.path.join(OUT_DIR, "image_scores.csv"), index=False)
    print("Saved image_scores.csv")
    print("Aggregated groups (healthy/unhealthy):", agg.groupby("label").size().to_dict())

    # Evaluate AUROC at group level using different aggregations
    labels = agg["label"].values

    for col in ["mean_score", "p95_score", "max_score"]:
        scores = agg[col].values
        auc = roc_auc_score(labels, scores)
        s_h = scores[labels == 0]
        s_u = scores[labels == 1]
        ks = ks_2samp(s_h, s_u)
        print(f"\n=== IMAGE-LEVEL using {col} ===")
        print("AUROC:", float(auc))
        print("KS D:", float(ks.statistic), "p:", float(ks.pvalue))
        print("Healthy mean/std:", float(s_h.mean()), float(s_h.std()))
        print("Unhealthy mean/std:", float(s_u.mean()), float(s_u.std()))

        # ROC curve plot for this aggregation
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"roc_{col}.png"), dpi=200)
        plt.close()

    print("\nSaved ROC plots in:", OUT_DIR)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
