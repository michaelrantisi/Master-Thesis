import os
import re
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_manifest import load_manifest
from mmf import eps_eff_map_from_rgb, eps_feature_stack
from models import ResNet50_6ch

MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
OUT_ROOT = r"./runs_loso_robust"
os.makedirs(OUT_ROOT, exist_ok=True)


def read_rgb01(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def build_input(rgb01: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rgb":
        return rgb01.astype(np.float32)
    if mode == "mmf":
        return eps_feature_stack(rgb01).astype(np.float32)
    if mode == "rgb+eps":
        eps = eps_eff_map_from_rgb(rgb01)
        eps01 = (eps - 1.0) / (20.0 - 1.0 + 1e-8)
        eps01 = np.clip(eps01, 0.0, 1.0).astype(np.float32)
        return np.concatenate([rgb01, eps01[..., None]], axis=2)
    if mode == "rgb+mmf":
        return np.concatenate([rgb01, eps_feature_stack(rgb01)], axis=2).astype(np.float32)
    raise ValueError(mode)


def num_input_channels(mode: str) -> int:
    return build_input(np.zeros((8, 8, 3), dtype=np.float32), mode).shape[2]


class ResNetClassifier(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        if in_ch == 6:
            self.model = ResNet50_6ch(in_ch=6)
            self.backbone = None
            self.head = None
        else:
            import torchvision.models as tvm
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
            conv1 = m.conv1
            new_conv = nn.Conv2d(in_ch, conv1.out_channels, kernel_size=conv1.kernel_size,
                                 stride=conv1.stride, padding=conv1.padding, bias=False)
            with torch.no_grad():
                w = conv1.weight
                if in_ch <= 3:
                    new_conv.weight[:, :in_ch] = w[:, :in_ch]
                else:
                    new_conv.weight[:, :3] = w
                    mean_w = w.mean(dim=1, keepdim=True)
                    for c in range(3, in_ch):
                        new_conv.weight[:, c:c+1] = mean_w
            m.conv1 = new_conv
            m.fc = nn.Identity()
            self.backbone = m
            self.head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 1))
            self.model = None

    def forward(self, x):
        if self.model is not None:
            return self.model(x)
        feat = self.backbone(x)
        return self.head(feat).squeeze(1)


class TissueDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = build_input(read_rgb01(row["path"]), self.mode)
        y = 0.0 if row["label"] == "healthy" else 1.0
        return torch.from_numpy(x).permute(2, 0, 1).float(), torch.tensor(y, dtype=torch.float32), row["src_id"], row["path"]


@torch.no_grad()
def predict_dataset(model: nn.Module, loader: DataLoader, device: str):
    rows = []
    model.eval()
    for x, y, src_id, path in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        for i in range(len(prob)):
            rows.append({
                "y": int(y[i].item()),
                "prob": float(prob[i]),
                "src_id": str(src_id[i]),
                "path": str(path[i]),
            })
    return pd.DataFrame(rows)


def aggregate_by_source(df_pred: pd.DataFrame):
    agg = (
        df_pred.groupby(["y", "src_id"])
        .agg(
            mean_prob=("prob", "mean"),
            p95_prob=("prob", lambda x: np.percentile(x, 95)),
            max_prob=("prob", "max"),
            n_tiles=("prob", "count"),
        )
        .reset_index()
    )
    agg["label"] = np.where(agg["y"] == 1, "unhealthy", "healthy")
    return agg


def bootstrap_auc(y, s, n_boot=5000, seed=42):
    y = np.asarray(y)
    s = np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    idx_all = np.arange(len(y))
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        yb = y[idx]
        sb = s[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(roc_auc_score(yb, sb))
    vals = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "lo": float(np.percentile(vals, 2.5)),
        "hi": float(np.percentile(vals, 97.5)),
        "n_boot_valid": int(len(vals)),
    }


def infer_acquisition_group(src_id: str) -> str:
    sid = src_id.lower()
    magn = "20x" if "20x" in sid else ("4x" if "4x" in sid else "unknownmag")
    modality = "bf" if "bf" in sid else ("ph1" if "ph1" in sid else "unknownmod")
    return f"{magn}_{modality}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rgb+eps", choices=["rgb", "mmf", "rgb+eps", "rgb+mmf"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_manifest(MANIFEST)
    srcs = sorted(df["src_id"].unique())
    run_dir = os.path.join(OUT_ROOT, args.mode.replace("+", "_"))
    os.makedirs(run_dir, exist_ok=True)

    source_rows = []
    tile_rows = []
    for holdout in srcs:
        train_df = df[df["src_id"] != holdout].reset_index(drop=True)
        test_df = df[df["src_id"] == holdout].reset_index(drop=True)

        train_loader = DataLoader(TissueDataset(train_df, args.mode), batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(TissueDataset(test_df, args.mode), batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = ResNetClassifier(in_ch=num_input_channels(args.mode)).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr = 0.0
            for x, y, _, _ in tqdm(train_loader, desc=f"{args.mode} holdout={holdout} ep{epoch}"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                tr += loss.item()

        pred_df = predict_dataset(model, test_loader, device)
        pred_df["holdout"] = holdout
        tile_rows.append(pred_df)

        src_df = aggregate_by_source(pred_df)
        assert len(src_df) == 1, "LOSO test fold should contain exactly one source row"
        row = src_df.iloc[0].to_dict()
        row["holdout"] = holdout
        row["acquisition_group"] = infer_acquisition_group(holdout)
        source_rows.append(row)

    source_df = pd.DataFrame(source_rows)
    tile_df = pd.concat(tile_rows, ignore_index=True)

    source_df.to_csv(os.path.join(run_dir, "loso_per_source_predictions.csv"), index=False)
    tile_df.to_csv(os.path.join(run_dir, "loso_per_tile_predictions.csv"), index=False)

    ci_rows = []
    y = source_df["y"].astype(int).to_numpy()
    for col in ["mean_prob", "p95_prob", "max_prob"]:
        auc = float(roc_auc_score(y, source_df[col].astype(float).to_numpy()))
        ci = bootstrap_auc(y, source_df[col].astype(float).to_numpy())
        ci_rows.append({
            "score_col": col,
            "auc": auc,
            "bootstrap_mean": ci["mean"],
            "ci_95_lo": ci["lo"],
            "ci_95_hi": ci["hi"],
            "n_boot_valid": ci["n_boot_valid"],
        })
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(run_dir, "loso_auc_bootstrap_ci.csv"), index=False)

    print("\nPer-source table saved to:", os.path.join(run_dir, "loso_per_source_predictions.csv"))
    print("Bootstrap CI table saved to:", os.path.join(run_dir, "loso_auc_bootstrap_ci.csv"))
    print(ci_df)


if __name__ == "__main__":
    main()
