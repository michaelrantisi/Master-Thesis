import os
import json
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_manifest import load_manifest, split_by_source
from mmf import eps_eff_map_from_rgb, eps_feature_stack
from models import ResNet50_6ch

MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
OUT_ROOT = r"./runs_seed_sweep"
os.makedirs(OUT_ROOT, exist_ok=True)


def read_rgb01(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def build_input(rgb01: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
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
    raise ValueError(f"Unknown mode: {mode}")


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
            new_conv = nn.Conv2d(
                in_ch,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )
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
            self.head = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 1),
            )
            self.model = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        rgb = read_rgb01(row["path"])
        x = build_input(rgb, self.mode)
        y = 0.0 if row["label"] == "healthy" else 1.0
        return (
            torch.from_numpy(x).permute(2, 0, 1).float(),
            torch.tensor(y, dtype=torch.float32),
            row["src_id"],
        )


@torch.no_grad()
def predict_dataset(model: nn.Module, loader: DataLoader, device: str):
    logits_all, y_all, src_all = [], [], []
    model.eval()
    for x, y, src_id in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        logits_all.append(logits)
        y_all.append(y.numpy())
        src_all.extend(list(src_id))
    logits = np.concatenate(logits_all)
    y = np.concatenate(y_all).astype(int)
    prob = 1.0 / (1.0 + np.exp(-logits))
    return pd.DataFrame({"y": y, "prob": prob, "src_id": src_all})


def aggregate_by_source(df_pred: pd.DataFrame):
    return (
        df_pred.groupby(["y", "src_id"])
        .agg(
            mean_prob=("prob", "mean"),
            p95_prob=("prob", lambda x: np.percentile(x, 95)),
            max_prob=("prob", "max"),
            n_tiles=("prob", "count"),
        )
        .reset_index()
    )


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def run_one(mode: str, seed: int, epochs: int, batch_size: int, lr: float, weight_decay: float, device: str):
    set_all_seeds(seed)
    df = load_manifest(MANIFEST)
    tr_df, va_df = split_by_source(df, test_size=0.2, seed=seed)

    train_loader = DataLoader(TissueDataset(tr_df, mode), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TissueDataset(va_df, mode), batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNetClassifier(in_ch=num_input_channels(mode)).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"{mode} seed={seed} ep={epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        pred_df = predict_dataset(model, val_loader, device)
        src_df = aggregate_by_source(pred_df)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "tile_auc": float(roc_auc_score(pred_df["y"], pred_df["prob"])),
            "tile_acc": float(accuracy_score(pred_df["y"], (pred_df["prob"] >= 0.5).astype(int))),
            "tile_f1": float(f1_score(pred_df["y"], (pred_df["prob"] >= 0.5).astype(int))),
            "src_auc_mean": float(roc_auc_score(src_df["y"], src_df["mean_prob"])),
            "src_auc_p95": float(roc_auc_score(src_df["y"], src_df["p95_prob"])),
            "src_auc_max": float(roc_auc_score(src_df["y"], src_df["max_prob"])),
        }
        if best is None or row["src_auc_mean"] > best["src_auc_mean"]:
            best = row
    best.update({"mode": mode, "seed": seed})
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["rgb", "rgb+eps"], choices=["rgb", "mmf", "rgb+eps", "rgb+mmf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for mode in args.modes:
        for seed in args.seeds:
            rows.append(run_one(mode, seed, args.epochs, args.batch_size, args.lr, args.weight_decay, device))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_ROOT, "seed_sweep_results.csv"), index=False)
    summary = df.groupby("mode").agg(["mean", "std"])
    summary.to_csv(os.path.join(OUT_ROOT, "seed_sweep_summary.csv"))
    print(df)
    print("\nSaved:", os.path.join(OUT_ROOT, "seed_sweep_results.csv"))
    print("Saved:", os.path.join(OUT_ROOT, "seed_sweep_summary.csv"))


if __name__ == "__main__":
    main()
