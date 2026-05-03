import os
import argparse

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

MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
OUT_ROOT = r"./runs_smallcnn"
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
        return torch.from_numpy(x).permute(2, 0, 1).float(), torch.tensor(y, dtype=torch.float32), row["src_id"]


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


@torch.no_grad()
def predict_dataset(model: nn.Module, loader: DataLoader, device: str):
    rows = []
    model.eval()
    for x, y, src_id in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        for i in range(len(prob)):
            rows.append({"y": int(y[i].item()), "prob": float(prob[i]), "src_id": str(src_id[i])})
    return pd.DataFrame(rows)


def aggregate_by_source(df_pred: pd.DataFrame):
    return (
        df_pred.groupby(["y", "src_id"])
        .agg(mean_prob=("prob", "mean"), p95_prob=("prob", lambda x: np.percentile(x, 95)), max_prob=("prob", "max"))
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rgb+eps", choices=["rgb", "mmf", "rgb+eps", "rgb+mmf"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_manifest(MANIFEST)
    tr_df, va_df = split_by_source(df, test_size=0.2, seed=args.seed)

    train_ds = TissueDataset(tr_df, args.mode)
    val_ds = TissueDataset(va_df, args.mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sample_x, _, _ = train_ds[0]
    model = SmallCNN(sample_x.shape[0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = os.path.join(OUT_ROOT, args.mode.replace("+", "_"))
    os.makedirs(run_dir, exist_ok=True)
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"SmallCNN {args.mode} ep{epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        pred_df = predict_dataset(model, val_loader, device)
        src_df = aggregate_by_source(pred_df)
        tile_auc = roc_auc_score(pred_df["y"], pred_df["prob"])
        src_auc = roc_auc_score(src_df["y"], src_df["mean_prob"])
        print(f"epoch={epoch} train_loss={tr_loss:.5f} tile_auc={tile_auc:.4f} src_auc={src_auc:.4f}")
        if src_auc > best_auc:
            best_auc = src_auc
            pred_df.to_csv(os.path.join(run_dir, "best_val_tile_predictions.csv"), index=False)
            src_df.to_csv(os.path.join(run_dir, "best_val_source_predictions.csv"), index=False)
            torch.save(model.state_dict(), os.path.join(run_dir, "best_smallcnn.pt"))

    print("Done. Best source AUROC:", best_auc)


if __name__ == "__main__":
    main()
