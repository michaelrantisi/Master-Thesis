import os
from dataclasses import dataclass
from typing import Tuple

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

# -------------------------------
# Thesis-aligned CNN classifier
# Main task: healthy vs unhealthy classification
# Main novelty: metamaterial-formalism-derived channels
# -------------------------------

MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
OUT_ROOT = r"./runs_classifier"
os.makedirs(OUT_ROOT, exist_ok=True)


@dataclass
class Config:
    mode: str = "rgb+mmf"  # one of: rgb, mmf, rgb+eps, rgb+mmf
    batch_size: int = 16
    epochs: int = 12
    lr: float = 5e-5
    weight_decay: float = 1e-4
    seed: int = 42


CFG = Config()


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

    def __len__(self) -> int:
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
    return agg


def main():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Mode:", CFG.mode)

    df = load_manifest(MANIFEST)
    tr_df, va_df = split_by_source(df, test_size=0.2, seed=CFG.seed)

    train_loader = DataLoader(TissueDataset(tr_df, CFG.mode), batch_size=CFG.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TissueDataset(va_df, CFG.mode), batch_size=CFG.batch_size, shuffle=False, num_workers=0)

    in_ch = num_input_channels(CFG.mode)
    model = ResNetClassifier(in_ch=in_ch).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    run_dir = os.path.join(OUT_ROOT, CFG.mode.replace("+", "_"))
    os.makedirs(run_dir, exist_ok=True)
    best_auc = -1.0

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"Train {epoch}"):
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
        tile_auc = roc_auc_score(pred_df["y"], pred_df["prob"])
        tile_acc = accuracy_score(pred_df["y"], (pred_df["prob"] >= 0.5).astype(int))
        tile_f1 = f1_score(pred_df["y"], (pred_df["prob"] >= 0.5).astype(int))

        src_df = aggregate_by_source(pred_df)
        src_auc = roc_auc_score(src_df["y"], src_df["mean_prob"])

        print(
            f"Epoch {epoch}: train_loss={train_loss:.5f} "
            f"tile_auc={tile_auc:.4f} tile_acc={tile_acc:.4f} tile_f1={tile_f1:.4f} "
            f"src_auc(mean)={src_auc:.4f}"
        )

        if src_auc > best_auc:
            best_auc = src_auc
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            pred_df.to_csv(os.path.join(run_dir, "best_val_tile_predictions.csv"), index=False)
            src_df.to_csv(os.path.join(run_dir, "best_val_source_predictions.csv"), index=False)
            print("Saved best model and validation predictions.")

    print("Done. Best source-level AUROC:", best_auc)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
