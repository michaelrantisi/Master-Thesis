import os
from dataclasses import dataclass

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
from train_resnet_classifier import ResNetClassifier, build_input, num_input_channels

MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
OUT_ROOT = r"./runs_classifier_loso"
os.makedirs(OUT_ROOT, exist_ok=True)


@dataclass
class Config:
    mode: str = "rgb+eps" # one of: rgb, mmf, rgb+eps, rgb+mmf
    batch_size: int = 16
    epochs: int = 8
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


class TissueDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = build_input(read_rgb01(row["path"]), self.mode)
        y = 0.0 if row["label"] == "healthy" else 1.0
        return torch.from_numpy(x).permute(2, 0, 1).float(), torch.tensor(y, dtype=torch.float32), row["src_id"]


@torch.no_grad()
def predict_df(model: nn.Module, loader: DataLoader, device: str):
    rows = []
    model.eval()
    for x, y, src_id in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = y.numpy().astype(int)
        for i in range(len(probs)):
            rows.append({"y": int(y[i]), "src_id": src_id[i], "prob": float(probs[i])})
    return pd.DataFrame(rows)


def aggregate_sources(df_pred: pd.DataFrame):
    return (
        df_pred.groupby(["y", "src_id"])
        .agg(mean_prob=("prob", "mean"), p95_prob=("prob", lambda x: np.percentile(x, 95)), max_prob=("prob", "max"))
        .reset_index()
    )


def main():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Mode:", CFG.mode)

    df = load_manifest(MANIFEST)
    srcs = sorted(df["src_id"].unique())
    all_rows = []

    for holdout in srcs:
        print(f"\n=== HOLDOUT SOURCE: {holdout} ===")
        tr_df = df[df["src_id"] != holdout].reset_index(drop=True)
        te_df = df[df["src_id"] == holdout].reset_index(drop=True)

        train_loader = DataLoader(TissueDataset(tr_df, CFG.mode), batch_size=CFG.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(TissueDataset(te_df, CFG.mode), batch_size=CFG.batch_size, shuffle=False, num_workers=0)

        model = ResNetClassifier(in_ch=num_input_channels(CFG.mode)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(1, CFG.epochs + 1):
            model.train()
            losses = []
            for x, y, _ in tqdm(train_loader, desc=f"Train {holdout} ep{epoch}"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(f"epoch={epoch} train_loss={np.mean(losses):.5f}")

        pred_df = predict_df(model, test_loader, device)
        src_df = aggregate_sources(pred_df)
        src_df["holdout"] = holdout
        all_rows.append(src_df)

    out = pd.concat(all_rows, ignore_index=True)
    aucs = {
        "mean_prob": roc_auc_score(out["y"], out["mean_prob"]),
        "p95_prob": roc_auc_score(out["y"], out["p95_prob"]),
        "max_prob": roc_auc_score(out["y"], out["max_prob"]),
    }
    print("\nLOSO source-level AUROC:")
    for k, v in aucs.items():
        print(k, f"= {v:.4f}")

    run_dir = os.path.join(OUT_ROOT, CFG.mode.replace("+", "_"))
    os.makedirs(run_dir, exist_ok=True)
    out.to_csv(os.path.join(run_dir, "loso_source_predictions.csv"), index=False)
    pd.DataFrame([aucs]).to_csv(os.path.join(run_dir, "loso_summary.csv"), index=False)
    print("Saved LOSO results.")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
