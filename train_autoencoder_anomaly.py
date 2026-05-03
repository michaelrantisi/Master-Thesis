# train_autoencoder_anomaly.py
# Autoencoder anomaly-detection experiment for the choroid thesis pipeline.
#
# Idea:
#   Train only on HEALTHY tiles.
#   Evaluate reconstruction error on healthy and unhealthy held-out sources.
#   Higher reconstruction error = more anomalous / more likely unhealthy.
#
# Outputs:
#   runs_autoencoder/<mode>_split_seed<seed>/...
#   runs_autoencoder/<mode>_loso_seed<seed>/...
#
# Example commands:
#   python train_autoencoder_anomaly.py --eval split --mode rgb+eps --epochs 40
#   python train_autoencoder_anomaly.py --eval loso  --mode rgb+eps --epochs 25
#   python train_autoencoder_anomaly.py --eval loso --modes rgb eps mmf rgb+eps rgb+mmf --epochs 25

import os
import argparse
import json
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_manifest import load_manifest, split_by_source
from mmf import eps_eff_map_from_rgb, eps_feature_stack
from models import SmallConvAE


DEFAULT_MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
DEFAULT_OUT_ROOT = r"./runs_autoencoder"


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rgb01(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def build_input(rgb01: np.ndarray, mode: str) -> np.ndarray:
    """
    Returns image-like tensor in H x W x C format, normalized to [0, 1].
    Supported modes:
      rgb       : 3-channel RGB image
      eps       : 1-channel normalized effective permittivity map
      mmf       : 3-channel [eps_norm, local variance, gradient magnitude]
      rgb+eps   : 4-channel RGB + eps
      rgb+mmf   : 6-channel RGB + MMF stack
    """
    mode = mode.lower()

    if mode == "rgb":
        return rgb01.astype(np.float32)

    eps = eps_eff_map_from_rgb(rgb01)
    eps01 = (eps - 1.0) / (20.0 - 1.0 + 1e-8)
    eps01 = np.clip(eps01, 0.0, 1.0).astype(np.float32)

    if mode == "eps":
        return eps01[..., None]

    if mode == "mmf":
        return eps_feature_stack(rgb01).astype(np.float32)

    if mode == "rgb+eps":
        return np.concatenate([rgb01, eps01[..., None]], axis=2).astype(np.float32)

    if mode == "rgb+mmf":
        return np.concatenate([rgb01, eps_feature_stack(rgb01)], axis=2).astype(np.float32)

    raise ValueError(f"Unknown mode: {mode}")


def label_to_y(label) -> int:
    return 0 if str(label).lower() == "healthy" else 1


class AETileDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rgb = read_rgb01(row["path"])
        x = build_input(rgb, self.mode)
        x = torch.from_numpy(x).permute(2, 0, 1).float()

        y = label_to_y(row["label"])
        src_id = str(row["src_id"])
        path = str(row["path"])
        return x, torch.tensor(y, dtype=torch.long), src_id, path


@torch.no_grad()
def predict_reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> pd.DataFrame:
    model.eval()
    rows = []

    # Mean squared reconstruction error per tile and per channel.
    # This becomes the anomaly score.
    for x, y, src_id, path in tqdm(loader, desc="Predict reconstruction error"):
        x = x.to(device, non_blocking=True)
        x_hat = model(x)
        err = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()

        for i in range(len(err)):
            rows.append({
                "y": int(y[i].item()),
                "label": "unhealthy" if int(y[i].item()) == 1 else "healthy",
                "src_id": str(src_id[i]),
                "path": str(path[i]),
                "mse": float(err[i]),
            })

    return pd.DataFrame(rows)


def aggregate_by_source(tile_df: pd.DataFrame) -> pd.DataFrame:
    src_df = (
        tile_df.groupby(["y", "label", "src_id"])
        .agg(
            mean_mse=("mse", "mean"),
            median_mse=("mse", "median"),
            p95_mse=("mse", lambda x: np.percentile(x, 95)),
            max_mse=("mse", "max"),
            n_tiles=("mse", "count"),
        )
        .reset_index()
    )
    return src_df


def safe_auc(y, score) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def threshold_metrics_from_healthy(tile_or_source_df: pd.DataFrame, score_col: str, z: float = 3.0) -> Dict[str, float]:
    """
    Choose anomaly threshold from healthy score distribution:
        threshold = healthy_mean + z * healthy_std
    Then classify score >= threshold as unhealthy.
    """
    healthy = tile_or_source_df[tile_or_source_df["y"] == 0][score_col].astype(float).to_numpy()
    if len(healthy) == 0:
        return {"threshold": float("nan"), "acc": float("nan"), "f1": float("nan")}

    mu = float(np.mean(healthy))
    sigma = float(np.std(healthy) + 1e-12)
    threshold = mu + z * sigma

    y_true = tile_or_source_df["y"].astype(int).to_numpy()
    y_pred = (tile_or_source_df[score_col].astype(float).to_numpy() >= threshold).astype(int)

    return {
        "threshold": threshold,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_autoencoder(
    train_df: pd.DataFrame,
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> nn.Module:
    train_ds = AETileDataset(train_df, mode)
    if len(train_ds) == 0:
        raise RuntimeError("No training tiles. Check manifest and healthy labels.")

    sample_x, _, _, _ = train_ds[0]
    in_ch = int(sample_x.shape[0])

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallConvAE(in_ch=in_ch).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for x, _, _, _ in tqdm(loader, desc=f"AE train {mode} ep{epoch}"):
            x = x.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        print(f"epoch={epoch:03d} train_mse={np.mean(losses):.8f}")

    return model


def evaluate_and_save(
    model: nn.Module,
    eval_df: pd.DataFrame,
    mode: str,
    batch_size: int,
    device: str,
    run_dir: str,
    meta: Dict,
) -> Dict[str, float]:
    eval_ds = AETileDataset(eval_df, mode)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    tile_df = predict_reconstruction_errors(model, eval_loader, device)
    source_df = aggregate_by_source(tile_df)

    tile_path = os.path.join(run_dir, "ae_tile_scores.csv")
    source_path = os.path.join(run_dir, "ae_source_scores.csv")
    tile_df.to_csv(tile_path, index=False)
    source_df.to_csv(source_path, index=False)

    summary = {
        **meta,
        "mode": mode,
        "n_eval_tiles": int(len(tile_df)),
        "n_eval_sources": int(len(source_df)),
        "tile_auc_mse": safe_auc(tile_df["y"], tile_df["mse"]),
        "src_auc_mean_mse": safe_auc(source_df["y"], source_df["mean_mse"]),
        "src_auc_median_mse": safe_auc(source_df["y"], source_df["median_mse"]),
        "src_auc_p95_mse": safe_auc(source_df["y"], source_df["p95_mse"]),
        "src_auc_max_mse": safe_auc(source_df["y"], source_df["max_mse"]),
    }

    for score_col in ["mse"]:
        tm = threshold_metrics_from_healthy(tile_df, score_col, z=3.0)
        summary[f"tile_threshold_{score_col}"] = tm["threshold"]
        summary[f"tile_acc_{score_col}"] = tm["acc"]
        summary[f"tile_f1_{score_col}"] = tm["f1"]

    for score_col in ["mean_mse", "median_mse", "p95_mse", "max_mse"]:
        sm = threshold_metrics_from_healthy(source_df, score_col, z=3.0)
        summary[f"src_threshold_{score_col}"] = sm["threshold"]
        summary[f"src_acc_{score_col}"] = sm["acc"]
        summary[f"src_f1_{score_col}"] = sm["f1"]

    with open(os.path.join(run_dir, "ae_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "ae_summary.csv"), index=False)

    print("\nSaved:")
    print(" ", tile_path)
    print(" ", source_path)
    print(" ", os.path.join(run_dir, "ae_summary.csv"))
    print("\nSummary:")
    print(pd.DataFrame([summary]).T)

    return summary


def run_split(args, mode: str, df: pd.DataFrame, device: str) -> Dict[str, float]:
    """
    Source-aware train/validation experiment.
    Train autoencoder only on healthy tiles from training sources.
    Evaluate on validation sources containing both healthy and unhealthy.
    """
    tr_df, va_df = split_by_source(df, test_size=args.test_size, seed=args.seed)
    ae_train_df = tr_df[tr_df["label"].astype(str).str.lower() == "healthy"].reset_index(drop=True)

    run_dir = os.path.join(args.out_root, f"{mode.replace('+', '_')}_split_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n=== AE split experiment | mode={mode} ===")
    print(f"Healthy training tiles: {len(ae_train_df)}")
    print(f"Validation tiles: {len(va_df)}")
    print(f"Output: {run_dir}")

    model = train_autoencoder(
        ae_train_df,
        mode=mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    torch.save(model.state_dict(), os.path.join(run_dir, "ae_model.pt"))

    return evaluate_and_save(
        model,
        va_df,
        mode=mode,
        batch_size=args.batch_size,
        device=device,
        run_dir=run_dir,
        meta={
            "eval": "split",
            "seed": args.seed,
            "epochs": args.epochs,
            "train_healthy_tiles": int(len(ae_train_df)),
            "eval_tiles": int(len(va_df)),
        },
    )


def run_loso(args, mode: str, df: pd.DataFrame, device: str) -> Dict[str, float]:
    """
    Leave-One-Source-Out anomaly detection.
    For each held-out source:
      train only on healthy tiles from all other sources,
      score all tiles from the held-out source.
    """
    run_dir = os.path.join(args.out_root, f"{mode.replace('+', '_')}_loso_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    srcs = sorted(df["src_id"].astype(str).unique())
    all_tile_rows: List[pd.DataFrame] = []
    all_source_rows: List[pd.DataFrame] = []
    fold_summaries = []

    print(f"\n=== AE LOSO experiment | mode={mode} | folds={len(srcs)} ===")
    print(f"Output: {run_dir}")

    for holdout in srcs:
        print(f"\n--- HOLDOUT SOURCE: {holdout} ---")

        train_df = df[df["src_id"].astype(str) != holdout].reset_index(drop=True)
        test_df = df[df["src_id"].astype(str) == holdout].reset_index(drop=True)

        ae_train_df = train_df[train_df["label"].astype(str).str.lower() == "healthy"].reset_index(drop=True)
        print(f"Healthy training tiles: {len(ae_train_df)} | test tiles: {len(test_df)}")

        model = train_autoencoder(
            ae_train_df,
            mode=mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )

        test_ds = AETileDataset(test_df, mode)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        tile_df = predict_reconstruction_errors(model, test_loader, device)
        tile_df["holdout"] = holdout
        src_df = aggregate_by_source(tile_df)
        src_df["holdout"] = holdout

        all_tile_rows.append(tile_df)
        all_source_rows.append(src_df)

        fold_summaries.append({
            "holdout": holdout,
            "label": src_df["label"].iloc[0],
            "y": int(src_df["y"].iloc[0]),
            "mean_mse": float(src_df["mean_mse"].iloc[0]),
            "p95_mse": float(src_df["p95_mse"].iloc[0]),
            "max_mse": float(src_df["max_mse"].iloc[0]),
            "n_tiles": int(src_df["n_tiles"].iloc[0]),
        })

    tile_df_all = pd.concat(all_tile_rows, ignore_index=True)
    source_df_all = pd.concat(all_source_rows, ignore_index=True)
    fold_summary_df = pd.DataFrame(fold_summaries)

    tile_df_all.to_csv(os.path.join(run_dir, "ae_loso_tile_scores.csv"), index=False)
    source_df_all.to_csv(os.path.join(run_dir, "ae_loso_source_scores.csv"), index=False)
    fold_summary_df.to_csv(os.path.join(run_dir, "ae_loso_fold_summary.csv"), index=False)

    summary = {
        "eval": "loso",
        "seed": args.seed,
        "epochs": args.epochs,
        "mode": mode,
        "n_eval_tiles": int(len(tile_df_all)),
        "n_eval_sources": int(len(source_df_all)),
        "tile_auc_mse": safe_auc(tile_df_all["y"], tile_df_all["mse"]),
        "src_auc_mean_mse": safe_auc(source_df_all["y"], source_df_all["mean_mse"]),
        "src_auc_median_mse": safe_auc(source_df_all["y"], source_df_all["median_mse"]),
        "src_auc_p95_mse": safe_auc(source_df_all["y"], source_df_all["p95_mse"]),
        "src_auc_max_mse": safe_auc(source_df_all["y"], source_df_all["max_mse"]),
    }

    for score_col in ["mse"]:
        tm = threshold_metrics_from_healthy(tile_df_all, score_col, z=3.0)
        summary[f"tile_threshold_{score_col}"] = tm["threshold"]
        summary[f"tile_acc_{score_col}"] = tm["acc"]
        summary[f"tile_f1_{score_col}"] = tm["f1"]

    for score_col in ["mean_mse", "median_mse", "p95_mse", "max_mse"]:
        sm = threshold_metrics_from_healthy(source_df_all, score_col, z=3.0)
        summary[f"src_threshold_{score_col}"] = sm["threshold"]
        summary[f"src_acc_{score_col}"] = sm["acc"]
        summary[f"src_f1_{score_col}"] = sm["f1"]

    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "ae_loso_summary.csv"), index=False)
    with open(os.path.join(run_dir, "ae_loso_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nLOSO summary:")
    print(pd.DataFrame([summary]).T)
    print("\nSaved LOSO outputs to:", run_dir)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--eval", choices=["split", "loso"], default="split")
    parser.add_argument("--mode", default=None, choices=["rgb", "eps", "mmf", "rgb+eps", "rgb+mmf"])
    parser.add_argument("--modes", nargs="+", default=None, choices=["rgb", "eps", "mmf", "rgb+eps", "rgb+mmf"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    os.makedirs(args.out_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Manifest:", args.manifest)

    df = load_manifest(args.manifest)
    df["src_id"] = df["src_id"].astype(str)

    if args.modes is not None:
        modes = args.modes
    elif args.mode is not None:
        modes = [args.mode]
    else:
        modes = ["rgb+eps"]

    summaries = []
    for mode in modes:
        if args.eval == "split":
            summaries.append(run_split(args, mode, df, device))
        else:
            summaries.append(run_loso(args, mode, df, device))

    combined = pd.DataFrame(summaries)
    combined_path = os.path.join(args.out_root, f"ae_{args.eval}_combined_summary_seed{args.seed}.csv")
    combined.to_csv(combined_path, index=False)
    print("\nCombined summary saved:", combined_path)
    print(combined)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
