# analyze_eps_statistics.py
# Computes standalone effective-permittivity statistics for thesis tables/figures.
#
# Example:
#   python analyze_eps_statistics.py
#
# Outputs:
#   runs_eps_stats/eps_tile_stats.csv
#   runs_eps_stats/eps_source_stats.csv
#   runs_eps_stats/eps_class_summary.csv
#   runs_eps_stats/fig_eps_mean_boxplot.png
#   runs_eps_stats/fig_eps_gradient_boxplot.png

import os
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp

from data_manifest import load_manifest
from mmf import eps_eff_map_from_rgb, eps_feature_stack


DEFAULT_MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
DEFAULT_OUT_ROOT = "./runs_eps_stats"


def read_rgb01(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def label_to_y(label) -> int:
    return 0 if str(label).lower() == "healthy" else 1


def compute_tile_stats(path: str):
    rgb = read_rgb01(path)
    eps = eps_eff_map_from_rgb(rgb)
    mmf = eps_feature_stack(rgb)  # [eps01, variance, gradient]
    eps01 = mmf[..., 0]
    var = mmf[..., 1]
    grad = mmf[..., 2]

    return {
        "eps_mean": float(np.mean(eps)),
        "eps_std": float(np.std(eps)),
        "eps_min": float(np.min(eps)),
        "eps_max": float(np.max(eps)),
        "eps_p05": float(np.percentile(eps, 5)),
        "eps_p50": float(np.percentile(eps, 50)),
        "eps_p95": float(np.percentile(eps, 95)),
        "eps01_mean": float(np.mean(eps01)),
        "local_var_mean": float(np.mean(var)),
        "local_var_p95": float(np.percentile(var, 95)),
        "grad_mean": float(np.mean(grad)),
        "grad_p95": float(np.percentile(grad, 95)),
    }


def plot_box(df: pd.DataFrame, col: str, ylabel: str, title: str, out_path: str):
    healthy = df[df["y"] == 0][col].astype(float).to_numpy()
    unhealthy = df[df["y"] == 1][col].astype(float).to_numpy()

    plt.figure(figsize=(6, 5))
    plt.boxplot([healthy, unhealthy], labels=["Healthy", "Unhealthy"], showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    df = load_manifest(args.manifest)
    rows = []

    for _, row in df.iterrows():
        stats = compute_tile_stats(row["path"])
        stats.update({
            "label": str(row["label"]),
            "y": label_to_y(row["label"]),
            "src_id": str(row["src_id"]),
            "path": str(row["path"]),
        })
        rows.append(stats)

    tile_df = pd.DataFrame(rows)
    tile_df.to_csv(os.path.join(args.out_root, "eps_tile_stats.csv"), index=False)

    source_df = (
        tile_df.groupby(["y", "label", "src_id"])
        .agg(
            eps_mean=("eps_mean", "mean"),
            eps_std=("eps_std", "mean"),
            eps_p95=("eps_p95", "mean"),
            local_var_mean=("local_var_mean", "mean"),
            local_var_p95=("local_var_p95", "mean"),
            grad_mean=("grad_mean", "mean"),
            grad_p95=("grad_p95", "mean"),
            n_tiles=("eps_mean", "count"),
        )
        .reset_index()
    )
    source_df.to_csv(os.path.join(args.out_root, "eps_source_stats.csv"), index=False)

    # Class summary at source level.
    summary = (
        source_df.groupby(["y", "label"])
        .agg(
            n_sources=("src_id", "count"),
            eps_mean_mean=("eps_mean", "mean"),
            eps_mean_std=("eps_mean", "std"),
            eps_std_mean=("eps_std", "mean"),
            local_var_mean=("local_var_mean", "mean"),
            grad_mean=("grad_mean", "mean"),
        )
        .reset_index()
    )

    # Non-parametric tests between healthy and unhealthy sources.
    test_rows = []
    for col in ["eps_mean", "eps_std", "eps_p95", "local_var_mean", "local_var_p95", "grad_mean", "grad_p95"]:
        h = source_df[source_df["y"] == 0][col].astype(float).to_numpy()
        u = source_df[source_df["y"] == 1][col].astype(float).to_numpy()
        if len(h) > 0 and len(u) > 0:
            mw = mannwhitneyu(h, u, alternative="two-sided")
            ks = ks_2samp(h, u)
            test_rows.append({
                "feature": col,
                "healthy_mean": float(np.mean(h)),
                "unhealthy_mean": float(np.mean(u)),
                "healthy_std": float(np.std(h)),
                "unhealthy_std": float(np.std(u)),
                "mannwhitney_p": float(mw.pvalue),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            })

    tests = pd.DataFrame(test_rows)
    summary.to_csv(os.path.join(args.out_root, "eps_class_summary.csv"), index=False)
    tests.to_csv(os.path.join(args.out_root, "eps_statistical_tests.csv"), index=False)

    plot_box(
        source_df,
        "eps_mean",
        "Mean effective permittivity",
        "Source-level effective permittivity by class",
        os.path.join(args.out_root, "fig_eps_mean_boxplot.png"),
    )
    plot_box(
        source_df,
        "local_var_mean",
        "Mean local ε_eff variance",
        "Source-level ε_eff heterogeneity by class",
        os.path.join(args.out_root, "fig_eps_variance_boxplot.png"),
    )
    plot_box(
        source_df,
        "grad_mean",
        "Mean ε_eff gradient magnitude",
        "Source-level ε_eff gradient by class",
        os.path.join(args.out_root, "fig_eps_gradient_boxplot.png"),
    )

    print("Saved outputs to:", args.out_root)
    print("\nClass summary:")
    print(summary)
    print("\nStatistical tests:")
    print(tests)


if __name__ == "__main__":
    main()
