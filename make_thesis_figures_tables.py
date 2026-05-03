# make_thesis_figures_tables.py
# Creates thesis-ready tables and figures from CNN, LOSO, seed-sweep, and autoencoder CSV outputs.
#
# Example:
#   python make_thesis_figures_tables.py
#
# Optional:
#   python make_thesis_figures_tables.py --autoencoder-root ./runs_autoencoder --loso-root ./runs_loso_robust --seed-root ./runs_seed_sweep

import os
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_table(df: pd.DataFrame, out_csv: str, out_docx_like_txt: str = None) -> None:
    df.to_csv(out_csv, index=False)
    if out_docx_like_txt:
        with open(out_docx_like_txt, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))


def plot_roc_from_scores(df: pd.DataFrame, score_col: str, title: str, out_path: str) -> None:
    y = df["y"].astype(int).to_numpy()
    s = df[score_col].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_score_boxplot(df: pd.DataFrame, score_col: str, title: str, ylabel: str, out_path: str) -> None:
    healthy = df[df["y"].astype(int) == 0][score_col].astype(float).to_numpy()
    unhealthy = df[df["y"].astype(int) == 1][score_col].astype(float).to_numpy()

    plt.figure(figsize=(6, 5))
    plt.boxplot([healthy, unhealthy], labels=["Healthy", "Unhealthy"], showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_source_bar(df: pd.DataFrame, score_col: str, title: str, ylabel: str, out_path: str) -> None:
    tmp = df.copy()
    tmp["name"] = tmp["src_id"].astype(str) + "\n(" + tmp["label"].astype(str) + ")"
    tmp = tmp.sort_values(["y", score_col])

    plt.figure(figsize=(max(7, 0.45 * len(tmp)), 5))
    plt.bar(np.arange(len(tmp)), tmp[score_col].astype(float).to_numpy())
    plt.xticks(np.arange(len(tmp)), tmp["name"], rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def collect_autoencoder(autoencoder_root: str, out_dir: str) -> None:
    rows = []

    # Read all split and LOSO summary files.
    for p in glob.glob(os.path.join(autoencoder_root, "*", "ae_summary.csv")):
        df = pd.read_csv(p)
        df["source_file"] = p
        rows.append(df)

    for p in glob.glob(os.path.join(autoencoder_root, "*", "ae_loso_summary.csv")):
        df = pd.read_csv(p)
        df["source_file"] = p
        rows.append(df)

    if not rows:
        print("No autoencoder summary files found.")
        return

    summary = pd.concat(rows, ignore_index=True)

    keep_cols = [
        "eval", "mode", "epochs", "n_eval_tiles", "n_eval_sources",
        "tile_auc_mse", "src_auc_mean_mse", "src_auc_median_mse",
        "src_auc_p95_mse", "src_auc_max_mse",
        "src_acc_mean_mse", "src_f1_mean_mse",
    ]
    keep_cols = [c for c in keep_cols if c in summary.columns]
    table = summary[keep_cols].sort_values(["eval", "mode"])

    save_table(
        table,
        os.path.join(out_dir, "table_autoencoder_results.csv"),
        os.path.join(out_dir, "table_autoencoder_results.txt"),
    )

    # Make one set of plots per AE run folder.
    for source_path in summary["source_file"]:
        run_dir = os.path.dirname(source_path)
        run_name = os.path.basename(run_dir)
        fig_prefix = os.path.join(out_dir, f"ae_{run_name}")

        if os.path.exists(os.path.join(run_dir, "ae_source_scores.csv")):
            src_df = pd.read_csv(os.path.join(run_dir, "ae_source_scores.csv"))
        elif os.path.exists(os.path.join(run_dir, "ae_loso_source_scores.csv")):
            src_df = pd.read_csv(os.path.join(run_dir, "ae_loso_source_scores.csv"))
        else:
            continue

        if len(src_df["y"].unique()) >= 2:
            plot_roc_from_scores(
                src_df,
                "mean_mse",
                f"Autoencoder source-level ROC: {run_name}",
                fig_prefix + "_source_roc.png",
            )

        plot_score_boxplot(
            src_df,
            "mean_mse",
            f"Autoencoder source-level anomaly score: {run_name}",
            "Mean reconstruction MSE",
            fig_prefix + "_source_boxplot.png",
        )

        plot_source_bar(
            src_df,
            "mean_mse",
            f"Autoencoder source-level anomaly scores: {run_name}",
            "Mean reconstruction MSE",
            fig_prefix + "_source_bar.png",
        )

    print("Autoencoder tables/figures saved to:", out_dir)


def collect_loso(loso_root: str, out_dir: str) -> None:
    rows = []
    for p in glob.glob(os.path.join(loso_root, "*", "loso_auc_ci.csv")):
        df = pd.read_csv(p)
        mode = os.path.basename(os.path.dirname(p)).replace("_", "+")
        df["mode"] = mode
        rows.append(df)

    if rows:
        ci = pd.concat(rows, ignore_index=True)
        cols = ["mode", "score_col", "auc", "ci_95_lo", "ci_95_hi", "n_boot_valid"]
        cols = [c for c in cols if c in ci.columns]
        save_table(
            ci[cols].sort_values(["mode", "score_col"]),
            os.path.join(out_dir, "table_loso_auc_ci.csv"),
            os.path.join(out_dir, "table_loso_auc_ci.txt"),
        )

    for p in glob.glob(os.path.join(loso_root, "*", "loso_per_source_predictions.csv")):
        mode = os.path.basename(os.path.dirname(p)).replace("_", "+")
        src_df = pd.read_csv(p)
        if "label" not in src_df.columns:
            src_df["label"] = np.where(src_df["y"].astype(int) == 1, "unhealthy", "healthy")

        if len(src_df["y"].unique()) >= 2:
            plot_roc_from_scores(
                src_df,
                "mean_prob",
                f"LOSO source-level ROC: {mode}",
                os.path.join(out_dir, f"cnn_loso_{mode.replace('+', '_')}_source_roc.png"),
            )

        plot_source_bar(
            src_df,
            "mean_prob",
            f"LOSO source-level CNN scores: {mode}",
            "Mean predicted probability of unhealthy",
            os.path.join(out_dir, f"cnn_loso_{mode.replace('+', '_')}_source_bar.png"),
        )

    print("LOSO CNN tables/figures saved to:", out_dir)


def collect_seed_sweep(seed_root: str, out_dir: str) -> None:
    p = os.path.join(seed_root, "seed_sweep_results.csv")
    if not os.path.exists(p):
        print("No seed_sweep_results.csv found.")
        return

    df = pd.read_csv(p)
    metric_cols = [c for c in df.columns if c.startswith("src_auc") or c in ["tile_auc", "tile_acc", "tile_f1"]]

    summary_rows = []
    for mode, g in df.groupby("mode"):
        row = {"mode": mode, "n_runs": len(g)}
        for c in metric_cols:
            row[c + "_mean"] = float(g[c].mean())
            row[c + "_std"] = float(g[c].std())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    save_table(
        summary,
        os.path.join(out_dir, "table_seed_sweep_summary.csv"),
        os.path.join(out_dir, "table_seed_sweep_summary.txt"),
    )

    # Simple bar plot for source-level AUROC mean by mode.
    if "src_auc_mean" in df.columns:
        modes = []
        means = []
        stds = []
        for mode, g in df.groupby("mode"):
            modes.append(mode)
            means.append(g["src_auc_mean"].mean())
            stds.append(g["src_auc_mean"].std())

        plt.figure(figsize=(6, 5))
        plt.bar(np.arange(len(modes)), means, yerr=stds, capsize=5)
        plt.xticks(np.arange(len(modes)), modes, rotation=30)
        plt.ylabel("Source-level AUROC, mean ± SD")
        plt.title("Seed-sweep robustness of source-level AUROC")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "seed_sweep_src_auc_bar.png"), dpi=300)
        plt.close()

    print("Seed-sweep table/figure saved to:", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder-root", default="./runs_autoencoder")
    parser.add_argument("--loso-root", default="./runs_loso_robust")
    parser.add_argument("--seed-root", default="./runs_seed_sweep")
    parser.add_argument("--out-dir", default="./thesis_outputs")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    collect_autoencoder(args.autoencoder_root, args.out_dir)
    collect_loso(args.loso_root, args.out_dir)
    collect_seed_sweep(args.seed_root, args.out_dir)

    print("\nDone. Check:", args.out_dir)


if __name__ == "__main__":
    main()
