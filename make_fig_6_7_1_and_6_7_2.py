# make_fig_6_7_1_and_6_7_2.py
# Generates:
#  - Figure_6_7_1_MIL_Aggregation_AUROC.png  (from runs_mil/val_src_aggregates_calibrated.csv)
#  - Figure_6_7_2_LOSO_AUROC_Comparison.png  (from runs_loso/loso_src_scores.csv)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = r"C:\Users\User\Desktop\thesis_choroid_pipeline"

MIL_CSV  = os.path.join(ROOT, "runs_mil",  "val_src_aggregates_calibrated.csv")
LOSO_CSV = os.path.join(ROOT, "runs_loso", "loso_src_scores.csv")

OUT_DIR = os.path.join(ROOT, "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_671 = os.path.join(OUT_DIR, "Figure_6_7_1_MIL_Aggregation_AUROC.png")
FIG_672 = os.path.join(OUT_DIR, "Figure_6_7_2_LOSO_AUROC_Comparison.png")


def _infer_label_column(df: pd.DataFrame):
    # Expect either y in {0,1} or label in {"healthy","unhealthy"}.
    if "y" in df.columns:
        y = df["y"].astype(int).to_numpy()
        return y
    if "label" in df.columns:
        y = (df["label"].astype(str).str.lower() == "unhealthy").astype(int).to_numpy()
        return y
    raise ValueError("Could not find label column. Expected 'y' or 'label' in CSV.")


def compute_aucs_from_src_df(df: pd.DataFrame, score_cols):
    y = _infer_label_column(df)
    aucs = {}
    for c in score_cols:
        if c not in df.columns:
            continue
        s = df[c].astype(float).to_numpy()
        # guard: need both classes present
        if len(np.unique(y)) < 2:
            aucs[c] = np.nan
        else:
            aucs[c] = float(roc_auc_score(y, s))
    return aucs


def plot_bar(values_dict, title, outfile, highlight_key=None):
    keys = list(values_dict.keys())
    vals = [values_dict[k] for k in keys]

    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys)
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUROC")
    plt.title(title)

    # annotate values
    for i, v in enumerate(vals):
        if np.isnan(v):
            txt = "NA"
        else:
            txt = f"{v:.3f}"
        plt.text(i, 0.02, txt, ha="center", va="bottom")

    # highlight a bar if requested
    if highlight_key is not None and highlight_key in keys:
        idx = keys.index(highlight_key)
        # emphasize with hatch + thicker edge
        bars[idx].set_hatch("//")
        bars[idx].set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print("Saved:", outfile)


def make_fig_6_7_1():
    """
    Figure 6.7.1: MIL aggregation comparison on the (single) evaluation split,
    using the calibrated per-source aggregates saved by 6_mil_topk_eval.py.
    """
    df = pd.read_csv(MIL_CSV)

    # Use these columns if present (your script used these names)
    score_cols = ["mean", "p95", "top10", "max"]

    aucs = compute_aucs_from_src_df(df, score_cols)

    # nice labels
    label_map = {"mean": "mean", "p95": "p95", "top10": "top-10", "max": "max"}
    aucs_nice = {label_map[k]: aucs[k] for k in score_cols if k in aucs}

    plot_bar(
        aucs_nice,
        title="MIL aggregation comparison (calibrated) — Image-level AUROC",
        outfile=FIG_671,
        highlight_key="mean",
    )


def make_fig_6_7_2():
    """
    Figure 6.7.2: LOSO AUROC comparison across aggregation strategies.
    loso_src_scores.csv contains one row per held-out source with columns mean/p95/top10/max and y/label.
    """
    df = pd.read_csv(LOSO_CSV)

    score_cols = ["mean", "p95", "top10", "max"]
    aucs = compute_aucs_from_src_df(df, score_cols)

    label_map = {"mean": "mean", "p95": "p95", "top10": "top-10", "max": "max"}
    aucs_nice = {label_map[k]: aucs[k] for k in score_cols if k in aucs}

    plot_bar(
        aucs_nice,
        title="LOSO cross-validation — Image-level AUROC by aggregation",
        outfile=FIG_672,
        highlight_key="mean",
    )


if __name__ == "__main__":
    # sanity checks
    if not os.path.exists(MIL_CSV):
        raise FileNotFoundError(f"Missing: {MIL_CSV}")
    if not os.path.exists(LOSO_CSV):
        raise FileNotFoundError(f"Missing: {LOSO_CSV}")

    make_fig_6_7_1()
    make_fig_6_7_2()

    print("\nDone.")
    print("Figures saved to:", OUT_DIR)
