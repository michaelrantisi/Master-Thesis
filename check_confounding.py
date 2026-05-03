import os
import re
import argparse
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from data_manifest import load_manifest

MANIFEST = r"C:\Users\User\Desktop\thesis_choroid_pipeline\tiles_out\manifest.csv"
OUT_DIR = r"./runs_confounding"
os.makedirs(OUT_DIR, exist_ok=True)


def infer_source_metadata(src_id: str):
    sid = src_id.lower()
    magnification = "20x" if "20x" in sid else ("4x" if "4x" in sid else "unknown")
    modality = "BF" if "bf" in sid else ("Ph1" if "ph1" in sid else "unknown")
    return magnification, modality


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    df = load_manifest(MANIFEST)
    src_df = df[["src_id", "label"]].drop_duplicates().copy()
    src_df[["magnification", "modality"]] = src_df["src_id"].apply(lambda s: pd.Series(infer_source_metadata(s)))

    src_df.to_csv(os.path.join(OUT_DIR, "source_metadata_table.csv"), index=False)

    mag_tab = pd.crosstab(src_df["label"], src_df["magnification"])
    mod_tab = pd.crosstab(src_df["label"], src_df["modality"])
    comb_tab = pd.crosstab(src_df["label"], src_df["magnification"] + "_" + src_df["modality"])

    mag_tab.to_csv(os.path.join(OUT_DIR, "crosstab_label_vs_magnification.csv"))
    mod_tab.to_csv(os.path.join(OUT_DIR, "crosstab_label_vs_modality.csv"))
    comb_tab.to_csv(os.path.join(OUT_DIR, "crosstab_label_vs_acquisition_group.csv"))

    stats_rows = []
    for name, tab in [("magnification", mag_tab), ("modality", mod_tab), ("acquisition_group", comb_tab)]:
        if tab.shape == (2, 2):
            _, p = fisher_exact(tab.to_numpy())
            stats_rows.append({"variable": name, "test": "Fisher exact", "p_value": p})
        else:
            chi2, p, dof, _ = chi2_contingency(tab.to_numpy())
            stats_rows.append({"variable": name, "test": "Chi-square", "p_value": p, "dof": dof})

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(OUT_DIR, "confounding_tests.csv"), index=False)

    print("Saved source metadata and confounding tables to", OUT_DIR)
    print(stats_df)


if __name__ == "__main__":
    main()
