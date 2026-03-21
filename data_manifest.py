# data_manifest.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_manifest(manifest_csv: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    df["path"] = df["path"].astype(str)
    return df

def split_by_source(df: pd.DataFrame, test_size=0.2, seed=42):
    train_parts = []
    val_parts = []

    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        srcs = sorted(sub["src_id"].unique())
        tr_src, va_src = train_test_split(srcs, test_size=test_size, random_state=seed)

        train_parts.append(sub[sub["src_id"].isin(tr_src)])
        val_parts.append(sub[sub["src_id"].isin(va_src)])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    return train_df, val_df
