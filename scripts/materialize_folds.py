from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd


def main():
    LABELS_DIR = Path(r"data\labels_jan_jul_atr")
    SPLITS_DIR = Path(r"splits_jan_jul_atr")
    WEIGHTS_DIR = Path(r"weights_jan_jul_atr")
    OUT_BASE = Path(r"data\labels_jan_jul_atr_folds")
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Index labels by timestamp for fast lookups
    label_files = sorted(glob.glob(str(LABELS_DIR / "*.parquet")))
    assert label_files, f"No label files under {LABELS_DIR}"
    labels = pd.concat([pd.read_parquet(p) for p in label_files], ignore_index=True)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)

    for k in range(1, 6):
        tr = pd.read_parquet(SPLITS_DIR / f"fold{k}_train.parquet")
        va = pd.read_parquet(SPLITS_DIR / f"fold{k}_val.parquet")
        tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
        va["timestamp"] = pd.to_datetime(va["timestamp"], utc=True)

        # Merge train weights
        w = pd.read_parquet(WEIGHTS_DIR / f"fold{k}_train_weights.parquet")
        w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True)

        train_df = tr.merge(labels, on=["timestamp", "label"], how="left")
        train_df = train_df.merge(w[["timestamp", "sample_w"]], on="timestamp", how="left")
        train_df["sample_w"] = train_df["sample_w"].fillna(1.0)

        val_df = va.merge(labels, on=["timestamp", "label"], how="left")

        out_train = OUT_BASE / f"fold{k}_train"
        out_val = OUT_BASE / f"fold{k}_val"
        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(out_train / "labels.parquet", index=False)
        val_df.to_parquet(out_val / "labels.parquet", index=False)

        print(f"Fold {k}: wrote {len(train_df):,} train and {len(val_df):,} val rows -> {OUT_BASE}")


if __name__ == "__main__":
    main()

