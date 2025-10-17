from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd

from src.labeling.weights import compute_uniqueness_weights


def main():
    # Load all labels to attach end_ts/time_to_hit if needed
    all_paths = sorted(glob.glob(r"data\labels_jan_jul_atr\*.parquet"))
    assert all_paths, "No label files found at data/labels_jan_jul_atr/*.parquet"
    labels_all = pd.concat([pd.read_parquet(p) for p in all_paths], ignore_index=True)
    labels_all["timestamp"] = pd.to_datetime(labels_all["timestamp"], utc=True)

    splits = Path("splits_jan_jul_atr")
    outdir = Path("weights_jan_jul_atr"); outdir.mkdir(parents=True, exist_ok=True)

    for k in range(1, 6):
        tr = pd.read_parquet(splits / f"fold{k}_train.parquet")
        tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)

        # Join auxiliary fields to infer t1 if needed
        aux_cols = [c for c in ("end_ts", "time_to_hit_sec", "label") if c in labels_all.columns]
        df = tr.merge(labels_all[["timestamp", *aux_cols]], on="timestamp", how="left")

        w = compute_uniqueness_weights(df, timestamp_col="timestamp", horizon_minutes=30, normalize=True)
        df_out = df[["timestamp", "label"]].copy()
        df_out["sample_w"] = w.values
        df_out.to_parquet(outdir / f"fold{k}_train_weights.parquet", index=False)
        print(f"fold{k}: wrote {len(df_out):,} weights -> {outdir}")


if __name__ == "__main__":
    main()

