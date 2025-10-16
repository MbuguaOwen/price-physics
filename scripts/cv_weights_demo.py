from __future__ import annotations
import glob, pandas as pd, numpy as np
from pathlib import Path
from src.models.purged_cv import PurgedKFoldTime
from src.labeling.weights import compute_uniqueness_weights

LABEL_GLOB = r"data\labels_jan_jul_atr\*.parquet"

def main():
    paths = sorted(glob.glob(LABEL_GLOB))
    assert paths, f"No label files at {LABEL_GLOB}"
    labels = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    t0 = pd.to_datetime(labels.get("timestamp"), utc=True, errors="coerce")
    t1 = pd.to_datetime(labels.get("end_ts"),   utc=True, errors="coerce")
    if not (t0.notna().all() and t1.notna().all()):
        raise AssertionError("Need timestamp and end_ts (relabel with end_ts patch).")

    cv = PurgedKFoldTime(n_splits=5, embargo=pd.Timedelta(minutes=10))

    for fold,(tr_idx, te_idx) in enumerate(cv.split(labels, t0=t0, t1=t1), start=1):
        train = labels.iloc[tr_idx].copy()
        test  = labels.iloc[te_idx].copy()

        # Compute uniqueness weights ONLY on the training subset
        w = compute_uniqueness_weights(train, horizon_minutes=30, normalize=True)
        train["uniq_w"] = w.values

        # (Optional) combine with class weights
        if "label" in train.columns:
            cls_w = train["label"].map(train["label"].value_counts(normalize=True)).rpow(-1.0)
            cls_w /= cls_w.mean()
            train["sample_w"] = train["uniq_w"] * train["label"].map(cls_w).fillna(1.0)
            mean_sw = float(train["sample_w"].mean())
        else:
            mean_sw = float(train["uniq_w"].mean())

        print(
            f"Fold {fold}: train={len(train):,} test={len(test):,} "
            f"mean uniq_w={float(train['uniq_w'].mean()):.3f} mean sample_w={mean_sw:.3f}"
        )

if __name__ == "__main__":
    main()

