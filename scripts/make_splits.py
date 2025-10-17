from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd

from src.models.purged_cv import PurgedKFoldTime


def main():
    # Load labels (assumes timestamp, label, end_ts present)
    paths = sorted(glob.glob(r"data\labels_jan_jul_atr\*.parquet"))
    assert paths, "No label files found at data/labels_jan_jul_atr/*.parquet"
    labels = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    labels["end_ts"] = pd.to_datetime(labels["end_ts"], utc=True)

    splits_dir = Path("splits_jan_jul_atr"); splits_dir.mkdir(parents=True, exist_ok=True)

    cv = PurgedKFoldTime(n_splits=5, embargo=pd.Timedelta(minutes=10))
    t0 = labels["timestamp"]
    t1 = labels["end_ts"]

    for k, (tr, te) in enumerate(cv.split(labels, t0=t0, t1=t1), start=1):
        (labels.iloc[tr][["timestamp", "label"]]
               .to_parquet(splits_dir / f"fold{k}_train.parquet", index=False))
        (labels.iloc[te][["timestamp", "label"]]
               .to_parquet(splits_dir / f"fold{k}_val.parquet", index=False))
        print(f"fold{k}: train={len(tr):,}  val={len(te):,}  wrote -> {splits_dir}")


if __name__ == "__main__":
    main()

