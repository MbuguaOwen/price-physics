from __future__ import annotations
import glob, pandas as pd, numpy as np
from pathlib import Path
from src.labeling.weights import compute_uniqueness_weights

LABEL_GLOB = r"data\labels_jan_jul_atr\*.parquet"
OUT_PATH = Path(r"data\labels_jan_jul_atr\_weights.parquet")
HORIZON_MIN = 30

def main():
    paths = sorted(glob.glob(LABEL_GLOB))
    assert paths, f"No label files at {LABEL_GLOB}"
    frames=[]
    for p in paths:
        df = pd.read_parquet(p)
        # Keep minimal columns
        keep = ["timestamp","label","time_to_hit_sec","end_ts","t1","symbol"]
        cols = [c for c in keep if c in df.columns]
        if "timestamp" not in cols and "timestamp" in df.columns:
            cols.append("timestamp")
        frames.append(df[cols].copy() if cols else df.copy())
    d = pd.concat(frames, ignore_index=True)
    # Ensure timestamp exists
    if "timestamp" not in d.columns:
        raise ValueError("No 'timestamp' column found in labels; cannot compute time windows.")
    w = compute_uniqueness_weights(d, timestamp_col="timestamp", horizon_minutes=HORIZON_MIN, normalize=True)
    out = d[[c for c in ("timestamp","symbol","label") if c in d.columns]].copy()
    out["weight"] = w.values
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote: {OUT_PATH}  rows={len(out)}  mean_w={out['weight'].mean():.4f}  min={out['weight'].min():.4f}  max={out['weight'].max():.4f}")

if __name__ == "__main__":
    main()

