import argparse, pandas as pd, numpy as np
from src.utils.progress import pbar
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot_labeled", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--purge_time", default="1h")
    ap.add_argument("--embargo_time", default="30m")
    ap.add_argument("--progress", action="store_true", help="Show tqdm progress bars")
    args = ap.parse_args()
    df = pd.read_csv(args.snapshot_labeled, low_memory=False)

    # normalize datetimes if present
    for col in ("t0", "t1", "event_end_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")

    # create stable row key tied to THIS snapshot file order
    df = df.reset_index(drop=True)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", df.index.astype("int64"))

    # Optional: ensure month column is string if present
    if "month" in df.columns:
        df["month"] = df["month"].astype(str)

    # Drop rows with missing t0
    df = df[df["t0"].notna()].copy()
    # choose time key for fold slicing: prefer t1, else event_end_ts, else t0
    time_key = "t1" if "t1" in df.columns and df["t1"].notna().any() else ("event_end_ts" if "event_end_ts" in df.columns else "t0")
    df = df.sort_values(time_key).reset_index(drop=True)
    T = df[time_key]; cuts = np.linspace(0, len(T), args.k+1, dtype=int)
    records = []
    bar = pbar(total=args.k, desc="folds") if args.progress else None
    for f in range(args.k):
        lo, hi = cuts[f], cuts[f+1]
        # Ensure window bounds are timezone-aware Timestamps
        t_lo = pd.to_datetime(T.iloc[lo], utc=True)
        t_hi = pd.to_datetime(T.iloc[hi-1], utc=True)
        purge_start = t_lo - pd.Timedelta(args.purge_time)
        embargo_end = t_hi + pd.Timedelta(args.embargo_time)
        if "event_end_ts" in df.columns:
            inter = (df["t0"] <= embargo_end) & (df["event_end_ts"] >= purge_start)
        elif "t1" in df.columns:
            inter = (df["t0"] <= embargo_end) & (df["t1"] >= purge_start)
        else:
            inter = pd.Series(False, index=df.index)
        is_val   = (df[time_key] >= t_lo) & (df[time_key] <= t_hi)
        is_train = (~is_val) & (~inter)

        # Replace positional usage with row_id mapping
        n = len(df)
        row_ids = df["row_id"].to_numpy()

        def clip_idx(idx, n):
            idx = np.asarray(idx, dtype=np.int64)
            return idx[(idx >= 0) & (idx < n)]

        tr_pos = clip_idx(np.flatnonzero(is_train.values), n)
        val_pos = clip_idx(np.flatnonzero(is_val.values), n)

        tr_ids = row_ids[tr_pos]
        val_ids = row_ids[val_pos]

        records.extend([{"row_id": int(r), "fold": f, "split": "train"} for r in tr_ids])
        records.extend([{"row_id": int(r), "fold": f, "split": "val"}   for r in val_ids])
        if bar:
            bar.update(1)
    folds_out = pd.DataFrame.from_records(records)
    folds_out.to_csv(args.out_csv, index=False)
    print("Wrote CV folds:", args.out_csv, "rows=", len(folds_out))
    if bar:
        bar.close()
if __name__ == "__main__":
    main()
