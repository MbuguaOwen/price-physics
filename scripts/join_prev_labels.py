import argparse, os, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)   # e.g., data/images_jan_jul/train_snapshot.csv
    ap.add_argument("--prev_labels", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--label_symbol_col", default="symbol")
    ap.add_argument("--label_time_col", default="t1")  # your previous labels' event time (window end)
    ap.add_argument("--label_value_col", default="label")
    ap.add_argument("--time_tolerance", default="0s")  # e.g., "1s" if clocks drift slightly
    args = ap.parse_args()

    snap = pd.read_csv(args.snapshot, parse_dates=["t0","t1"], low_memory=False)
    labs = pd.read_csv(args.prev_labels, low_memory=False)
    if args.label_time_col in labs.columns:
        labs[args.label_time_col] = pd.to_datetime(labs[args.label_time_col], utc=True, errors="coerce")
    else:
        raise ValueError(f"Previous labels missing time column: {args.label_time_col}")

    # Force UTC for snapshot times
    snap["t1"] = pd.to_datetime(snap["t1"], utc=True, errors="coerce")

    out_frames = []
    unmatched = 0
    for sym, sub in snap.groupby("symbol", sort=False):
        L = labs[labs[args.label_symbol_col] == sym].copy()
        if L.empty:
            continue
        L = L.sort_values(args.label_time_col)
        S = sub.sort_values("t1")
        joined = pd.merge_asof(
            S, L, left_on="t1", right_on=args.label_time_col,
            direction="nearest",
            tolerance=pd.Timedelta(args.time_tolerance)
        )
        out_frames.append(joined)

    if not out_frames:
        print("No matches found; check symbol/time columns and tolerance.")
        return
    out = pd.concat(out_frames, ignore_index=True)

    # Establish label col
    if args.label_value_col not in out.columns:
        raise ValueError(f"Expected label column '{args.label_value_col}' not found in previous labels.")
    out["label"] = out[args.label_value_col]

    # Ensure we have an end time for event span; fallback to t1 if missing
    if "event_end_ts" not in out.columns:
        print("[WARN] previous labels missing 'event_end_ts'; using 't1' as event end.")
        out["event_end_ts"] = out["t1"]

    # Keep minimal required columns + everything else
    cols = list(out.columns)
    # Write
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    cov = out["label"].notna().mean()
    print(f"Wrote: {args.out_csv}  rows={len(out)}  label_coverage={cov:.2%}")

if __name__ == "__main__":
    main()

