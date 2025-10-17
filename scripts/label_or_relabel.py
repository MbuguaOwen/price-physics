import argparse, os, glob, pandas as pd, numpy as np
from pathlib import Path

# ---------- IO helpers ----------
def load_any(path, nrows=None):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if ext == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    if ext in (".feather", ".fea"):
        df = pd.read_feather(path)
        return df.head(nrows) if nrows else df
    raise ValueError(f"Unsupported extension: {ext}")

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

# ---------- Previous-labels join ----------
def try_join_prev_labels(snapshot_csv, labels_path, out_csv,
                         label_symbol_col="symbol", label_time_col="t1",
                         label_value_col="label", time_tolerance="0s"):
    snap = pd.read_csv(snapshot_csv, parse_dates=["t0","t1"], low_memory=False)
    labs = load_any(labels_path)
    # Basic sanity: must have symbol + a time column + a label column
    cols = {c.lower(): c for c in labs.columns}
    if label_symbol_col not in labs.columns or label_time_col not in labs.columns or label_value_col not in labs.columns:
        raise ValueError("Previous labels missing required columns; got: "
                         f"{list(labs.columns)} ; need '{label_symbol_col}', '{label_time_col}', '{label_value_col}'")

    labs[label_time_col] = pd.to_datetime(labs[label_time_col], utc=True, errors="coerce")
    snap["t1"] = pd.to_datetime(snap["t1"], utc=True, errors="coerce")

    out_frames = []
    matched = 0
    for sym, sub in snap.groupby("symbol", sort=False):
        L = labs[labs[label_symbol_col] == sym].copy()
        if L.empty: 
            continue
        L = L.sort_values(label_time_col)
        S = sub.sort_values("t1")
        joined = pd.merge_asof(
            S, L, left_on="t1", right_on=label_time_col,
            direction="nearest", tolerance=pd.Timedelta(time_tolerance)
        )
        out_frames.append(joined)
        matched += joined[label_value_col].notna().sum()

    if not out_frames:
        raise RuntimeError("No matches found when aligning previous labels to snapshot.")

    out = pd.concat(out_frames, ignore_index=True)
    out["label"] = out[label_value_col]
    if "event_end_ts" not in out.columns:
        # If no end timestamp in previous labels, fall back to t1; downstream uniqueness & CV will still work.
        out["event_end_ts"] = out["t1"]

    cov = out["label"].notna().mean()
    print(f"[join_prev_labels] matched_rows={matched} coverage={cov:.2%} of snapshot rows={len(out)}")
    save_csv(out, out_csv)
    return out_csv, cov

# ---------- Relabel fallback (ATR + triple barrier) ----------
def ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        for c in ["timestamp","ts","time","datetime","date"]:
            if c in df.columns:
                df = df.set_index(pd.to_datetime(df[c], utc=True, errors="coerce"))
                break
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    return df.sort_index()

def atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = np.empty_like(tr); atr[:] = np.nan
    if len(tr) < n+1: return pd.Series(atr, index=df.index)
    atr[n-1] = np.nanmean(tr[:n]); alpha = 1.0/n
    for i in range(n, len(tr)): atr[i] = (1-alpha)*atr[i-1] + alpha*tr[i]
    return pd.Series(atr, index=df.index)

def compute_end_index(bars_index: pd.DatetimeIndex, start_idx: int,
                      horizon_bars: int | None = None,
                      horizon_minutes: int | None = None) -> int:
    n = len(bars_index)
    if horizon_bars is not None:
        return min(start_idx + horizon_bars, n - 1)
    # minutes case
    t_end = bars_index[start_idx] + pd.Timedelta(minutes=int(horizon_minutes))
    # rightmost index with timestamp <= t_end
    end = int(np.searchsorted(bars_index.values, t_end.to_datetime64(), side="right") - 1)
    end = max(end, start_idx + 1)  # ensure at least one future bar
    return min(end, n - 1)

def first_hit_label(df: pd.DataFrame, i0: int, horizon_bars: int | None, horizon_minutes: int | None, up: float, dn: float):
    hi, lo = df["high"].values, df["low"].values
    end = compute_end_index(df.index, i0, horizon_bars, horizon_minutes)
    if i0 >= end: return 2, i0
    for j in range(i0+1, end+1):
        if hi[j] >= up: return 1, j
        if lo[j] <= dn: return 0, j
    return 2, end

def relabel_snapshot(snapshot_csv, out_csv, atr_window=14, tp_mult=2.0, sl_mult=2.0,
                     horizon_bars=None, horizon_minutes=None, bars_root=""):
    snap = pd.read_csv(snapshot_csv, low_memory=False)
    out_rows = []
    grouped = snap.groupby("source_file", sort=False)
    for src, sub in grouped:
        bars_path = src if os.path.isabs(src) else os.path.join(bars_root, src) if bars_root else src
        if not os.path.exists(bars_path):
            print(f"[WARN] bars missing: {bars_path}"); continue
        df = pd.read_parquet(bars_path)
        df = ensure_dtindex(df)
        df["ATR"] = atr_wilder(df, atr_window)
        idx = df.index
        t1 = pd.to_datetime(sub["t1"].values, utc=True, errors="coerce")
        i0 = np.searchsorted(idx.values, t1.values, side="right") - 1
        i0 = np.clip(i0, 0, len(idx)-1)
        closes, atr = df["close"].values, df["ATR"].values
        labels, end_ix, tp_prices, sl_prices = [], [], [], []
        for i in i0:
            e, a = closes[i], atr[i]
            if not np.isfinite(a):
                labels.append(2); end_ix.append(i); tp_prices.append(np.nan); sl_prices.append(np.nan); continue
            up, dn = e + tp_mult*a, e - sl_mult*a
            lab, j = first_hit_label(df, i, horizon_bars, horizon_minutes, up, dn)
            labels.append(lab); end_ix.append(j); tp_prices.append(up); sl_prices.append(dn)
        sub2 = sub.copy()
        sub2["label"] = labels
        sub2["event_end_ts"] = idx.values[np.array(end_ix)]
        sub2["tp_price"] = tp_prices
        sub2["sl_price"] = sl_prices
        out_rows.append(sub2)
    if not out_rows:
        raise RuntimeError("Relabel produced 0 rows; check bars paths.")
    out = pd.concat(out_rows, ignore_index=True)
    save_csv(out, out_csv)
    print(f"[relabel] wrote: {out_csv} rows={len(out)}")
    return out_csv

# ---------- Orchestrator ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--prev_labels_root", default="outputs")
    ap.add_argument("--prev_labels", default="")  # optional explicit path
    ap.add_argument("--label_symbol_col", default="symbol")
    ap.add_argument("--label_time_col", default="t1")
    ap.add_argument("--label_value_col", default="label")
    ap.add_argument("--time_tolerance", default="0s")
    # relabel fallback params
    ap.add_argument("--atr_window", type=int, default=14)
    ap.add_argument("--tp_mult", type=float, default=2.0)
    ap.add_argument("--sl_mult", type=float, default=2.0)
    ap.add_argument("--horizon_bars", type=int, default=None,
                    help="Vertical barrier as N bars (mutually exclusive with --horizon_minutes)")
    ap.add_argument("--horizon_minutes", type=int, default=None,
                    help="Vertical barrier as wall-time minutes (mutually exclusive with --horizon_bars)")
    ap.add_argument("--bars_root", default="")  # optional prefix for bars paths
    args = ap.parse_args()

    # sanity: exactly one of horizon_bars or horizon_minutes
    if (args.horizon_bars is None) == (args.horizon_minutes is None):
        raise SystemExit("Specify exactly ONE of --horizon_bars or --horizon_minutes")

    # 1) Resolve previous labels path
    cand = []
    if args.prev_labels and os.path.exists(args.prev_labels):
        cand = [args.prev_labels]
    else:
        # find best candidate in outputs/ with likely names
        patterns = ["*label*.csv","*labels*.csv","*labeled*.csv","*label*.parquet","*label*.feather"]
        for pat in patterns:
            cand.extend(glob.glob(os.path.join(args.prev_labels_root, "**", pat), recursive=True))
        # prefer files with many rows and recent mtime
        cand = sorted(set(cand), key=lambda p: (os.path.getmtime(p), os.path.getsize(p)), reverse=True)

    # 2) Try join, else relabel
    for path in cand:
        try:
            print("[try] joining with:", path)
            out_csv, cov = try_join_prev_labels(
                snapshot_csv=args.snapshot, labels_path=path, out_csv=args.out_csv,
                label_symbol_col=args.label_symbol_col, label_time_col=args.label_time_col,
                label_value_col=args.label_value_col, time_tolerance=args.time_tolerance
            )
            if cov >= 0.95:  # good coverage
                return
            else:
                print(f"[info] coverage only {cov:.1%}, will attempt relabel fallback...")
                break
        except Exception as e:
            print("[join failed]", e)

    # Fallback relabel
    relabel_snapshot(
        snapshot_csv=args.snapshot, out_csv=args.out_csv,
        atr_window=args.atr_window, tp_mult=args.tp_mult, sl_mult=args.sl_mult,
        horizon_bars=args.horizon_bars, horizon_minutes=args.horizon_minutes, bars_root=args.bars_root
    )

if __name__ == "__main__":
    main()
