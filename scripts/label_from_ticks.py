import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import math

from src.labeling.triple_barrier import ensure_datetime_index


# ---------- helpers ----------
def pick_cols_case_insensitive(df_or_cols, pref):
    cols = df_or_cols
    if not isinstance(df_or_cols, (list, tuple, set)):
        cols = df_or_cols.columns
    m = {str(c).lower(): c for c in cols}
    return m.get(pref)


def parse_ticks_ts(series):
    # numeric ms or string/mixed -> numpy datetime64[ns]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce").to_numpy("datetime64[ns]")
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed").to_numpy("datetime64[ns]")


def ts_price_names_from_header_csv(fp):
    hdr = pd.read_csv(fp, nrows=0)
    cols = list(hdr.columns)
    ts = (
        pick_cols_case_insensitive(cols, "timestamp")
        or pick_cols_case_insensitive(cols, "ts")
        or pick_cols_case_insensitive(cols, "time")
    )
    price = (
        pick_cols_case_insensitive(cols, "price")
        or pick_cols_case_insensitive(cols, "last")
        or pick_cols_case_insensitive(cols, "close")
        or pick_cols_case_insensitive(cols, "mid")
    )
    return ts, price


def parse_ticks_ts_array(values):
    # numeric ms or ISO strings -> numpy datetime64[ns]
    if pd.api.types.is_integer_dtype(values) or pd.api.types.is_float_dtype(values):
        return pd.to_datetime(values, unit="ms", utc=True, errors="coerce").to_numpy("datetime64[ns]")
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed").to_numpy("datetime64[ns]")


def load_ticks_csv_window(fp, ts_col, price_col, t_min, t_max, chunksize=5_000_000):
    """Stream a large CSV and keep only rows within [t_min, t_max]."""
    # Work in milliseconds window to compare quickly
    tmin_ns = np.datetime64(t_min.to_datetime64(), "ns").astype("int64")
    tmax_ns = np.datetime64(t_max.to_datetime64(), "ns").astype("int64")

    # We’ll read timestamp as object (string) or int; price as float32 to save RAM
    usecols = [ts_col, price_col]
    tts_list, px_list = [], []
    for chunk in pd.read_csv(fp, usecols=usecols, chunksize=chunksize):
        # parse ts column
        ts_vals = parse_ticks_ts_array(chunk[ts_col])
        # drop NaT
        mask = ~pd.isna(ts_vals)
        if not mask.any():
            continue
        ts_vals = ts_vals[mask]
        px = chunk.loc[mask, price_col].astype("float32").to_numpy()

        # fast filter by numeric ns
        ts_ns = ts_vals.astype("datetime64[ns]").astype("int64")
        keep = (ts_ns >= tmin_ns) & (ts_ns <= tmax_ns)
        if keep.any():
            tts_list.append(ts_vals[keep])
            px_list.append(px[keep])

    if not tts_list:
        # nothing in window
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype="float32")
    tts = np.concatenate(tts_list)
    px = np.concatenate(px_list)
    order = np.argsort(tts)
    return tts[order], px[order]


def load_ticks_parquet_window(fp, t_min, t_max):
    # Read only timestamp + price-like columns, then filter by time
    tk = pd.read_parquet(fp)
    cols = {c.lower(): c for c in tk.columns}
    ts_col = cols.get("timestamp") or cols.get("ts") or cols.get("time")
    price_col = cols.get("price") or cols.get("last") or cols.get("close") or cols.get("mid")
    if ts_col is None or price_col is None:
        raise SystemExit(f"Parquet ticks missing ts/price in {fp}: {list(tk.columns)}")
    ts_vals = parse_ticks_ts_array(tk[ts_col])
    px = tk[price_col].astype("float32").to_numpy()
    mask = (~pd.isna(ts_vals)) & (ts_vals >= t_min.to_datetime64()) & (ts_vals <= t_max.to_datetime64())
    ts_vals = ts_vals[mask]
    px = px[mask]
    order = np.argsort(ts_vals)
    return ts_vals[order], px[order]


def load_ticks(ticks_root: Path, symbol: str, month: str):
    # Try likely file names (ticks + 1m variants; parquet or CSV)
    cands = [
        ticks_root / f"{symbol}-ticks-{month}.parquet",
        ticks_root / f"{symbol}-ticks-{month}.csv",
        ticks_root / f"{symbol}-1m-{month}.parquet",
        ticks_root / f"{symbol}-1m-{month}.csv",
    ]
    for fp in cands:
        if fp.exists():
            if fp.suffix == ".parquet":
                return pd.read_parquet(fp), fp
            return pd.read_csv(fp), fp
    return None, None


def pick_price_columns_for_atr(bars):
    # case-insensitive, fallback to 'close'
    m = {c.lower(): c for c in bars.columns}
    close = m.get("close") or m.get("price")
    if close is None:
        raise ValueError(f"No close/price column in bars: {list(bars.columns)}")
    high = m.get("high", close)
    low = m.get("low", close)
    return bars[close], bars[high], bars[low]


def atr_wilder(close, high, low, n=14):
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def touch_label_ticks(tts, px, t0, t_end, up, dn):
    # earliest first-touch using a price stream
    i0 = int(np.searchsorted(tts, t0.to_datetime64(), side="right"))
    i1 = int(np.searchsorted(tts, t_end.to_datetime64(), side="right"))
    if i0 >= i1:
        return 2
    seg = px[i0:i1]
    if seg.size == 0:
        return 2
    tp_idx = np.where(seg >= up)[0]
    sl_idx = np.where(seg <= dn)[0]
    if tp_idx.size == 0 and sl_idx.size == 0:
        return 2
    if tp_idx.size == 0:
        return 0
    if sl_idx.size == 0:
        return 1
    return 1 if tp_idx[0] <= sl_idx[0] else 0


def touch_label_hilo(tts, hi, lo, t0, t_end, up, dn):
    # earliest first-touch using bar highs/lows
    i0 = int(np.searchsorted(tts, t0.to_datetime64(), side="right"))
    i1 = int(np.searchsorted(tts, t_end.to_datetime64(), side="right"))
    if i0 >= i1:
        return 2
    seg_h = hi[i0:i1]
    seg_l = lo[i0:i1]
    tp_idx = np.where(seg_h >= up)[0]
    sl_idx = np.where(seg_l <= dn)[0]
    if tp_idx.size == 0 and sl_idx.size == 0:
        return 2
    if tp_idx.size == 0:
        return 0
    if sl_idx.size == 0:
        return 1
    return 1 if tp_idx[0] <= sl_idx[0] else 0


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ticks_root", required=True)
    ap.add_argument("--bars_root", required=True)
    ap.add_argument("--atr_window", type=int, default=14)
    ap.add_argument("--tp_mult", type=float, required=True)
    ap.add_argument("--sl_mult", type=float, required=True)
    ap.add_argument("--horizon_minutes", type=int, required=True)
    ap.add_argument("--resume", action="store_true", help="Skip months that already have chunk files")
    ap.add_argument(
        "--mode",
        choices=["auto", "ticks", "hilo"],
        default="auto",
        help=(
            "Labeling source: 'ticks' requires tick price stream, "
            "'hilo' forces OHLC, 'auto' prefers ticks then OHLC."
        ),
    )
    ap.add_argument("--tp_sl_policy_json", type=str, default=None,
                    help="JSON mapping group-> {tp_mult, sl_mult}. If given, overrides tp/sl per row.")
    ap.add_argument("--policy_group", choices=["global","month","regime"], default="month")
    ap.add_argument("--month_to_regime_json", type=str, default=None,
                    help="JSON mapping month->regime; used when --policy_group=regime")
    args = ap.parse_args()

    # --- begin: policy loading (place right after args = parser.parse_args()) ---
    import json

    policy = None
    if getattr(args, "tp_sl_policy_json", None):
        with open(args.tp_sl_policy_json, "r") as f:
            raw = json.load(f)
        # Normalize keys to strings for consistent lookup
        policy = {str(k): v for k, v in raw.items()}
        print(f"[policy] loaded {len(policy)} groups from {args.tp_sl_policy_json} (group={args.policy_group})")

    month_to_regime = None
    if getattr(args, "policy_group", None) == "regime":
        if not getattr(args, "month_to_regime_json", None):
            raise ValueError("--month_to_regime_json is required when --policy_group=regime")
        with open(args.month_to_regime_json, "r") as f:
            month_to_regime = json.load(f)
        # Keep keys as strings "YYYY-MM"
        month_to_regime = {str(k): str(v) for k, v in month_to_regime.items()}
        print(f"[policy] loaded month→regime mapping with {len(month_to_regime)} months")

    def _tp_sl_for_month(month_str: str):
        """
        Resolve TP/SL for a given sample month:
          * default to args.tp_mult / args.sl_mult
          * override from policy if provided
        """
        tp = float(args.tp_mult)
        sl = float(args.sl_mult)

        if policy is None:
            return tp, sl

        key = None
        if args.policy_group == "month":
            key = str(month_str)
        elif args.policy_group == "regime":
            if month_to_regime is None:
                return tp, sl
            key = month_to_regime.get(str(month_str))

        if key is not None:
            entry = policy.get(str(key))
            if entry:
                tp = float(entry.get("tp_mult", tp))
                sl = float(entry.get("sl_mult", sl))
        return tp, sl
    # --- end: policy loading ---

    ticks_root = Path(args.ticks_root)
    out_path = Path(args.out_csv)
    chunk_dir = out_path.with_suffix("")  # folder for chunk files
    chunk_dir.mkdir(parents=True, exist_ok=True)

    snap = pd.read_csv(args.snapshot)
    req = {"source_file", "month", "symbol", "idx_in_shard", "window", "t0"}
    missing = req - set(snap.columns)
    if missing:
        raise SystemExit(f"snapshot missing columns: {sorted(missing)}")

    snap["idx_in_shard"] = snap["idx_in_shard"].astype(int)
    snap["window"] = snap["window"].astype(int)
    snap["t0"] = pd.to_datetime(snap["t0"], utc=True, errors="coerce", format="mixed")

    months = (
        snap[["source_file", "month", "symbol"]].drop_duplicates().to_records(index=False).tolist()
    )
    rows_total = len(snap)

    outer = tqdm(total=len(months), desc="months", unit="mo", position=0, leave=True)
    inner = tqdm(total=rows_total, desc="rows", unit="row", position=1, leave=True, miniters=2000)

    for src, month, symbol in months:
        g = snap[(snap["source_file"] == src) & (snap["month"] == month) & (snap["symbol"] == symbol)]
        chunk_file = chunk_dir / f"chunk_{symbol}_{month}.csv"
        if args.resume and chunk_file.exists():
            inner.update(len(g))
            outer.update(1)
            continue

        outer.set_postfix_str(f"{symbol} {month}")
        t0_wall = time.time()

        # bars + ATR for this shard
        bars = ensure_datetime_index(pd.read_parquet(src))
        close, high_b, low_b = pick_price_columns_for_atr(bars)
        atr = atr_wilder(close, high_b, low_b, args.atr_window)
        bidx = bars.index
        
        # compute time window only needed for this month’s rows
        starts = (g["idx_in_shard"] + g["window"] - 1).astype(int).to_numpy()
        valid = starts < (len(bidx) - 1)
        if not np.any(valid):
            # write chunk of timeouts and continue
            g_out = g.copy(); g_out["label"] = 2; g_out["event_end_ts"] = pd.NaT
            g_out.to_csv(chunk_file, index=False)
            inner.update(len(g)); outer.update(1)
            continue
        starts = starts[valid]
        t0s = bidx[starts]
        t_min = t0s.min()
        t_max = t0s.max() + pd.Timedelta(minutes=args.horizon_minutes)

        # ---- ticks-only mode, windowed loading ----
        hi = lo = px = None
        if args.mode == "ticks":
            tk_path_candidates = [
                ticks_root / f"{symbol}-ticks-{month}.parquet",
                ticks_root / f"{symbol}-ticks-{month}.csv",
            ]
            fp = None
            for cand in tk_path_candidates:
                if cand.exists():
                    fp = cand
                    break
            if fp is None:
                raise SystemExit(f"--mode=ticks requested but no file found for {symbol} {month} under {ticks_root}")

            if fp.suffix == ".parquet":
                tts, px = load_ticks_parquet_window(fp, t_min, t_max)
            else:
                ts_col, price_col = ts_price_names_from_header_csv(fp)
                if ts_col is None or price_col is None:
                    raise SystemExit(f"--mode=ticks but missing ts/price columns in {fp}")
                tts, px = load_ticks_csv_window(fp, ts_col, price_col, t_min, t_max)

            if tts.size == 0:
                # No ticks in the window → all timeouts
                g_out = g.copy()
                g_out["label"] = 2
                g_out["event_end_ts"] = g_out["t0"] + pd.Timedelta(minutes=args.horizon_minutes)
                g_out.to_csv(chunk_file, index=False)
                inner.update(len(g)); outer.update(1)
                continue

            mode = "ticks"  # explicit
        else:
            # Retain existing auto/hilo behavior for non-ticks mode
            tk, fp = load_ticks(ticks_root, symbol, month)
            if tk is None:
                print(
                    f"[warn] missing ticks for {month}: {ticks_root}/{symbol}-ticks-{month}.* or {symbol}-1m-{month}.*"
                )
                g_out = g.copy()
                g_out["label"] = 2
                g_out.to_csv(chunk_file, index=False)
                inner.update(len(g)); outer.update(1)
                continue
            tk.columns = tk.columns.str.strip()
            cols = {c.lower(): c for c in tk.columns}
            ts_col = cols.get("timestamp") or cols.get("ts") or cols.get("time")
            if ts_col is None:
                print(f"[warn] ticks missing ts column: {list(tk.columns)} in {fp}; defaulting to timeouts")
                g_out = g.copy(); g_out["label"] = 2
                g_out.to_csv(chunk_file, index=False)
                inner.update(len(g)); outer.update(1)
                continue
            tts = parse_ticks_ts(tk[ts_col])
            ok = ~pd.isna(tts)
            tk = tk.loc[ok]
            order = np.argsort(tts[ok])
            tts = tts[ok][order]

            mode = None
            if args.mode == "hilo":
                if not ("high" in cols and "low" in cols):
                    raise SystemExit(
                        f"--mode=hilo requested but no high/low columns in {fp}: {list(tk.columns)}"
                    )
                hi = tk[cols["high"]].to_numpy(float)[order]
                lo = tk[cols["low"]].to_numpy(float)[order]
                mode = "hilo"
            else:  # auto: prefer ticks then OHLC
                price_col = cols.get("price") or cols.get("last") or cols.get("close") or cols.get("mid")
                if price_col is not None:
                    px = tk[price_col].to_numpy(float)[order]
                    mode = "ticks"
                elif "high" in cols and "low" in cols:
                    hi = tk[cols["high"]].to_numpy(float)[order]
                    lo = tk[cols["low"]].to_numpy(float)[order]
                    mode = "hilo"
                else:
                    print(
                        f"[warn] ticks missing price/high/low columns: {list(tk.columns)} in {fp}; defaulting to timeouts"
                    )
                    g_out = g.copy(); g_out["label"] = 2
                    g_out.to_csv(chunk_file, index=False)
                    inner.update(len(g)); outer.update(1)
                    continue

        # label this chunk
        out_rows = []
        for _, r in g.iterrows():
            start = int(r.idx_in_shard + r.window - 1)
            lab = 2
            if start < len(bidx) - 1:
                t0 = bidx[start]
                t_end = t0 + pd.Timedelta(minutes=args.horizon_minutes)
                c0 = float(close.iloc[start])
                a0 = float(atr.iloc[start])
                if np.isfinite(a0) and a0 > 0:
                    month_str = str(r["month"]) if "month" in r else None
                    tp_mult, sl_mult = _tp_sl_for_month(month_str)
                    up = c0 + tp_mult * a0
                    dn = c0 - sl_mult * a0
                    if mode == "ticks":
                        lab = touch_label_ticks(tts, px, t0, t_end, up, dn)
                    else:
                        lab = touch_label_hilo(tts, hi, lo, t0, t_end, up, dn)
            rd = r.to_dict()
            rd["label"] = lab
            rd["event_end_ts"] = t_end
            out_rows.append(rd)
            inner.update(1)

        pd.DataFrame(out_rows).to_csv(chunk_file, index=False)
        dt = time.time() - t0_wall
        outer.set_postfix_str(f"{symbol} {month} ({len(g)} rows in {dt:.1f}s)")
        outer.update(1)

    outer.close()
    inner.close()

    # merge chunks
    chunks = sorted(chunk_dir.glob("chunk_*.csv"))
    if not chunks:
        print("[tick-touch] no chunks to merge")
        return
    pd.concat((pd.read_csv(p) for p in chunks), ignore_index=True).to_csv(out_path, index=False)
    print(f"[tick-touch] wrote: {out_path} (from {len(chunks)} chunks)")


if __name__ == "__main__":
    main()
