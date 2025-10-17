# scripts/label_from_ticks.py
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.labeling.triple_barrier import ensure_datetime_index


# ---------- helpers ----------
def pick_cols_case_insensitive(df, pref):
    m = {c.lower(): c for c in df.columns}
    return m.get(pref)

def parse_ticks_ts(series):
    # numeric ms or string/mixed -> numpy datetime64[ns]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce").to_numpy("datetime64[ns]")
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed").to_numpy("datetime64[ns]")

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
    low  = m.get("low",  close)
    return bars[close], bars[high], bars[low]

def atr_wilder(close, high, low, n=14):
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

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
    seg_h = hi[i0:i1]; seg_l = lo[i0:i1]
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
    args = ap.parse_args()

    ticks_root = Path(args.ticks_root)
    out_path = Path(args.out_csv)
    chunk_dir = out_path.with_suffix("")  # folder for chunk files
    chunk_dir.mkdir(parents=True, exist_ok=True)

    snap = pd.read_csv(args.snapshot)
    req = {"source_file","month","symbol","idx_in_shard","window","t0"}
    missing = req - set(snap.columns)
    if missing:
        raise SystemExit(f"snapshot missing columns: {sorted(missing)}")

    snap["idx_in_shard"] = snap["idx_in_shard"].astype(int)
    snap["window"] = snap["window"].astype(int)
    snap["t0"] = pd.to_datetime(snap["t0"], utc=True, errors="coerce", format="mixed")

    months = snap[["source_file","month","symbol"]].drop_duplicates().to_records(index=False).tolist()
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

        # ticks (or 1m OHLC)
        tk, fp = load_ticks(ticks_root, symbol, month)
        if tk is None:
            print(f"[warn] missing ticks for {month}: {ticks_root}/{symbol}-ticks-{month}.* or {symbol}-1m-{month}.*")
            # write timeouts so we keep shape consistent
            g_out = g.copy()
            g_out["label"] = 2
            g_out.to_csv(chunk_file, index=False)
            inner.update(len(g))
            outer.update(1)
            continue

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

        # choose price representation
        mode = "ticks"
        px = None; hi = None; lo = None
        if "high" in cols and "low" in cols:
            mode = "hilo"
            hi = tk[cols["high"]].to_numpy(float)[order]
            lo = tk[cols["low"]].to_numpy(float)[order]
        else:
            price_col = cols.get("price") or cols.get("close") or cols.get("last") or cols.get("mid")
            if price_col is None:
                print(f"[warn] ticks missing price columns: {list(tk.columns)} in {fp}; defaulting to timeouts")
                g_out = g.copy(); g_out["label"] = 2
                g_out.to_csv(chunk_file, index=False)
                inner.update(len(g)); outer.update(1)
                continue
            px = tk[price_col].to_numpy(float)[order]

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
                    up = c0 + args.tp_mult * a0
                    dn = c0 - args.sl_mult * a0
                    if mode == "ticks":
                        lab = touch_label_ticks(tts, px, t0, t_end, up, dn)
                    else:
                        lab = touch_label_hilo(tts, hi, lo, t0, t_end, up, dn)
            rd = r.to_dict()
            rd["label"] = lab
            out_rows.append(rd)
            inner.update(1)

        pd.DataFrame(out_rows).to_csv(chunk_file, index=False)
        dt = time.time() - t0_wall
        outer.set_postfix_str(f"{symbol} {month} ({len(g)} rows in {dt:.1f}s)")
        outer.update(1)

    outer.close(); inner.close()

    # merge chunks
    chunks = sorted(chunk_dir.glob("chunk_*.csv"))
    if not chunks:
        print("[tick-touch] no chunks to merge")
        return
    pd.concat((pd.read_csv(p) for p in chunks), ignore_index=True).to_csv(out_path, index=False)
    print(f"[tick-touch] wrote: {out_path} (from {len(chunks)} chunks)")

if __name__ == "__main__":
    main()

