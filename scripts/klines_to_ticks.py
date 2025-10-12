import argparse, glob, os, re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_volume", "taker_buy_quote_asset_volume", "ignore"
]

def _to_ms(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = x.dropna().median() if len(x.dropna()) else 0
    # Heuristic: ms ~1e12, µs ~1e15, ns ~1e18
    if med >= 1e18:   # ns -> ms
        return (x / 1_000_000).round()
    if med >= 1e15:   # µs -> ms
        return (x / 1_000).round()
    if med < 1e12:    # s -> ms
        return (x * 1_000).round()
    return x.round()  # already ms

def _read_kline_csv(path: str) -> pd.DataFrame:
    # Headerless or with header, both supported
    try:
        df = pd.read_csv(path, header=None, names=KLINE_COLS, dtype=str)
        if df.shape[1] != 12:
            raise ValueError
    except Exception:
        df = pd.read_csv(path)
        # If vendor header names differ, align them by position
        if df.shape[1] != 12:
            raise ValueError(f"Unexpected number of columns in {path}: {df.shape[1]}")
        df.columns = KLINE_COLS
    # Cast numerics
    num_cols = ["open","high","low","close","volume","quote_asset_volume",
                "number_of_trades","taker_buy_volume","taker_buy_quote_asset_volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["open_time_ms"]  = _to_ms(df["open_time"])
    df["close_time_ms"] = _to_ms(df["close_time"])
    return df

def _bridge_path(open_, high, low, close, n, rng: np.random.Generator):
    n = int(max(2, n))
    t = np.linspace(0, 1, n)
    # Choose order: open→(hi/lo)→(lo/hi)→close
    hi_first = bool(rng.integers(0, 2))
    i_hi = int(rng.integers(1, n-1))
    i_lo = int(rng.integers(1, n-1))
    if i_hi == i_lo:
        i_lo = max(1, min(n-2, i_lo + (1 if i_lo < (n-1)//2 else -1)))
    anchors = {0: open_, n-1: close, i_hi: high, i_lo: low}

    # Piecewise-linear through sorted anchor indices
    idxs = sorted(anchors.keys())
    y = np.zeros(n, dtype=float)
    for a, b in zip(idxs[:-1], idxs[1:]):
        y[a:b+1] = np.linspace(anchors[a], anchors[b], b-a+1)
    # Gentle noise, then clamp to [low, high]
    noise = rng.normal(0, 0.001 * max(1e-9, high - low), size=n)
    y = np.clip(y + noise, low, high)
    # Enforce exact OHLC at anchors
    y[0], y[-1], y[i_hi], y[i_lo] = open_, close, high, low
    return y

def _split_volume(total_vol: float, n: int, rng: np.random.Generator):
    if total_vol <= 0 or n <= 0:
        return np.zeros(n, dtype=float)
    # Dirichlet partition -> sums exactly to total_vol, min floor to avoid zeros
    w = rng.dirichlet(np.full(n, 1.0))
    qty = total_vol * w
    # Prevent zero-qty rows from being dropped downstream
    qty = np.maximum(qty, total_vol * 1e-8)
    # Re-normalize exact conservation
    return qty * (total_vol / qty.sum())

def _ibm_flags(n: int, buy_taker_share: float, rng: np.random.Generator):
    # is_buyer_maker == 1 means sell-taker; sell share = 1 - buy_taker_share
    p_sell = float(np.clip(1.0 - buy_taker_share, 0.0, 1.0))
    return (rng.random(n) < p_sell).astype(np.int8)

def kline_to_ticks(row, mode: str, cap_trades: int, tpm: int, rng: np.random.Generator):
    open_, high, low, close = float(row.open), float(row.high), float(row.low), float(row.close)
    vol = float(row.volume)
    n_trades = int(row.number_of_trades)
    buy_taker_vol = float(row.taker_buy_volume)
    buy_share = (buy_taker_vol / vol) if vol > 0 else 0.5

    if mode == "tradecount":
        n = max(4, min(cap_trades, n_trades if n_trades > 0 else tpm))
    else:  # fixed ticks per minute
        n = max(4, int(tpm))

    # Times
    t0 = int(row.open_time_ms); t1 = int(row.close_time_ms)
    ts = np.linspace(t0, max(t0+1, t1), n).astype(np.int64)

    # Price path + qty + side flags
    prices = _bridge_path(open_, high, low, close, n, rng)
    qty = _split_volume(vol, n, rng)
    ibm = _ibm_flags(n, buy_share, rng)

    df = pd.DataFrame({
        "timestamp": ts,
        "price": prices.astype(np.float32),
        "qty": qty.astype(np.float32),
        "is_buyer_maker": ibm,
    })
    return df

def infer_symbol(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"^([A-Z0-9]+)-1m-", base)
    return m.group(1) if m else os.path.splitext(base)[0].split("-")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klines_glob", required=True, help='e.g. "data/bars_klines/*.csv"')
    ap.add_argument("--out_dir", required=True, help="Where to write tick CSVs")
    ap.add_argument("--mode", choices=["tradecount","tpm"], default="tradecount")
    ap.add_argument("--max_trades_per_bar", type=int, default=600, help="Cap when mode=tradecount")
    ap.add_argument("--ticks_per_minute", type=int, default=120, help="Used when mode=tpm or trades=0")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(args.klines_glob))
    if not files:
        tqdm.write("No kline files found."); return

    for p in tqdm(files, desc="Synthesizing ticks from klines", unit="file"):
        df = _read_kline_csv(p)
        if df.empty: 
            tqdm.write(f"Empty kline file {p}"); continue

        # Build ticks per minute and concat
        ticks = []
        for _, row in df.iterrows():
            ticks.append(kline_to_ticks(
                row,
                mode=args.mode,
                cap_trades=args.max_trades_per_bar,
                tpm=args.ticks_per_minute,
                rng=rng
            ))
        out = pd.concat(ticks, ignore_index=True)
        sym = infer_symbol(p)
        # Use original filename stem for continuity
        stem = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.out_dir, f"{stem}.csv")
        out.to_csv(out_path, index=False)
        tqdm.write(f"Wrote {out_path} {len(out):,} ticks")

if __name__ == "__main__":
    main()
