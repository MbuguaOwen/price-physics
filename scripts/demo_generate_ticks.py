import argparse, numpy as np, pandas as pd
from tqdm.auto import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    total_ticks = max(0, args.days * 24 * 60 * 60)
    rng = np.random.default_rng(1337)

    ticks_per_chunk = max(1, min(60 * 60, total_ticks or 1))
    price_segments, qty_segments, ts_segments = [], [], []
    base_time = pd.Timestamp("2025-01-01", tz="UTC")
    last_price = 30000.0

    with tqdm(total=total_ticks, desc="Simulating ticks", unit="tick") as pbar:
        generated = 0
        while generated < total_ticks:
            chunk_size = min(ticks_per_chunk, total_ticks - generated)
            increments = rng.normal(0, 5, size=chunk_size)
            prices = last_price + np.cumsum(increments)
            last_price = prices[-1]

            price_segments.append(prices)
            qty_segments.append(rng.uniform(0.01, 0.5, size=chunk_size))
            ts_segments.append(pd.date_range(base_time + pd.Timedelta(seconds=generated),
                                             periods=chunk_size, freq="s", tz="UTC").to_numpy())

            generated += chunk_size
            pbar.update(chunk_size)

    if total_ticks:
        price = np.concatenate(price_segments)
        qty = np.concatenate(qty_segments)
        ts = np.concatenate(ts_segments)
    else:
        price = np.array([])
        qty = np.array([])
        ts = pd.DatetimeIndex([])

    df = pd.DataFrame({"timestamp": ts, "price": price, "qty": qty})
    df.to_csv(args.out, index=False)
    print("Wrote", args.out, len(df))

if __name__ == "__main__":
    main()
