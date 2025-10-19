import argparse, numpy as np, pandas as pd
from src.labeling.triple_barrier import ensure_datetime_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--horizon_minutes", type=int, required=True)
    ap.add_argument("--atr_window", type=int, default=14)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--q_tp", type=float, default=None, help="optional percentile for TP multiplier (0-1)")
    ap.add_argument("--q_sl", type=float, default=None, help="optional percentile for SL multiplier (0-1)")
    args = ap.parse_args()

    df = pd.read_csv(args.snapshot, usecols=["source_file", "idx_in_shard", "window"]).sample(
        args.sample, random_state=7
    )
    df["idx_in_shard"] = df["idx_in_shard"].astype(int)
    df["window"] = df["window"].astype(int)

    mfeR, maeR = [], []
    for fp, g in df.groupby("source_file"):
        b = ensure_datetime_index(pd.read_parquet(fp))
        cols = {c.lower(): c for c in b.columns}
        c = b[cols.get("close") or cols.get("price")]
        h = b.get(cols.get("high"), c)
        l = b.get(cols.get("low"), c)
        prev = c.shift(1)
        TR = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        ATR = TR.ewm(alpha=1 / args.atr_window, adjust=False).mean()
        idx = b.index.values
        starts = (g["idx_in_shard"] + g["window"] - 1).astype(int).to_numpy()
        for s in starts:
            if s >= len(b) - 1:
                continue
            t_end = pd.Timestamp(idx[s]).tz_localize("UTC") + pd.Timedelta(minutes=args.horizon_minutes)
            e = int(np.searchsorted(idx, t_end.to_datetime64(), side="right") - 1)
            e = min(max(e, s + 1), len(b) - 1)
            c0 = float(c.iloc[s])
            a0 = float(ATR.iloc[s])
            if not (np.isfinite(a0) and a0 > 0):
                continue
            seg_h = float(h.iloc[s + 1 : e + 1].max())
            seg_l = float(l.iloc[s + 1 : e + 1].min())
            mfeR.append(max(0.0, seg_h - c0) / a0)
            maeR.append(max(0.0, c0 - seg_l) / a0)

    mfeR, maeR = np.array(mfeR), np.array(maeR)

    def q(arr, p):
        return float(np.nanquantile(arr, p))

    print(
        "MFE/ATR @ 0.90/0.95/0.98/0.99/0.995:",
        *(q(mfeR, p) for p in (0.90, 0.95, 0.98, 0.99, 0.995)),
    )
    print("MAE/ATR @ 0.60/0.70/0.80/0.90:", *(q(maeR, p) for p in (0.60, 0.70, 0.80, 0.90)))
    if args.q_tp is not None or args.q_sl is not None:
        if args.q_tp is not None:
            print("tp_mult:", q(mfeR, args.q_tp))
        if args.q_sl is not None:
            print("sl_mult:", q(maeR, args.q_sl))


if __name__ == "__main__":
    main()

