import os, numpy as np, pandas as pd
from src.labeling.triple_barrier import ensure_datetime_index

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--bars_root", default="data/bars_dollar")
    ap.add_argument("--horiz_min", type=int, default=60)
    ap.add_argument("--vol_min", type=int, default=60)
    args = ap.parse_args()

    S = pd.read_csv(args.snapshot, usecols=["source_file","idx_in_shard","window","t0","t1"])
    S = S.reset_index(drop=True); S["row_id"]=S.index.astype("int64")
    S["idx_in_shard"]=S["idx_in_shard"].astype(int); S["window"]=S["window"].astype(int)
    S["t0"]=pd.to_datetime(S["t0"], utc=True, errors="coerce")

    rows=[]
    for fp, g in S.groupby("source_file"):
        bars = ensure_datetime_index(pd.read_parquet(fp))
        cols = {c.lower(): c for c in bars.columns}
        close = cols.get("close") or list(bars.columns)[-1]
        c = bars[close]
        idx = bars.index

        # map each sample start index
        start_idx = (g["idx_in_shard"] + g["window"] - 1).astype(int).to_numpy()
        start_idx = np.clip(start_idx, 1, len(idx)-1)

        # simple realized vol (rolling stdev of 1-lag returns over vol_min)
        r = c.pct_change().fillna(0.0)
        vol = r.rolling(args.vol_min, min_periods=5).std().fillna(method="bfill").to_numpy()

        # short horizon return (horiz_min forward)
        t_h = pd.to_timedelta(args.horiz_min, unit="m")
        idx_np = idx.values

        def fwd_index(si):
            t_end = idx[si] + t_h
            j = int(np.searchsorted(idx_np, t_end.to_datetime64(), side="right")-1)
            return min(max(j, si), len(idx)-1)

        out_i = []
        for si in start_idx:
            fj = fwd_index(si)
            ret_m = float((c.iloc[fj]/c.iloc[si])-1.0)
            vol_m = float(vol[si])
            hod = int(pd.Timestamp(idx[si]).tz_convert("UTC").hour) if getattr(idx[si], 'tz', None) else int(pd.Timestamp(idx[si], tz='UTC').hour)
            out_i.append((ret_m, vol_m, hod))

        gi = g[["row_id"]].copy()
        gi["ret_m"] = [x[0] for x in out_i]
        gi["vol_m"] = [x[1] for x in out_i]
        gi["hour"]  = [x[2] for x in out_i]
        rows.append(gi)

    F = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    F.to_csv(args.out_csv, index=False)
    print("wrote features:", args.out_csv, "rows=", len(F))


if __name__ == "__main__":
    main()

