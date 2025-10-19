import argparse, pandas as pd, numpy as np
from src.utils.progress import pbar
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot_labeled", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--event_end_minutes", type=int, default=None,
                    help="If event_end_ts is absent, derive as t0 + this many minutes")
    ap.add_argument("--progress", action="store_true", help="Show tqdm progress bars")
    args = ap.parse_args()
    # Load CSV and normalize timestamps to UTC
    df = pd.read_csv(args.snapshot_labeled, low_memory=False)
    for col in ("t0", "t1", "event_end_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")

    # Derive event_end_ts if missing and minutes provided
    if "event_end_ts" not in df.columns or df["event_end_ts"].isna().all():
        if args.event_end_minutes is None:
            raise SystemExit("event_end_ts missing and --event_end_minutes not provided")
        if "t0" not in df.columns:
            raise SystemExit("Cannot derive event_end_ts: t0 missing in CSV")
        df["event_end_ts"] = df["t0"] + pd.to_timedelta(int(args.event_end_minutes), unit="m")

    # Guard against rows with unparseable timestamps
    if "t0" in df.columns and "event_end_ts" in df.columns:
        bad = df["t0"].isna() | df["event_end_ts"].isna()
        if bad.any():
            df = df[~bad].copy()
            df.reset_index(drop=True, inplace=True)
    w = np.ones(len(df), dtype=float)
    bar = pbar(total=len(df), desc="uniqueness") if args.progress else None
    groups = df.groupby("symbol", sort=False) if "symbol" in df.columns else [("_all", df)]
    for sym, D in groups:
        # Use int64 nanoseconds for vectorized math (tz-aware safe)
        starts = D["t0"].view("int64").to_numpy()
        ends   = D["event_end_ts"].view("int64").to_numpy()
        times = np.unique(np.sort(np.r_[starts, ends]))
        if len(times)==0: continue
        idx = {t:i for i,t in enumerate(times)}
        s = np.array([idx[t] for t in starts]); e = np.array([idx[t] for t in ends])
        diff = np.zeros(len(times)+1, dtype=int)
        for a,b in zip(s,e): diff[a]+=1; diff[b]-=1
        conc = diff.cumsum()[:-1]
        ww = [1.0/max(1.0, float(conc[a:b].mean())) if b>a else 1.0/max(1.0, float(conc[a])) for a,b in zip(s,e)]
        w[D.index.values] = np.array(ww)
        if bar:
            bar.update(len(D))
    df["uniqueness"] = w
    df.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv, "rows=", len(df))
    if bar:
        bar.close()
if __name__ == "__main__":
    main()
