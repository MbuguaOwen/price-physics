import argparse, pandas as pd, numpy as np
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot_labeled", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--purge_time", default="1h")
    ap.add_argument("--embargo_time", default="30m")
    args = ap.parse_args()
    df = pd.read_csv(args.snapshot_labeled, parse_dates=["t0","t1","event_end_ts"], low_memory=False)
    df = df.sort_values("t1").reset_index(drop=True)
    T = df["t1"]; cuts = np.linspace(0, len(T), args.k+1, dtype=int)
    parts = []
    for f in range(args.k):
        lo, hi = cuts[f], cuts[f+1]
        t_lo, t_hi = T.iloc[lo], T.iloc[hi-1]
        purge_start = t_lo - pd.Timedelta(args.purge_time)
        embargo_end = t_hi + pd.Timedelta(args.embargo_time)
        inter = (df["t0"] <= embargo_end) & (df["event_end_ts"] >= purge_start)
        is_test  = (df["t1"] >= t_lo) & (df["t1"] <= t_hi)
        is_train = (~is_test) & (~inter)
        p = df.loc[:, ["t0","t1","event_end_ts"]].copy()
        p["fold"] = f
        p["split"] = np.where(is_test, "test", np.where(is_train, "train", "drop"))
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print("Wrote CV folds:", args.out_csv, "rows=", len(out))
if __name__ == "__main__":
    main()

