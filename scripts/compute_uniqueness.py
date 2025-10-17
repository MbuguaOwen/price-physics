import argparse, pandas as pd, numpy as np
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot_labeled", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.snapshot_labeled, parse_dates=["t0","t1","event_end_ts"], low_memory=False)
    w = np.ones(len(df), dtype=float)
    for sym, D in df.groupby("symbol", sort=False):
        starts = D["t0"].values.astype("datetime64[ns]")
        ends   = D["event_end_ts"].values.astype("datetime64[ns]")
        times = np.unique(np.sort(np.r_[starts, ends]))
        if len(times)==0: continue
        idx = {t:i for i,t in enumerate(times)}
        s = np.array([idx[t] for t in starts]); e = np.array([idx[t] for t in ends])
        diff = np.zeros(len(times)+1, dtype=int)
        for a,b in zip(s,e): diff[a]+=1; diff[b]-=1
        conc = diff.cumsum()[:-1]
        ww = [1.0/max(1.0, float(conc[a:b].mean())) if b>a else 1.0/max(1.0, float(conc[a])) for a,b in zip(s,e)]
        w[D.index.values] = np.array(ww)
    df["uniqueness"] = w
    df.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv, "rows=", len(df))
if __name__ == "__main__":
    main()

