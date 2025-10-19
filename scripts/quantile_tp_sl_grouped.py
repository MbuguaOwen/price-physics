import argparse, json, os
import numpy as np, pandas as pd
from src.labeling.triple_barrier import ensure_datetime_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--horizon_minutes", type=int, required=True)
    ap.add_argument("--q_tp", type=float, default=0.60)
    ap.add_argument("--q_sl", type=float, default=0.40)
    ap.add_argument("--winsor", type=float, default=0.99)
    ap.add_argument("--min_group_n", type=int, default=5000)
    ap.add_argument("--tp_bounds", type=float, nargs=2, default=(15.0, 120.0))
    ap.add_argument("--sl_bounds", type=float, nargs=2, default=(8.0, 40.0))
    ap.add_argument("--uniqueness_csv", type=str, default=None,
                    help="Optional CSV with columns ['month','uniqueness'] to weight rows by mean uniqueness per month.")
    ap.add_argument("--month_to_regime_json", type=str, default=None,
                    help="Optional JSON mapping month->regime for grouping.")
    ap.add_argument("--group", choices=["month","regime"], default="month")
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--atr_window", type=int, default=14)
    ap.add_argument("--bars_root", default="data/bars_dollar")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    S = pd.read_csv(args.snapshot)
    S = S.reset_index(drop=True)

    def pick_cols(b):
        cols = {c.lower(): c for c in b.columns}
        c = cols.get("close") or list(b.columns)[-1]
        h = cols.get("high", c); l = cols.get("low", c)
        return b[c], b[h], b[l]

    def atr_wilder(c,h,l,n=14):
        prev = c.shift(1)
        tr = pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/n, adjust=False).mean()
    
    # Optional uniqueness weighting by month
    month_uni_w = None
    if args.uniqueness_csv and os.path.exists(args.uniqueness_csv):
        try:
            U = pd.read_csv(args.uniqueness_csv)
            if "month" in U.columns and "uniqueness" in U.columns:
                month_uni_w = U.groupby("month", as_index=True)["uniqueness"].mean().to_dict()
        except Exception:
            month_uni_w = None

    # Optional regime mapping
    month_to_regime = None
    if args.group == "regime" and args.month_to_regime_json and os.path.exists(args.month_to_regime_json):
        try:
            month_to_regime = json.load(open(args.month_to_regime_json, "r"))
        except Exception:
            month_to_regime = None

    rows=[]
    S_samp = S.sample(min(args.sample, len(S)), random_state=7)
    for fp, g in S_samp.groupby("source_file"):
        b = ensure_datetime_index(pd.read_parquet(fp))
        c,h,l = pick_cols(b)
        atr = atr_wilder(c,h,l,args.atr_window)
        idx = b.index.values
        for s,w,m in zip(g["idx_in_shard"].astype(int), g["window"].astype(int), g["month"]):
            start = int(s+w-1)
            if start>=len(b)-1: continue
            t_end = pd.to_datetime(b.index[start]) + pd.Timedelta(minutes=args.horizon_minutes)
            end = int(np.searchsorted(idx, t_end.to_datetime64(), side="right")-1)
            end = min(max(end, start+1), len(b)-1)
            a0 = float(atr.iloc[start]); c0 = float(c.iloc[start])
            if not np.isfinite(a0) or a0<=0: continue
            seg_h = float(h.iloc[start+1:end+1].max())
            seg_l = float(l.iloc[start+1:end+1].min())
            mfe = (seg_h - c0)/a0
            mae = (c0 - seg_l)/a0
            grp = m if args.group == "month" else (month_to_regime.get(m, "unknown") if month_to_regime else "unknown")
            rows.append((grp, m, mfe, mae))
    D = pd.DataFrame(rows, columns=["group","month","MFE_ATR","MAE_ATR"])

    # Winsorize globally at specified tails
    if len(D):
        p_low = max(0.0, 1.0 - float(args.winsor))
        p_high = min(1.0, float(args.winsor))
        if p_low > 0 or p_high < 1:
            lo_mfe, hi_mfe = D["MFE_ATR"].quantile([p_low, p_high]).to_numpy()
            lo_mae, hi_mae = D["MAE_ATR"].quantile([p_low, p_high]).to_numpy()
            D["MFE_ATR"] = D["MFE_ATR"].clip(lower=lo_mfe, upper=hi_mfe)
            D["MAE_ATR"] = D["MAE_ATR"].clip(lower=lo_mae, upper=hi_mae)

    # Weights: per-row equals mean uniqueness of its month if provided
    if month_uni_w:
        D["w"] = D["month"].map(month_uni_w).fillna(1.0).astype(float)
    else:
        D["w"] = 1.0

    def wquantile(x: np.ndarray, q: float, w: np.ndarray | None):
        x = np.asarray(x, dtype=float)
        if w is None:
            return float(np.quantile(x, q))
        w = np.asarray(w, dtype=float)
        if len(x) == 0:
            return float("nan")
        order = np.argsort(x)
        x_sorted = x[order]
        w_sorted = w[order]
        cw = np.cumsum(w_sorted)
        if cw[-1] <= 0 or not np.isfinite(cw[-1]):
            return float(np.quantile(x, q))
        t = q * cw[-1]
        return float(np.interp(t, cw, x_sorted))

    # Global fallback quantiles
    tp_global = wquantile(D["MFE_ATR"].to_numpy(), float(args.q_tp), D["w"].to_numpy()) if len(D) else 0.0
    sl_global = wquantile(D["MAE_ATR"].to_numpy(), float(args.q_sl), D["w"].to_numpy()) if len(D) else 0.0

    # Grouped quantiles (with fallback and clamping)
    out = {}
    for g, G in D.groupby("group"):
        n = len(G)
        if n < int(args.min_group_n):
            tp = tp_global
            sl = sl_global
        else:
            tp = wquantile(G["MFE_ATR"].to_numpy(), float(args.q_tp), G["w"].to_numpy())
            sl = wquantile(G["MAE_ATR"].to_numpy(), float(args.q_sl), G["w"].to_numpy())
        tp = float(np.clip(tp, args.tp_bounds[0], args.tp_bounds[1]))
        sl = float(np.clip(sl, args.sl_bounds[0], args.sl_bounds[1]))
        out[g] = {"tp_mult": tp, "sl_mult": sl}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote:", args.out_json, "keys:", list(out)[:5], "...")


if __name__ == "__main__":
    main()
