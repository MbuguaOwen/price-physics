import argparse, json
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV with binary-head preds: row_id, fold, y_true_bin, p_tp")
    ap.add_argument("--labels_csv", required=True, help="Snapshot labeled CSV with row_id and month")
    ap.add_argument("--month_to_regime_json", type=str, default=None, help="Optional JSON map month->regime")
    ap.add_argument("--win", type=float, default=50.50335478971694)
    ap.add_argument("--loss", type=float, default=18.404720361751057)
    ap.add_argument("--min_trades", type=int, default=5000)
    args = ap.parse_args()

    P = pd.read_csv(args.preds)
    if "row_id" not in P.columns:
        raise SystemExit("preds CSV must contain 'row_id'")
    # robust y_true detection
    if "y_true_bin" in P.columns:
        y_col = "y_true_bin"
    elif "y_true" in P.columns:
        # treat class 1 as TP
        y_col = "y_true"
        P[y_col] = (P[y_col].astype(int) == 1).astype(int)
    else:
        raise SystemExit("preds CSV must contain 'y_true_bin' or 'y_true'")
    if "p_tp" not in P.columns:
        # If 3-class preds are provided (p1 column), try to infer p_tp = p1
        p1 = P[[c for c in P.columns if c.lower() in ("p1", "p_tp")] ]
        if not p1.empty:
            P["p_tp"] = p1.iloc[:, 0]
        else:
            raise SystemExit("preds CSV must have 'p_tp' or a 'p1' column for TP probability")

    L = pd.read_csv(args.labels_csv)
    if "row_id" not in L.columns:
        L = L.reset_index(drop=True)
        L.insert(0, "row_id", L.index.astype("int64"))
    if "month" not in L.columns:
        raise SystemExit("labels_csv must contain 'month' column to map regimes")

    # Month -> regime mapping (optional)
    month_to_regime = None
    if args.month_to_regime_json:
        try:
            month_to_regime = json.load(open(args.month_to_regime_json, "r"))
        except Exception:
            month_to_regime = None

    # Join to bring 'month' and 'regime'
    J = P.merge(L[["row_id", "month"]], on="row_id", how="left")
    if month_to_regime:
        J["regime"] = J["month"].map(month_to_regime).fillna("unknown")
    else:
        J["regime"] = J["month"].astype(str)

    W = float(args.win); Ls = float(args.loss)

    def ev_from_prec(prec: float) -> float:
        return float(prec * W - (1.0 - prec) * Ls)

    out = {}
    for reg, G in J.groupby("regime"):
        p = np.asarray(G["p_tp"].clip(0, 1).to_numpy(), dtype=float)
        y = np.asarray(G[y_col].astype(int).to_numpy(), dtype=int)
        best = None
        for t in np.linspace(0.0, 1.0, 201):
            sel = p >= t
            n = int(sel.sum())
            if n < int(args.min_trades):
                continue
            prec = float(y[sel].mean()) if n > 0 else 0.0
            e = ev_from_prec(prec)
            if (best is None) or (e > best[2]):
                best = (float(t), prec, float(e), n)
        if best is None:
            out[reg] = {"thr": None, "precision": None, "EV_per_trade": None, "n_trades": int(len(G))}
        else:
            out[reg] = {"thr": round(best[0], 3), "precision": round(best[1], 4), "EV_per_trade": round(best[2], 3), "n_trades": int(best[3])}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

