from __future__ import annotations
import argparse, json, numpy as np, pandas as pd
from pathlib import Path


def load_preds(path):
    df = pd.read_parquet(path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if prob_cols:
        P = df[prob_cols].values.astype("float64")
    else:
        logit_cols = [c for c in df.columns if c.startswith("logits_")]
        L = df[logit_cols].values.astype("float64")
        ex = np.exp(L - L.max(axis=1, keepdims=True))
        P = ex / ex.sum(axis=1, keepdims=True)
    y = df["label"].values
    df = df.assign(p_pos=P[:, -1], p_neg=P[:, 0], conf=P.max(axis=1))
    return df


def pnl_metric(df, th=0.5, conf_tau=0.0, fee_bp=1.0):
    act = (df["p_pos"] >= th).astype(int) - (df["p_neg"] >= th).astype(int)
    mask = df["conf"] >= conf_tau
    act = act[mask]; y = df["label"][mask]
    pnl = ((act == y) * 1.0 + (act != y) * -1.0) - fee_bp / 10000.0
    if len(pnl) == 0:
        return -np.inf, 0.0, 0
    sharpe = pnl.mean() / (pnl.std(ddof=1) + 1e-9) * np.sqrt(252 * 24 * 60)
    return sharpe, pnl.mean(), len(pnl)


def sweep(path, out_json):
    df = load_preds(path)
    best = {"sharpe": -1e9}
    for th in np.linspace(0.5, 0.9, 17):
        for conf in np.linspace(0.0, 0.9, 19):
            s, mu, n = pnl_metric(df, th, conf, fee_bp=1.0)
            if s > best["sharpe"]:
                best.update({"th": float(th), "conf": float(conf), "sharpe": float(s), "mean_pnl": float(mu), "n_trades": int(n)})
    Path(out_json).write_text(json.dumps(best, indent=2))
    print("Best:", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_parquet", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    sweep(args.preds_parquet, args.out_json)

