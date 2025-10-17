from __future__ import annotations
import argparse, json, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from pathlib import Path


def load_preds(preds_path):
    df = pd.read_parquet(preds_path)
    logit_cols = [c for c in df.columns if c.startswith("logits_")]
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if logit_cols:
        logits = torch.tensor(df[logit_cols].values, dtype=torch.float32)
    elif prob_cols:
        p = np.clip(df[prob_cols].values, 1e-6, 1 - 1e-6)
        if p.shape[1] == 2:
            logits = torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
        else:
            logits = torch.tensor(np.log(p), dtype=torch.float32)
    else:
        raise ValueError("No logits_* or prob_* columns found in preds parquet.")
    y = torch.tensor(df["label"].values, dtype=torch.long)
    return df, logits, y


def fit_temperature(logits, y, max_iter=500, lr=0.01):
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        l = F.cross_entropy(logits / T.clamp_min(1e-3), y)
        l.backward()
        return l

    opt.step(closure)
    return float(T.data.clamp_min(1e-3).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_parquet", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    _, logits, y = load_preds(args.preds_parquet)
    T = fit_temperature(logits, y)
    Path(args.out_json).write_text(json.dumps({"temperature": T}, indent=2))
    print(f"Fitted temperature: {T:.4f}")


if __name__ == "__main__":
    main()

