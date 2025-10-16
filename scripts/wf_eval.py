import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os, yaml, json, numpy as np, pandas as pd
from pathlib import Path
from src.sim.pnl_sim import probs_to_trades, pnl_from_positions
from tqdm.auto import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wf.yaml")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--outputs_dir", default="outputs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.outputs_dir); out_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=4, desc="Walk-forward evaluation", unit="step")

    rng = np.random.default_rng(2025)
    N = 1200
    logits = rng.normal(size=(N,3))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    prices = 10000 * (1 + 0.0005 * np.cumsum(rng.normal(size=N))).clip(min=1)
    progress.update()

    positions = probs_to_trades(probs, threshold=float(cfg["thresholding"]["min_confidence"]))
    pnl_df = pnl_from_positions(
        positions, prices,
        fee_bps=float(cfg["cost_model"]["fee_bps"]),
        slippage_bps=float(cfg["cost_model"]["slippage_bps"]),
    )
    progress.update()
    report = {
        "folds": len(cfg["folds"]),
        "coverage_pct": float((positions!=0).mean()*100.0),
        "final_equity": float(pnl_df["equity"].iloc[-1]),
        "max_drawdown_pct": float(pnl_df["drawdown"].max()*100.0),
    }
    (out_dir / "reports").mkdir(exist_ok=True, parents=True)
    with open(out_dir / "reports" / "wf_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    pnl_df.to_csv(out_dir / "reports" / "wf_equity.csv", index=False)
    progress.update()

    md = f"""# Walk-Forward OOS Summary (Scaffold)

- Folds: {len(cfg["folds"])}
- Coverage: {report["coverage_pct"]:.2f}%
- Final Equity: {report["final_equity"]:.4f}
- Max Drawdown: {report["max_drawdown_pct"]:.2f}%

> This scaffold uses simulated probabilities until you run real inference.
"""
    with open(out_dir / "reports" / "wf_report.md", "w") as f:
        f.write(md)

    progress.update()
    progress.close()

    tqdm.write("Wrote scaffold walk-forward report to outputs/reports/.")

if __name__ == "__main__":
    main()
