import argparse, glob, os, yaml, pandas as pd, numpy as np
from tqdm.auto import tqdm
from src.labeling.triple_barrier import triple_barrier_labels, ewma_vol, atr_vol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/tbm.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(args.bars_glob))
    if not files:
        tqdm.write("No bar files found.")
        return

    for p in tqdm(files, desc="Labeling bars", unit="file"):
        df = pd.read_parquet(p)
        close = df["close"].astype(float)
        rets = close.pct_change().fillna(0.0)
        if cfg.get("vol_method","ewma") == "atr" and all(k in df.columns for k in ["high","low","close"]):
            vol = atr_vol(df, period=int(cfg.get("atr_period",14))) / df["close"]
        else:
            vol = ewma_vol(rets, span=int(cfg.get("ewma_span",32)))
        labels = triple_barrier_labels(close, vol, float(cfg["pt_mult"]), float(cfg["sl_mult"]), int(cfg["max_holding_bars"]))
        out = os.path.join(args.out_dir, os.path.basename(p).replace(".parquet","_labels.parquet"))
        labels.to_parquet(out, index=False)
        tqdm.write(f"Wrote {out} {len(labels)}")

if __name__ == "__main__":
    main()
