import argparse, glob, os, yaml, numpy as np, pandas as pd
from tqdm.auto import tqdm
from src.imaging.exact import robust_scale, make_gaf_gadf, make_mtf, make_rp, make_cwt
from src.imaging.calendar import calendar_maps

def sliding_windows(arr: np.ndarray, window: int, stride: int = 1):
    n = len(arr)
    for start in range(0, n - window + 1, stride):
        yield start, arr[start:start+window]

def build_stack(bars: pd.DataFrame, window: int, image_size: int, cfg):
    price = bars["close"].to_numpy(dtype=float)
    qty = bars.get("qty", pd.Series(np.ones(len(bars)))).to_numpy(dtype=float)
    stride = int(cfg.get("stride", 1))
    total_windows = max(0, (len(price) - window) // stride + 1)
    num_calendar_channels = 4 if cfg.get("calendar_embeddings", True) else 0
    num_channels = 6 + num_calendar_channels
    if total_windows <= 0:
        return np.empty((0, num_channels, image_size, image_size), dtype=np.float32), np.empty(0, dtype=np.int64)

    X = []; idx_map = []
    window_iter = tqdm(sliding_windows(price, window, stride),
                       total=total_windows,
                       desc=f"Windows w={window}",
                       unit="window",
                       leave=False)
    for start, win in window_iter:
        p = robust_scale(win)
        gasf, gadf = make_gaf_gadf(p, image_size)
        mtf = make_mtf(p, image_size, n_bins=8)
        rp = make_rp(p, image_size)
        cwt = make_cwt(p, image_size)
        q = robust_scale(qty[start:start+window])
        q_gasf, _ = make_gaf_gadf(q, image_size)
        channels = [gasf, gadf, mtf, rp, cwt, q_gasf]
        if cfg.get("calendar_embeddings", True):
            cal = calendar_maps(bars["timestamp"].iloc[:start+window], image_size)
            channels.extend(cal)
        img = np.stack(channels, axis=0).astype(np.float32)
        X.append(img); idx_map.append(start)
        window_iter.set_postfix({"images": len(X)})
    return np.stack(X, axis=0), np.array(idx_map)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/imaging.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out_dir, exist_ok=True)
    windows = cfg.get("windows",[64])
    image_size = int(cfg.get("image_size",64))

    bar_files = sorted(glob.glob(args.bars_glob))
    if not bar_files:
        tqdm.write("No bar files found.")
        return

    for p in tqdm(bar_files, desc="Building image stacks", unit="file"):
        bars = pd.read_parquet(p)
        if "timestamp" not in bars.columns:
            bars["timestamp"] = pd.to_datetime(pd.RangeIndex(start=0, stop=len(bars), step=1), unit="s", utc=True)
        base = os.path.basename(p).replace(".parquet","")
        for w in tqdm(windows, desc=f"{base} windows", unit="window", leave=False):
            X, idx = build_stack(bars, int(w), image_size, cfg)
            np.save(os.path.join(args.out_dir, f"{base}_w{w}_images.npy"), X)
            np.save(os.path.join(args.out_dir, f"{base}_w{w}_index.npy"), idx)
            tqdm.write(f"Wrote {base} window {w} -> {X.shape}")

if __name__ == "__main__":
    main()
