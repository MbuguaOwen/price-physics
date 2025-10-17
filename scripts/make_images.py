import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, glob, os, yaml, numpy as np, pandas as pd
import warnings
warnings.filterwarnings('ignore', message='Some quantiles are equal.*', module='pyts.preprocessing.discretizer')
from tqdm.auto import tqdm
from src.imaging.exact import robust_scale, make_gaf_gadf, make_mtf, make_rp, make_cwt
from src.imaging.calendar import calendar_maps
import csv
from pathlib import Path
try:
    from PIL import Image  # optional, used only for previews
    _PIL_OK = True
except Exception:
    _PIL_OK = False
from numpy.lib.format import open_memmap

def sliding_windows(arr: np.ndarray, window: int, stride: int = 1):
    n = len(arr)
    for start in range(0, n - window + 1, stride):
        yield start, arr[start:start+window]

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def get_run_tag(bars_path: str) -> str:
    base = os.path.splitext(os.path.basename(bars_path))[0]
    # Extract SYMBOL and YYYY-MM robustly (e.g., BTCUSDT_2025-01_*.parquet)
    import re as _re
    dm = _re.search(r"(20\d{2}-\d{2})", base)
    if dm:
        ym = dm.group(1)
        prefix = base[: dm.start()].rstrip("_- ")
        # take the last token of the prefix as symbol
        toks = _re.split(r"[_\-]+", prefix)
        sym = toks[-1].upper() if toks else prefix.upper()
        if sym:
            return f"{sym}_{ym}"
    return base


def _open_manifest(man_path: str):
    write_header = not os.path.exists(man_path)
    f = open(man_path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if write_header:
        w.writerow(["relpath","idx_in_shard","source_file","symbol","month","t0","t1","window","stride","C","H","W","transform"])
    return f, w


def _save_png_previews(batch: np.ndarray, out_dir: str, run_tag: str, shard_idx: int) -> None:
    if not _PIL_OK:
        return
    try:
        prev_dir = os.path.join(out_dir, run_tag, "previews")
        ensure_dir(prev_dir)
        k = min(16, batch.shape[0])
        for i in range(k):
            arr = batch[i]
            arr = np.clip(arr, 0, 1)
            arr = np.moveaxis(arr, 0, -1)
            if arr.shape[2] == 1:
                img = Image.fromarray((arr[:, :, 0] * 255.0).round().astype(np.uint8), mode="L")
            else:
                img = Image.fromarray((arr * 255.0).round().astype(np.uint8))
            img.save(os.path.join(prev_dir, f"shard{shard_idx:04d}_sample{i:02d}.png"))
    except Exception:
        pass


def flush_npy_shard(shard_idx: int, imgs: list, metas: list, out_root: str, run_tag: str, source_file: str, symbol: str, month: str, save_png_previews: bool):
    if not imgs:
        return 0
    batch = np.stack(imgs, axis=0).astype(np.float32, copy=False)
    shard_rel = os.path.join(run_tag, "shards", f"images_{shard_idx:04d}.npy")
    shard_full = os.path.join(out_root, shard_rel)
    ensure_dir(os.path.dirname(shard_full))
    np.save(shard_full, batch)
    if save_png_previews:
        _save_png_previews(batch, out_root, run_tag, shard_idx)

    man_path = os.path.join(out_root, run_tag, "manifest.csv")
    mf, writer = _open_manifest(man_path)
    try:
        for i, m in enumerate(metas):
            writer.writerow([
                shard_rel.replace("\\", "/"),
                i,
                source_file,
                symbol,
                month,
                m["t0"],
                m["t1"],
                m["window"],
                m["stride"],
                m["C"],
                m["H"],
                m["W"],
                m["transform"],
            ])
    finally:
        mf.close()
    count = len(imgs)
    imgs.clear(); metas.clear()
    return count


def build_stack_streaming(bars: pd.DataFrame, window: int, image_size: int, cfg, out_root: str, run_tag: str, batch_size: int = 2048, save_png_previews: bool = False) -> dict:
    skipped = 0
    written = 0
    # Continue shard indexing if prior shards exist (avoid overwrite across windows/reruns)
    shards_dir = os.path.join(out_root, run_tag, "shards")
    try:
        existing = sorted(glob.glob(os.path.join(shards_dir, "images_*.npy")))
        if existing:
            import re as _re
            indices = []
            for fp in existing:
                m = _re.search(r"images_(\d{4})\.npy$", os.path.basename(fp))
                if m:
                    indices.append(int(m.group(1)))
            shard_idx = (max(indices) + 1) if indices else 0
        else:
            shard_idx = 0
    except Exception:
        shard_idx = 0
    imgs: list = []
    metas: list = []

    price = bars["close"].to_numpy(dtype=float)
    qty = bars.get("qty", pd.Series(np.ones(len(bars)))).to_numpy(dtype=float)
    ts_all = pd.to_datetime(bars['timestamp'], utc=True)
    stride = int(cfg.get("stride", 1))
    total_windows = max(0, (len(price) - window) // stride + 1)
    if total_windows <= 0:
        return {"written": 0, "skipped": 0, "shards": 0}

    # Derive symbol and month from run_tag if possible
    parts = run_tag.split("_")
    symbol = parts[0] if len(parts) >= 1 else run_tag
    month = parts[1] if len(parts) >= 2 else ""

    # Transform composition string for manifest
    base_transforms = ["gasf", "gadf", "mtf", "qty_gasf"]
    if cfg.get('enable_rp', False):
        base_transforms.append("rp")
    if cfg.get('enable_cwt', False):
        base_transforms.append("cwt")
    if cfg.get("calendar_embeddings", True):
        base_transforms.append("calendar")
    trans_str = ",".join(base_transforms)

    window_iter = tqdm(sliding_windows(price, window, stride),
                       total=total_windows,
                       desc=f"Windows w={window}",
                       unit="window",
                       leave=False)
    for start, win in window_iter:
        p = robust_scale(win)
        if len(p) < 1 or not np.all(np.isfinite(p)):
            skipped += 1
            continue
        s_eff = min(image_size, len(p))
        gasf, gadf = make_gaf_gadf(p, s_eff)
        mtf_bins = int(cfg.get("mtf_bins", 8))
        mtf = make_mtf(p, s_eff, n_bins=mtf_bins)
        rp = make_rp(p, s_eff) if cfg.get('enable_rp', False) else None
        cwt = make_cwt(p, s_eff) if cfg.get('enable_cwt', False) else None
        q = robust_scale(qty[start:start+window])
        q_gasf, _ = make_gaf_gadf(q, s_eff)
        channels = [gasf, gadf, mtf, rp, cwt, q_gasf]
        if cfg.get("calendar_embeddings", True):
            cal = calendar_maps(ts_all.iloc[:start+window], s_eff)
            channels.extend(cal)
        channels = [ch for ch in channels if ch is not None]
        img = np.stack(channels, axis=0).astype(np.float32)

        # Meta
        t0 = ts_all.iloc[start]
        t1 = ts_all.iloc[min(start + window - 1, len(ts_all) - 1)]
        meta = {
            "t0": str(t0),
            "t1": str(t1),
            "window": int(window),
            "stride": int(stride),
            "C": int(img.shape[0]),
            "H": int(img.shape[1]),
            "W": int(img.shape[2]),
            "transform": trans_str,
        }

        imgs.append(img)
        metas.append(meta)

        if len(imgs) >= int(batch_size):
            written += flush_npy_shard(shard_idx, imgs, metas, out_root, run_tag, str(Path(cfg.get('source_file', '')) or ''), symbol, month, save_png_previews)
            shard_idx += 1
            window_iter.set_postfix({"written": written, "skipped": skipped})

    if imgs:
        written += flush_npy_shard(shard_idx, imgs, metas, out_root, run_tag, str(Path(cfg.get('source_file', '')) or ''), symbol, month, save_png_previews)
        shard_idx += 1

    return {"written": int(written), "skipped": int(skipped), "shards": int(shard_idx)}


def build_stack_memmap(bars: pd.DataFrame, window: int, image_size: int, cfg, out_dir: str, base: str):
    skipped = 0
    price = bars["close"].to_numpy(dtype=float)
    qty = bars.get("qty", pd.Series(np.ones(len(bars)))).to_numpy(dtype=float)
    ts_all = pd.to_datetime(bars['timestamp'], utc=True)
    stride = int(cfg.get("stride", 1))
    total_windows = max(0, (len(price) - window) // stride + 1)
    if total_windows <= 0:
        images_path = os.path.join(out_dir, f"{base}_w{int(window)}_images.npy")
        index_path = os.path.join(out_dir, f"{base}_w{int(window)}_index.npy")
        np.save(images_path, np.empty((0, 0, image_size, image_size), dtype=np.float32))
        np.save(index_path, np.empty((0,), dtype=np.int64))
        return 0, 0, image_size, 0

    mm = None
    mm_idx = None
    write_i = 0
    window_iter = tqdm(sliding_windows(price, window, stride),
                       total=total_windows,
                       desc=f"Windows w={window}",
                       unit="window",
                       leave=False)
    for start, win in window_iter:
        p = robust_scale(win)
        if len(p) < 1:
            skipped += 1
            continue
        s_eff = min(image_size, len(p))
        gasf, gadf = make_gaf_gadf(p, s_eff)
        mtf_bins = min(int(cfg.get("mtf_bins", 8)), max(2, s_eff))
        cwt_scales = int(cfg.get("cwt_scales", min(32, s_eff)))
        mtf = make_mtf(p, s_eff, n_bins=mtf_bins)
        rp = make_rp(p, s_eff) if cfg.get('enable_rp', False) else None
        cwt = make_cwt(p, s_eff, scales=cwt_scales) if cfg.get('enable_cwt', False) else None
        q = robust_scale(qty[start:start+window])
        q_gasf, _ = make_gaf_gadf(q, s_eff)
        channels = [gasf, gadf, mtf, rp, cwt, q_gasf]
        if cfg.get("calendar_embeddings", True):
            cal = calendar_maps(ts_all.iloc[:start+window], s_eff)
            channels.extend(cal)
        channels = [ch for ch in channels if ch is not None]
        img = np.stack(channels, axis=0).astype(np.float32)

        if mm is None:
            images_path = os.path.join(out_dir, f"{base}_w{int(window)}_images.npy")
            index_path = os.path.join(out_dir, f"{base}_w{int(window)}_index.npy")
            mm = open_memmap(images_path, mode='w+', dtype=np.float32,
                             shape=(total_windows, img.shape[0], s_eff, s_eff))
            mm_idx = open_memmap(index_path, mode='w+', dtype=np.int64, shape=(total_windows,))

        mm[write_i] = img
        mm_idx[write_i] = start
        write_i += 1
        window_iter.set_postfix({"written": write_i, "skipped": skipped})

    # If any windows were skipped, shrink files to the actual count by rewriting headers
    if write_i != total_windows and mm is not None:
        images_path = mm.filename
        index_path = mm_idx.filename
        del mm; del mm_idx
        mm_small = open_memmap(images_path, mode='w+', dtype=np.float32,
                               shape=(write_i, img.shape[0], s_eff, s_eff))
        mm_idx_small = open_memmap(index_path, mode='w+', dtype=np.int64, shape=(write_i,))
        del mm_small; del mm_idx_small

    return write_i, (img.shape[0] if 'img' in locals() else 0), s_eff, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/imaging.yaml")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--save_png_previews", action="store_true", default=False)
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
        # For manifest metadata
        cfg["source_file"] = os.path.abspath(p)
        run_tag = get_run_tag(p)
        ensure_dir(os.path.join(args.out_dir, run_tag, "shards"))
        for w in tqdm(windows, desc=f"{run_tag} windows", unit="window", leave=False):
            stats = build_stack_streaming(bars, int(w), image_size, cfg, args.out_dir, run_tag, args.batch_size, args.save_png_previews)
            tqdm.write(f"{run_tag} w={int(w)} -> written={stats['written']} skipped={stats['skipped']} shards={stats['shards']}")

if __name__ == "__main__":
    main()
