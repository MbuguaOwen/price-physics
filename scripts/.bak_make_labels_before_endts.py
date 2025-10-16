import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows PowerShell runbook (examples):
# $env:PYTHONPATH = "$PWD"
#
# python -m scripts.make_labels `
#   --bars_glob "data/bars_dollar/*2025-0[1-7]*.parquet" `
#   --out_dir "data/labels_jan_jul_atr_h240" `
#   --config "configs/tbm.yaml"
#
# Sensitivity:
# python -m scripts.make_labels `
#   --bars_glob "data/bars_dollar/*2025-0[1-7]*.parquet" `
#   --out_dir "data/labels_jan_jul_atr_h360" `
#   --config "configs/tbm.yaml"
# python -m scripts.make_labels `
#   --bars_glob "data/bars_dollar/*2025-0[1-7]*.parquet" `
#   --out_dir "data/labels_jan_jul_atr_h480" `
#   --config "configs/tbm.yaml"

import argparse, glob, json, os
from datetime import datetime, timezone
import yaml, pandas as pd, numpy as np
from tqdm.auto import tqdm

from src.labeling.triple_barrier import (
    validate_labeling_cfg,
    label_bars_with_diagnostics,
    triple_barrier_labels_t1,  # kept for backward compatibility
    triple_barrier_labels_ohlc_t1,  # kept for backward compatibility
    ewma_vol,
    atr_vol,
)


def add_end_ts_from_df(df, labels_df, cfg):
    """Augment labels_df with end_ts (first PT/SL hit or vertical cutoff) and time_to_hit_sec.
    Assumptions:
      - ATR mode (vol_method='atr') with pt_mult/sl_mult in price units of ATR
      - df has columns: 'open','high','low','close' and a UTC DatetimeIndex or 'timestamp' column
    """
    import pandas as pd, numpy as np

    # Resolve timestamps aligned to df
    ts = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get("timestamp"), utc=True)
    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.to_datetime(ts, utc=True)

    high = df["high"].to_numpy()
    low  = df["low"].to_numpy()
    close = df["close"].to_numpy()

    # Wilder ATR (EWMA) with default 14 unless provided
    period = int(cfg.get("atr_period", 14))
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().to_numpy()

    pt_mult = float(cfg.get("pt_mult", 100.0))
    sl_mult = float(cfg.get("sl_mult", 20.0))
    horizon = cfg.get("horizon", {"type":"clock","minutes":30})
    assert (horizon or {}).get("type","clock") == "clock", "This helper assumes clock-time vertical horizon."
    H = int((horizon or {}).get("minutes", 30))
    same_rule = cfg.get("same_bar_resolution","sl_first")

    # Prepare result arrays
    n = len(df)
    end_ts = np.empty(n, dtype="datetime64[ns]")
    end_ts[:] = np.datetime64("NaT")
    t_hit = np.full(n, np.nan)
    reason = np.full(n, "", dtype=object)

    # We assume labels_df index aligns to df's index (same events)
    lab = labels_df
    lab_index = lab.index.to_numpy()

    for i in lab_index:
        if i < 0 or i >= n:
            continue
        p0 = close[i]
        up = p0 + pt_mult * atr[i]
        dn = p0 - sl_mult * atr[i]
        t_vert = ts[i] + pd.Timedelta(minutes=H)

        hit_side = 0
        hit_time = t_vert
        # scan forward until vertical barrier
        j = i + 1
        while j < n and ts[j] < t_vert:
            hit_up = high[j] >= up
            hit_dn = low[j]  <= dn
            if hit_up and hit_dn:
                if same_rule == "pt_first":
                    hit_side = +1; hit_time = ts[j]; reason[i] = "pt_sl_same_bar_pt_first"
                elif same_rule == "sl_first":
                    hit_side = -1; hit_time = ts[j]; reason[i] = "pt_sl_same_bar_sl_first"
                else:
                    hit_side = 0;   hit_time = t_vert; reason[i] = "both_hit_neutral"
                break
            if hit_up:
                hit_side = +1; hit_time = ts[j]; reason[i] = "pt"; break
            if hit_dn:
                hit_side = -1; hit_time = ts[j]; reason[i] = "sl"; break
            j += 1

        end_ts[i] = np.datetime64(hit_time.value)
        t_hit[i] = (hit_time - ts[i]).total_seconds()

    out = lab.copy()
    # Ensure columns exist even if some entries remained NaT
    out["end_ts"] = pd.to_datetime(end_ts)
    out["time_to_hit_sec"] = t_hit
    if "event_end_reason" not in out.columns:
        out["event_end_reason"] = ""
        # Fill per-index reasons where known
        for i in lab_index:
            if i < len(reason) and isinstance(reason[i], str) and reason[i]:
                out.at[i, "event_end_reason"] = reason[i]
            elif pd.isna(out.at[i, "time_to_hit_sec"]):
                out.at[i, "event_end_reason"] = "vertical"
    return out

def to_bool(x, default=False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return bool(default)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def time_window_vol(returns: pd.Series, window_minutes: int) -> pd.Series:
    v = returns.rolling(f"{int(window_minutes)}T").std()
    return v.shift(1).bfill().fillna(returns.std())


from pandas.api.types import is_integer_dtype, is_datetime64_any_dtype


def to_datetime_utc(ts: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(ts):
        t = pd.to_datetime(ts, utc=True, errors="coerce")
        if t.dt.tz is None:
            t = t.dt.tz_localize("UTC")
        else:
            t = t.dt.tz_convert("UTC")
        return t
    if is_integer_dtype(ts):
        vmax = int(pd.to_numeric(ts, errors="coerce").max() or 0)
        unit = "ns" if vmax > 10 ** 14 else ("ms" if vmax > 10 ** 12 else "s")
        return pd.to_datetime(ts, unit=unit, utc=True)
    return pd.to_datetime(ts, utc=True, errors="coerce")


def compute_t1_clock(ts: pd.Series, minutes: int) -> np.ndarray:
    # Convert to UTC datetimes; accept int epochs or datetime-like
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if not t.is_monotonic_increasing:
        raise ValueError("Bars timestamp is not strictly increasing. Fix the input ordering before labeling.")
    tgt = t + pd.to_timedelta(minutes, unit="m")
    # Use .astype('int64') on datetime64 arrays instead of .view('int64') to avoid FutureWarning
    a = t.to_numpy(dtype="datetime64[ns]").astype("int64")
    b = tgt.to_numpy(dtype="datetime64[ns]").astype("int64")
    j = np.searchsorted(a, b, side="left")
    j = np.clip(j, 0, len(t) - 1).astype(int)
    return j


def _parse_ts(series: pd.Series) -> pd.Series:
    s = series
    if is_datetime64_any_dtype(s):
        t = pd.to_datetime(s, utc=True, errors="coerce")
        if t.dt.tz is None:
            t = t.dt.tz_localize("UTC")
        else:
            t = t.dt.tz_convert("UTC")
        return t
    if is_integer_dtype(s):
        vmax = int(pd.to_numeric(s, errors="coerce").max() or 0)
        unit = "ns" if vmax > 10 ** 14 else ("ms" if vmax > 10 ** 12 else "s")
        return pd.to_datetime(s, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True, errors="coerce")


def build_timebars_from_ticks(tick_paths, freq: str = "1T") -> pd.DataFrame:
    frames = []
    for tp in tick_paths:
        try:
            df = pd.read_csv(tp)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        ts_col = cols.get("timestamp") or cols.get("ts") or cols.get("time") or list(df.columns)[0]
        price_col = cols.get("price") or cols.get("p") or cols.get("close") or list(df.columns)[1]
        ts = _parse_ts(df[ts_col]).rename("timestamp")
        pr = pd.to_numeric(df[price_col], errors="coerce")
        x = pd.DataFrame({"timestamp": ts, "price": pr}).dropna()
        x = x.sort_values("timestamp").set_index("timestamp")
        g = x.resample(freq)
        tb = pd.DataFrame(
            {
                "open": g["price"].first(),
                "high": g["price"].max(),
                "low": g["price"].min(),
                "close": g["price"].last(),
            }
        )
        frames.append(tb)
    if not frames:
        raise RuntimeError("No ticks available to build timebars.")
    tb_all = pd.concat(frames).sort_index()
    tb = tb_all.resample(freq).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    tb = tb.dropna(subset=["close"])  # require close for returns
    return tb


def realized_vol_from_timebars(tb: pd.DataFrame, window_minutes: int = 30) -> pd.Series:
    rets = tb["close"].astype(float).pct_change().fillna(0.0)
    vol = rets.rolling(f"{int(window_minutes)}T").std().shift(1)
    return vol.rename("sigma")


def nearest_join_vol_to_bars(vol: pd.Series, bar_ts: pd.Series, tolerance_seconds: int = 60) -> pd.Series:
    vol_df = vol.dropna().to_frame(name="sigma").reset_index().rename(columns={vol.index.name or "index": "timestamp"})
    vol_df["timestamp"] = to_datetime_utc(vol_df["timestamp"])  # ensure tz-aware
    bars_df = pd.DataFrame({"timestamp": to_datetime_utc(bar_ts)})
    merged = pd.merge_asof(
        bars_df.sort_values("timestamp"),
        vol_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=int(tolerance_seconds)),
    )
    return merged["sigma"]


def main():
    ap = argparse.ArgumentParser(description="Label bars with triple-barrier (clock-horizon + percent/vol_time)")
    ap.add_argument("--bars_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/tbm.yaml")
    # External clock-vol options
    ap.add_argument("--vol_source", choices=["ticks", "timebars"], default=None,
                    help="Compute clock-window volatility from ticks or existing timebars, then join back to dollar bars.")
    ap.add_argument("--ticks_glob", default=None, help="Glob for raw tick CSVs (used when --vol_source=ticks).")
    ap.add_argument("--timebars_glob", default=None, help="Glob for timebars (parquet/csv) when --vol_source=timebars.")
    ap.add_argument("--timebar_freq", default="1T", help="Clock frequency for timebars (default 1T).")
    ap.add_argument("--vol_window_minutes", type=int, default=None, help="Clock window length in minutes for realized vol.")
    ap.add_argument("--nearest_tolerance_seconds", type=int, default=None, help="Max seconds for nearest join back to dollar bars.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}

    # Build labeling cfg (new schema under cfg['labeling']) with backward compatibility
    labeling = cfg.get("labeling")
    if labeling is None:
        # Fallback to legacy top-level keys
        labeling = {
            "use_ohlc": bool(cfg.get("use_ohlc", True)),
            "allow_same_bar_touch": bool(cfg.get("allow_same_bar_touch", True)),
            "same_bar_resolution": str(cfg.get("same_bar_resolution", "drop")),
            "horizon": cfg.get("horizon", {"type": "clock", "minutes": 30}),
            "max_holding_bars": int(cfg.get("max_holding_bars", 256)),
            "vol_method": str(cfg.get("vol_method", "atr")),
            "atr_kind": str(cfg.get("atr_kind", "wilder")),
            "atr_period": int(cfg.get("atr_period", 14)),
            "barriers": cfg.get("barriers", {"mode": "percent", "up": 0.005, "down": 0.004}),
        }
    labeling = validate_labeling_cfg(labeling)

    # Keep previous advanced options for time-vol if used in future (not needed for ATR-only run)
    vtd = cfg.get("vol_time_defaults", {}) or {}
    default_win = int(vtd.get("vol_window_minutes", 30))
    default_tol = int(vtd.get("nearest_tolerance_seconds", 60))
    default_freq = str(vtd.get("timebar_freq", "1T"))
    vol_source = args.vol_source
    ticks_glob = args.ticks_glob
    timebars_glob = args.timebars_glob
    timebar_freq = args.timebar_freq or default_freq
    vol_window_minutes = int(args.vol_window_minutes or default_win)
    nearest_tol = int(args.nearest_tolerance_seconds or default_tol)

    ensure_dir(args.out_dir)
    files = sorted(glob.glob(args.bars_glob))
    if not files:
        tqdm.write("No bar files found.")
        return

    for p in tqdm(files, desc="Labeling bars", unit="file"):
        cols = ["timestamp", "close"] + (["high", "low"] if labeling.get("use_ohlc", True) else [])
        try:
            df = pd.read_parquet(p, columns=cols)
        except Exception:
            df = pd.read_parquet(p)

        # Ensure datetime timestamps
        df["timestamp"] = to_datetime_utc(df["timestamp"])  # naive-safe
        # Label with diagnostics via unified helper, then augment with end_ts/time_to_hit_sec
        labels_df = label_bars_with_diagnostics(df, labeling)
        labels_df = add_end_ts_from_df(df, labels_df, labeling)

        # unit-like diagnostics for first 100 events
        k = min(100, len(labels_df))
        head = labels_df.iloc[:k].copy()
        if k > 0:
            # schema-safe: prefer entry_px, otherwise fallback to common names
            close_col = None
            for cand in ["entry_px", "close", "Close", "c", "price_close", "close_price", "last"]:
                if cand in head.columns:
                    close_col = cand
                    break
            if close_col:
                bad_mask = (~head['pt_px'].gt(head[close_col]).fillna(False)) | (~head['sl_px'].lt(head[close_col]).fillna(False))
            else:
                bad_mask = (~head['pt_px'].gt(head['sl_px']).fillna(False))
            if bool(bad_mask.any()):
                cols = [c for c in ["entry_px", "close", "pt_px", "sl_px"] if c in head.columns]
                bad = head.loc[bad_mask, cols].copy()
                n_bad = int(bad.shape[0])
                print(f"[WARN] Invalid barrier rows in first {k}: {n_bad}")
                print(bad.head(10))
                raise AssertionError("Invalid barrier construction; check ATR readiness / config / column mapping.")

        out = os.path.join(args.out_dir, os.path.basename(p).replace(".parquet", "_labels.parquet"))
        labels_df.to_parquet(out, index=False)

        # Meta sidecar
        n_rows = int(len(labels_df))
        n_dropped = int(labels_df.get("dropped", pd.Series([0]*n_rows)).sum())
        counts = pd.Series(labels_df["label"]).value_counts().to_dict()
        meta = {
            "bars_file": os.path.abspath(p),
            "out_file": os.path.abspath(out),
            "n_rows": n_rows,
            "n_dropped_same_bar": n_dropped,
            "label_counts": counts,
            "horizon": labeling.get("horizon", {}),
            "barriers": labeling.get("barriers", {}),
            "allow_same_bar_touch": bool(labeling.get("allow_same_bar_touch", True)),
            "same_bar_resolution": str(labeling.get("same_bar_resolution", "drop")),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(out.replace("_labels.parquet", "_labels.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

