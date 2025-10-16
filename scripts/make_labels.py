import argparse, glob, json, os
from datetime import datetime, timezone
import yaml, pandas as pd, numpy as np
from tqdm.auto import tqdm

from src.labeling.triple_barrier import (
    triple_barrier_labels_t1,
    triple_barrier_labels_ohlc_t1,
    ewma_vol,
    atr_vol,
)


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
    t = to_datetime_utc(ts)
    tgt = t + pd.to_timedelta(int(minutes), unit="m")
    a = t.view("int64").to_numpy()
    b = tgt.view("int64").to_numpy()
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

    horizon = cfg.get("horizon", {"type": "clock", "minutes": 30})
    hz_type = str(horizon.get("type", "clock")).lower()
    hz_mins = int(horizon.get("minutes", 30))

    barriers = cfg.get("barriers", {"mode": "percent", "up": 0.005, "down": 0.004})
    mode = str(barriers.get("mode", "percent")).lower()

    # Defaults for vol_time from YAML (non-breaking if missing)
    vtd = cfg.get("vol_time_defaults", {}) or {}
    default_win = int(vtd.get("vol_window_minutes", 30))
    default_tol = int(vtd.get("nearest_tolerance_seconds", 60))
    default_freq = str(vtd.get("timebar_freq", "1T"))

    # CLI overrides
    vol_source = args.vol_source
    ticks_glob = args.ticks_glob
    timebars_glob = args.timebars_glob
    timebar_freq = args.timebar_freq or default_freq
    vol_window_minutes = int(args.vol_window_minutes or default_win)
    nearest_tol = int(args.nearest_tolerance_seconds or default_tol)

    use_ohlc = to_bool(cfg.get("use_ohlc", True), default=True)
    allow_same_bar_touch = to_bool(cfg.get("allow_same_bar_touch", True), default=True)
    sbr = str(cfg.get("same_bar_resolution", "sl_first")).lower().strip()
    if sbr == "sl_first":
        sbr = "sl_wins"
    if sbr == "pt_first":
        sbr = "pt_wins"

    vol_method = str(cfg.get("vol_method", "atr")).lower()
    atr_period = int(cfg.get("atr_period", 14))
    atr_kind = str(cfg.get("atr_kind", "wilder")).lower()
    normalize_by_close = to_bool(cfg.get("normalize_by_close", True), default=True)
    ewma_span = int(cfg.get("ewma_span", 32))

    ensure_dir(args.out_dir)
    files = sorted(glob.glob(args.bars_glob))
    if not files:
        tqdm.write("No bar files found.")
        return

    for p in tqdm(files, desc="Labeling bars", unit="file"):
        cols = ["timestamp", "close"] + (["high", "low"] if use_ohlc else [])
        try:
            df = pd.read_parquet(p, columns=cols)
        except Exception:
            df = pd.read_parquet(p)

        # Ensure datetime timestamps
        df["timestamp"] = to_datetime_utc(df["timestamp"])  # naive-safe
        close = df["close"].astype(float)
        rets = close.pct_change().fillna(0.0)
        rets.index = df["timestamp"]

        # Barrier sizing
        if mode == "percent":
            vol = pd.Series(1.0, index=close.index)
            pt_mult = float(barriers.get("up", 0.005))
            sl_mult = float(barriers.get("down", 0.004))
        elif mode in ("vol_time", "time_vol", "realized_time"):
            # Strict external clock-vol: build from ticks OR provided timebars, then nearest-join back to dollar bars
            if (vol_source or "").lower() == "ticks":
                import glob as _glob
                tick_paths = sorted(_glob.glob(ticks_glob or ""))
                if not tick_paths:
                    raise SystemExit("vol_time: --vol_source=ticks but no --ticks_glob matched any files.")
                tb = build_timebars_from_ticks(tick_paths, freq=timebar_freq)
                vol_clock = realized_vol_from_timebars(tb, window_minutes=vol_window_minutes)
            elif (vol_source or "").lower() == "timebars":
                import glob as _glob
                tpaths = sorted(_glob.glob(timebars_glob or ""))
                if not tpaths:
                    raise SystemExit("vol_time: --vol_source=timebars but no --timebars_glob matched any files.")
                frames = []
                for fp in tpaths:
                    if fp.lower().endswith(".parquet"):
                        df_tb = pd.read_parquet(fp)
                    else:
                        df_tb = pd.read_csv(fp)
                    if "timestamp" not in df_tb.columns:
                        raise SystemExit(f"timebars missing timestamp: {fp}")
                    df_tb["timestamp"] = to_datetime_utc(df_tb["timestamp"])  # ensure tz-aware
                    df_tb = df_tb.sort_values("timestamp").set_index("timestamp")
                    frames.append(df_tb[["close"]])
                tb = pd.concat(frames).sort_index()
                tb = tb.resample(timebar_freq).last().dropna()
                vol_clock = realized_vol_from_timebars(tb, window_minutes=vol_window_minutes)
            else:
                # Fallback: compute on dollar-bar timestamps (clock-windowed)
                vol_clock = rets.rolling(f"{int(vol_window_minutes)}T").std().shift(1)

            # Join clock vol back to bars on nearest<=tolerance
            vol = nearest_join_vol_to_bars(vol_clock, df["timestamp"], tolerance_seconds=nearest_tol)
            vol = vol.fillna(method="ffill").fillna(method="bfill")
            pt_mult = float(barriers.get("k_up", 1.5))
            sl_mult = float(barriers.get("k_dn", 1.2))
        else:
            if use_ohlc and {"high", "low", "close"}.issubset(df.columns) and vol_method == "atr":
                vol = atr_vol(df, period=atr_period, kind=("ema" if atr_kind == "ema" else "wilder"), normalize_by_close=normalize_by_close)
            else:
                vol = ewma_vol(rets, span=ewma_span)
            pt_mult = float(cfg.get("pt_mult", 1.0))
            sl_mult = float(cfg.get("sl_mult", 1.0))

        # Horizon as indices
        if hz_type == "clock":
            t1 = compute_t1_clock(df["timestamp"], minutes=hz_mins)
        else:
            m = int(cfg.get("max_holding_bars", 256))
            n = len(df)
            t1 = np.minimum(np.arange(n) + m, n - 1).astype(int)

        # Labeling
        if use_ohlc and {"high", "low", "close"}.issubset(df.columns):
            labels_df = triple_barrier_labels_ohlc_t1(
                df=df,
                vol=vol,
                pt_mult=pt_mult,
                sl_mult=sl_mult,
                t1=t1,
                allow_same_bar_touch=allow_same_bar_touch,
                same_bar_resolution=sbr,
                rng_seed=1337,
            )
        else:
            labels_df = triple_barrier_labels_t1(
                close=close,
                vol=vol,
                pt_mult=pt_mult,
                sl_mult=sl_mult,
                t1=t1,
            )

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
            "horizon": {"type": hz_type, "minutes": hz_mins},
            "barriers": {"mode": mode, **{k: v for k, v in barriers.items() if k != "mode"}},
            "allow_same_bar_touch": bool(allow_same_bar_touch),
            "same_bar_resolution": sbr,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "vol_time": {
                "mode": mode,
                "source": (vol_source or "inline"),
                "timebar_freq": timebar_freq,
                "window_minutes": vol_window_minutes,
                "nearest_tolerance_seconds": nearest_tol,
                "join_coverage": float(vol.notna().mean()) if isinstance(vol, pd.Series) else None,
            },
        }
        with open(out.replace("_labels.parquet", "_labels.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
