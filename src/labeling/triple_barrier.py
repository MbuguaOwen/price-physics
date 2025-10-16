import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from typing import Literal


def _resolve_ohlc_columns(df: pd.DataFrame):
    """
    Map canonical names -> actual column names in df.
    Only requires high/low/close; 'open' is optional.
    Raises if required columns are missing.
    """
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    mapping = {
        "open": pick(["open", "Open", "o", "price_open", "open_price"]),
        "high": pick(["high", "High", "h", "price_high", "high_price"]),
        "low": pick(["low", "Low", "l", "price_low", "low_price"]),
        "close": pick(["close", "Close", "c", "price_close", "close_price", "last"]),
    }
    required = ("high", "low", "close")
    missing = [k for k in required if mapping.get(k) is None]
    if missing:
        raise ValueError(f"Missing OHLC columns in bars: {missing}; have={list(df.columns)}")
    return mapping


def ewma_vol(returns: pd.Series, span: int = 32) -> pd.Series:
    """EWMA volatility proxy of returns with no-lookahead (shifted by +1).

    Parameters
    - returns: pct-change series (index-aligned to prices)
    - span: EWMA span

    Returns
    - Series of EWMA std dev shifted by 1 bar to avoid lookahead.
    """
    vol = (
        returns.ewm(span=span, adjust=False)
        .std()
        .shift(1)  # NO LOOKAHEAD: use info up to i-1
        .bfill()
        .replace(0, np.nan)
        .ffill()
        .fillna(0.0)
    )
    return vol


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range using high/low and previous close.

    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"true_range requires column '{col}', got {list(df.columns)}")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr_vol(
    df: pd.DataFrame,
    period: int = 14,
    kind: Literal["wilder", "ema"] = "wilder",
    normalize_by_close: bool = True,
) -> pd.Series:
    """Compute no-lookahead ATR volatility.

    - Uses True Range and either Wilder's smoothing or EMA.
    - Shifted by +1 so bar i uses ATR up to i-1 (no lookahead).
    - If normalize_by_close, returns ATR/close (relative width).
    """
    tr = true_range(df)
    if kind == "wilder":
        atr = tr.ewm(alpha=1 / max(1, int(period)), adjust=False).mean()
    else:  # "ema"
        atr = tr.ewm(span=max(1, int(period)), adjust=False).mean()

    # NO LOOKAHEAD: shift so entry at i uses ATR up to i-1
    atr = atr.shift(1)

    if normalize_by_close:
        close = df["close"].astype(float)
        out = (atr / close).bfill().fillna(0.0)
        return out
    return atr.bfill().fillna(0.0)


def triple_barrier_labels(
    close: pd.Series,
    vol: pd.Series,
    pt_mult: float,
    sl_mult: float,
    max_holding_bars: int,
):
    """Close-only triple-barrier labels (backward compatibility).

    - Scans from i+1 to horizon t1[i].
    - Returns DataFrame with label, t1, pt_hit, sl_hit.
    """
    assert len(close) == len(vol), "close and vol must be same length"
    assert close.index.equals(vol.index), "close and vol must share index"
    n = len(close)
    t1 = np.minimum(np.arange(n) + int(max_holding_bars), n - 1).astype(int)
    labels = np.zeros(n, dtype=int)
    pt_idx = np.full(n, -1, dtype=int)
    sl_idx = np.full(n, -1, dtype=int)

    for i in range(n - 1):
        vi = float(vol.iloc[i])
        pt = float(close.iloc[i]) * (1 + float(pt_mult) * vi)
        sl = float(close.iloc[i]) * (1 - float(sl_mult) * vi)
        for j in range(i + 1, int(t1[i]) + 1):
            cj = float(close.iloc[j])
            if cj >= pt:
                labels[i] = 1
                pt_idx[i] = j
                break
            if cj <= sl:
                labels[i] = -1
                sl_idx[i] = j
                break

    return pd.DataFrame({"label": labels, "t1": t1, "pt_hit": pt_idx, "sl_hit": sl_idx})


def triple_barrier_labels_ohlc(
    df: pd.DataFrame,
    vol: pd.Series,
    pt_mult: float,
    sl_mult: float,
    max_holding_bars: int,
    allow_same_bar_touch: bool = False,
    same_bar_resolution: Literal["drop", "pt_wins", "sl_wins", "nearest", "random"] = "drop",
    rng_seed: int = 1337,
) -> pd.DataFrame:
    """OHLC-aware triple-barrier labeling with conservative no-lookahead.

    Invariants:
    - No lookahead: `vol` must already be shifted by +1 (bar i uses info up to i-1).
    - Default scanning starts at i+1 (no same-bar exits).
    - Ambiguous same-bar PT/SL resolution is configurable; default "drop".

    Returns DataFrame with columns: label, t1, pt_hit, sl_hit, dropped
    where pt_hit/sl_hit are the first index j where the barrier was hit (or -1),
    and dropped=1 flags ambiguous same-bar cases that were dropped.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"OHLC labeler requires columns {sorted(required)}; missing: {sorted(missing)}")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Cheap alignment assertions
    assert len(vol) == len(close), "vol length must equal df length"
    assert vol.index.equals(close.index), "vol index must match df index"

    n = len(close)
    t1 = np.minimum(np.arange(n) + int(max_holding_bars), n - 1).astype(int)
    labels = np.zeros(n, dtype=int)
    pt_idx = np.full(n, -1, dtype=int)
    sl_idx = np.full(n, -1, dtype=int)
    dropped = np.zeros(n, dtype=np.int8)

    rng = np.random.default_rng(int(rng_seed))

    for i in range(n - 1):
        vi = float(vol.iloc[i])
        ci = float(close.iloc[i])
        pt = ci * (1 + float(pt_mult) * vi)
        sl = ci * (1 - float(sl_mult) * vi)
        start = i if bool(allow_same_bar_touch) else i + 1

        for j in range(start, int(t1[i]) + 1):
            hit_pt = bool(high.iloc[j] >= pt)
            hit_sl = bool(low.iloc[j] <= sl)

            if hit_pt and not hit_sl:
                labels[i] = 1
                pt_idx[i] = j
                break
            if hit_sl and not hit_pt:
                labels[i] = -1
                sl_idx[i] = j
                break
            if hit_pt and hit_sl:
                # Ambiguous: both touched within same bar j
                res = str(same_bar_resolution).lower()
                if res == "drop":
                    labels[i] = 0
                    dropped[i] = 1
                    pt_idx[i] = -1
                    sl_idx[i] = -1
                    break
                elif res == "pt_wins":
                    labels[i] = 1
                    pt_idx[i] = j
                    break
                elif res == "sl_wins":
                    labels[i] = -1
                    sl_idx[i] = j
                    break
                elif res == "nearest":
                    dist_pt = abs(pt - ci)
                    dist_sl = abs(ci - sl)
                    if dist_pt <= dist_sl:
                        labels[i] = 1
                        pt_idx[i] = j
                    else:
                        labels[i] = -1
                        sl_idx[i] = j
                    break
                elif res == "random":
                    if float(rng.random()) < 0.5:
                        labels[i] = 1
                        pt_idx[i] = j
                    else:
                        labels[i] = -1
                        sl_idx[i] = j
                    break
                else:
                    # Unknown resolution: default safe behavior is drop
                    labels[i] = 0
                    dropped[i] = 1
                    pt_idx[i] = -1
                    sl_idx[i] = -1
                    break

    return pd.DataFrame(
        {"label": labels, "t1": t1, "pt_hit": pt_idx, "sl_hit": sl_idx, "dropped": dropped}
    )


# New: clock-horizon (t1 index array) variants
def _resolve_same_bar(res: str, rng) -> str:
    r = str(res or "drop").lower().strip()
    if r in ("pt_first", "pt_wins"):
        return "pt_wins"
    if r in ("sl_first", "sl_wins"):
        return "sl_wins"
    if r in ("nearest", "random", "drop"):
        return r
    return "drop"


def triple_barrier_labels_t1(
    close: pd.Series,
    vol: pd.Series,
    pt_mult: float,
    sl_mult: float,
    t1: np.ndarray,
) -> pd.DataFrame:
    """Close-only variant where horizon is provided as explicit index array t1.

    Parameters
    - close: price close series
    - vol: precomputed no-lookahead volatility proxy (aligned to close)
    - pt_mult/sl_mult: multipliers applied to vol to size barriers
    - t1: numpy array of horizon indices (inclusive upper bound)
    """
    close = close.astype(float)
    vol = vol.astype(float).reindex_like(close).ffill().fillna(vol.median())
    n = len(close)
    labels = np.zeros(n, dtype=int)
    pt_idx = np.full(n, -1, dtype=int)
    sl_idx = np.full(n, -1, dtype=int)

    for i in range(n - 1):
        horizon = int(t1[i])
        ci = float(close.iloc[i])
        vi = float(vol.iloc[i])
        pt = ci * (1 + float(pt_mult) * vi)
        sl = ci * (1 - float(sl_mult) * vi)
        for j in range(i + 1, min(n - 1, horizon) + 1):
            cj = float(close.iloc[j])
            if cj >= pt:
                labels[i] = 1
                pt_idx[i] = j
                break
            if cj <= sl:
                labels[i] = -1
                sl_idx[i] = j
                break
    return pd.DataFrame({"label": labels, "t1": t1, "pt_hit": pt_idx, "sl_hit": sl_idx})


def triple_barrier_labels_ohlc_t1(
    df: pd.DataFrame,
    vol: pd.Series,
    pt_mult: float,
    sl_mult: float,
    t1: np.ndarray,
    allow_same_bar_touch: bool = True,
    same_bar_resolution: Literal["drop", "pt_wins", "sl_wins", "nearest", "random"] = "sl_wins",
    rng_seed: int = 1337,
) -> pd.DataFrame:
    """OHLC-aware variant where horizon is provided as explicit index array t1.

    - allow_same_bar_touch=True allows exits at bar i (same bar) via configurable resolver.
    - same_bar_resolution supports aliases (sl_first/pt_first) via _resolve_same_bar.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"OHLC labeler requires columns {sorted(required)}; missing: {sorted(missing)}")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = vol.astype(float).reindex_like(close).ffill().fillna(vol.median())

    n = len(close)
    labels = np.zeros(n, dtype=int)
    pt_idx = np.full(n, -1, dtype=int)
    sl_idx = np.full(n, -1, dtype=int)
    dropped = np.zeros(n, dtype=np.int8)
    rng = np.random.RandomState(int(rng_seed))
    same = _resolve_same_bar(same_bar_resolution, rng)

    for i in range(n - 1):
        horizon = int(t1[i])
        ci = float(close.iloc[i])
        vi = float(vol.iloc[i])
        pt = ci * (1 + float(pt_mult) * vi)
        sl = ci * (1 - float(sl_mult) * vi)
        start = i if bool(allow_same_bar_touch) else i + 1

        for j in range(start, min(n - 1, horizon) + 1):
            hit_up = bool(high.iloc[j] >= pt)
            hit_dn = bool(low.iloc[j] <= sl)
            if hit_up and hit_dn:
                if same == "pt_wins":
                    labels[i] = 1; pt_idx[i] = j
                elif same == "sl_wins":
                    labels[i] = -1; sl_idx[i] = j
                elif same == "nearest":
                    du = abs(high.iloc[j] - pt); dd = abs(low.iloc[j] - sl)
                    if du <= dd: labels[i] = 1; pt_idx[i] = j
                    else: labels[i] = -1; sl_idx[i] = j
                elif same == "random":
                    if rng.rand() < 0.5: labels[i] = 1; pt_idx[i] = j
                    else: labels[i] = -1; sl_idx[i] = j
                else:
                    labels[i] = 0; dropped[i] = 1; pt_idx[i] = -1; sl_idx[i] = -1
                break
            if hit_up:
                labels[i] = 1; pt_idx[i] = j; break
            if hit_dn:
                labels[i] = -1; sl_idx[i] = j; break

    return pd.DataFrame({"label": labels, "t1": t1, "pt_hit": pt_idx, "sl_hit": sl_idx, "dropped": dropped})


# ===== New high-level labeling helpers with config + diagnostics =====
def validate_labeling_cfg(cfg: dict) -> dict:
    """
    Validate and normalize the labeling configuration.

    Enforces barrier mode exclusivity and derives implied defaults.
    - mode == 'atr': requires pt_mult/sl_mult; forbids up/down; normalize_by_close=False
    - mode == 'percent': requires up/down; forbids pt_mult/sl_mult; normalize_by_close=True
    """
    if cfg is None:
        raise ValueError("labeling config missing")
    lab = dict(cfg)
    b = dict(lab.get("barriers", {}))
    mode = str(b.get("mode", "percent")).lower()
    if mode not in {"atr", "percent"}:
        raise ValueError(f"barriers.mode must be 'atr' or 'percent', got {mode}")

    if mode == "atr":
        if any(k in b for k in ("up", "down")):
            raise ValueError("barriers.mode='atr' forbids keys 'up'/'down'; use pt_mult/sl_mult instead.")
        if not all(k in b for k in ("pt_mult", "sl_mult")):
            raise ValueError("barriers.mode='atr' requires 'pt_mult' and 'sl_mult'.")
        lab["normalize_by_close"] = False
    else:  # percent
        if any(k in b for k in ("pt_mult", "sl_mult")):
            raise ValueError("barriers.mode='percent' forbids 'pt_mult'/'sl_mult'; use up/down instead.")
        if not all(k in b for k in ("up", "down")):
            raise ValueError("barriers.mode='percent' requires 'up' and 'down'.")
        lab["normalize_by_close"] = True

    # Defaults (do not require 'open')
    lab.setdefault("use_ohlc", True)
    lab.setdefault("allow_same_bar_touch", True)
    lab.setdefault("same_bar_resolution", "drop")
    lab.setdefault("horizon", {"type": "clock", "minutes": 240})
    lab.setdefault("max_holding_bars", 5000)
    lab.setdefault("atr_kind", "wilder")
    lab.setdefault("atr_period", 14)
    lab.setdefault("exclusive_events", False)
    lab.setdefault("min_gap_bars", 0)
    lab["barriers"] = b
    return lab


def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR on OHLC (unshifted). Caller must apply shift(1) to avoid look-ahead."""
    m = _resolve_ohlc_columns(df)
    h, l, c = df[m["high"]], df[m["low"]], df[m["close"]]
    # Explicit form to match Wilder definition and clarity
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / max(1, int(period)), adjust=False).mean()


def build_barriers(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Construct PT/SL barrier prices per row; drop invalid rows.

    Returns a DataFrame indexed like df (filtered) with columns close, pt_px, sl_px.
    """
    mode = str(cfg.get("barriers", {}).get("mode", "percent")).lower()
    if mode not in {"atr", "percent"}:
        raise ValueError(f"barriers.mode must be 'atr' or 'percent', got {mode}")

    m = _resolve_ohlc_columns(df)
    ci = df[m["close"]].astype(float)
    if mode == "atr":
        atr = wilder_atr(df, int(cfg.get("atr_period", 14))).shift(1)  # NO LOOK-AHEAD
        # Clip ATR to tiny epsilon and forward-fill to avoid zero/NaN distances early
        eps = (ci.abs() * 1e-12).fillna(1e-12)
        atr = atr.clip(lower=eps).ffill()
        pt = ci + atr * float(cfg["barriers"]["pt_mult"])
        sl = ci - atr * float(cfg["barriers"]["sl_mult"])
        valid = ci.notna() & pt.notna() & sl.notna() & (pt > ci) & (sl < ci)
    else:
        up = float(cfg["barriers"]["up"])
        dn = float(cfg["barriers"]["down"])
        pt = ci * (1.0 + up)
        sl = ci * (1.0 - dn)
        valid = ci.notna() & pt.notna() & sl.notna()

    out = pd.DataFrame({"entry_px": ci, "pt_px": pt, "sl_px": sl})
    return out.loc[valid]


def _to_datetime_utc(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    return t


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a UTC DatetimeIndex. If there's a timestamp-like column,
    use it; otherwise require an existing DatetimeIndex.
    Handles epoch ns/ms/s and ISO strings. Tz-aware safe.
    """
    out = df.copy()

    # Already datetime index -> ensure UTC
    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        out.index = idx
        return out.sort_index()

    # Choose a timestamp column
    cand_cols = ["timestamp","time","datetime","Date","date","ts","ts_ms"]
    col = next((c for c in cand_cols if c in out.columns), None)
    if col is None:
        raise ValueError("No DatetimeIndex and no timestamp column found; have columns: " + str(list(out.columns)))

    s_ts = out[col]
    if is_datetime64_any_dtype(s_ts):
        dt = pd.to_datetime(s_ts, utc=True, errors="coerce")
    elif getattr(s_ts.dtype, 'kind', '') in "iu":  # integers (fast path)
        vmax = float(pd.to_numeric(s_ts, errors="coerce").max() or 0.0)
        unit = "ns" if vmax > 1e14 else ("ms" if vmax > 1e11 else "s")
        dt = pd.to_datetime(s_ts, unit=unit, utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(s_ts, utc=True, errors="coerce")

    if dt.isna().all():
        raise ValueError(f"Failed to parse datetime from column '{col}'.")

    out = out.drop(columns=[col])
    out.index = dt
    return out.sort_index()


def _compute_t1_clock_idx_from_index(index: pd.DatetimeIndex, minutes: int) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("_compute_t1_clock_idx_from_index requires a DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise ValueError("Bars index is not strictly increasing. Fix ordering before labeling.")
    tgt = index + pd.to_timedelta(int(minutes), unit="m")
    a = index.to_numpy(dtype="datetime64[ns]").astype("int64")
    b = tgt.to_numpy(dtype="datetime64[ns]").astype("int64")
    j = np.searchsorted(a, b, side="left")
    j = np.clip(j, 0, len(index) - 1).astype(int)
    return j


def label_bars_with_diagnostics(
    df: pd.DataFrame,
    labeling: dict,
) -> pd.DataFrame:
    """
    Label bars using provided labeling cfg and return diagnostics.

    Returns a DataFrame with columns:
      i, t_end, label, end_reason, pt_px, sl_px, pt_idx, sl_idx, t1_idx, horizon_minutes, dropped
    """
    cfg = validate_labeling_cfg(labeling)
    # Ensure datetime index
    df = ensure_datetime_index(df)

    # Build barriers on the full frame, then restrict to valid rows
    bar_df = build_barriers(df, cfg)
    dfv = df.loc[bar_df.index]
    m = _resolve_ohlc_columns(dfv)
    close = dfv[m["close"]].astype(float)
    high = dfv[m["high"]].astype(float)
    low = dfv[m["low"]].astype(float)
    pt_px = bar_df["pt_px"]
    sl_px = bar_df["sl_px"]

    # Horizon as indices (clock)
    hz = cfg.get("horizon", {})
    hz_mins = int(hz.get("minutes", 240))
    t1_idx = _compute_t1_clock_idx_from_index(dfv.index, minutes=hz_mins)

    n = len(close)
    allow_same = bool(cfg.get("allow_same_bar_touch", True))
    res = str(cfg.get("same_bar_resolution", "drop")).lower()
    exclusive = bool(cfg.get("exclusive_events", False))
    min_gap = int(cfg.get("min_gap_bars", 0))

    labels = np.zeros(n, dtype=int)
    pt_idx = np.full(n, -1, dtype=int)
    sl_idx = np.full(n, -1, dtype=int)
    dropped = np.zeros(n, dtype=np.int8)
    end_reason = np.array(["" for _ in range(n)], dtype=object)
    t_end = np.arange(n, dtype=int)

    next_start = 0
    for i in range(n - 1):
        if exclusive and i < next_start:
            dropped[i] = 1
            end_reason[i] = "dropped"
            t_end[i] = i
            continue

        ci = float(close.iloc[i])
        pt = float(pt_px.iloc[i])
        sl = float(sl_px.iloc[i])
        start = i if allow_same else i + 1
        horizon = int(t1_idx[i])
        if horizon < start:
            horizon = start

        hit = None
        hit_idx = -1
        for j in range(start, horizon + 1):
            hit_up = bool(high.iloc[j] >= pt)
            hit_dn = bool(low.iloc[j] <= sl)
            if hit_up and hit_dn:
                if res in ("pt_first", "pt_wins"):
                    hit = "pt"; hit_idx = j
                elif res in ("sl_first", "sl_wins"):
                    hit = "sl"; hit_idx = j
                else:  # drop neutral
                    dropped[i] = 1
                    end_reason[i] = "dropped"
                    hit = None
                    hit_idx = -1
                break
            if hit_up:
                hit = "pt"; hit_idx = j; break
            if hit_dn:
                hit = "sl"; hit_idx = j; break

        if hit == "pt":
            labels[i] = 1
            pt_idx[i] = hit_idx
            end_reason[i] = "pt"
            t_end[i] = hit_idx
        elif hit == "sl":
            labels[i] = -1
            sl_idx[i] = hit_idx
            end_reason[i] = "sl"
            t_end[i] = hit_idx
        elif dropped[i] == 1:
            labels[i] = 0
            t_end[i] = i
        else:
            labels[i] = 0
            end_reason[i] = "t1"
            t_end[i] = horizon

        if exclusive and t_end[i] >= i:
            next_start = max(next_start, int(t_end[i]) + 1 + min_gap)

    out = pd.DataFrame(
        {
            "i": np.arange(n, dtype=int),
            "t_end": t_end,
            "label": labels,
            "end_reason": end_reason,
            "entry_px": close.astype(float),
            "pt_px": pt_px.astype(float),
            "sl_px": sl_px.astype(float),
            "pt_idx": pt_idx,
            "sl_idx": sl_idx,
            "t1_idx": t1_idx,
            "horizon_minutes": int(hz_mins),
            "dropped": dropped,
        }
    )

    # Basic assertions (first 100 rows) — skip NaNs and zero-distance edges
    k = min(100, n)
    if k > 0:
        ci = close.iloc[:k].to_numpy()
        ptk = out["pt_px"].iloc[:k].to_numpy()
        slk = out["sl_px"].iloc[:k].to_numpy()
        mask = np.isfinite(ptk) & np.isfinite(slk)
        if mask.any():
            assert (ptk[mask] > ci[mask]).all()
            assert (slk[mask] < ci[mask]).all()
        assert (out["t1_idx"].iloc[:k].to_numpy() >= np.arange(k)).all()

    return out
