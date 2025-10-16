import numpy as np
import pandas as pd
from typing import Literal


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
    vol = vol.astype(float).reindex_like(close).fillna(method="ffill").fillna(vol.median())
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
    vol = vol.astype(float).reindex_like(close).fillna(method="ffill").fillna(vol.median())

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
