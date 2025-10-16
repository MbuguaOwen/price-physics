from __future__ import annotations
import numpy as np, pandas as pd

def _infer_t1(df: pd.DataFrame, horizon_minutes: int = 30) -> pd.Series:
    """Return end timestamps (t1). Prefer explicit columns if present; otherwise approximate."""
    ts = pd.to_datetime(df.get("timestamp", df.index), utc=True, errors="coerce")
    # Best: explicit end_ts/t1
    for c in ("end_ts","t1","vertical_time","t_vert","t_end"):
        if c in df.columns:
            t1 = pd.to_datetime(df[c], utc=True, errors="coerce")
            if t1.notna().any():
                return t1
    # Next: use time_to_hit_sec if available
    if "time_to_hit_sec" in df.columns:
        t1 = ts + pd.to_timedelta(df["time_to_hit_sec"].fillna(horizon_minutes*60), unit="s")
        return t1
    # Fallback: horizon
    return ts + pd.to_timedelta(horizon_minutes, unit="m")

def compute_uniqueness_weights(
    df_labels: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon_minutes: int = 30,
    normalize: bool = True,
) -> pd.Series:
    """
    Lopez de Prado 'sample uniqueness' weights:
      w_i = mean_{t in [t0_i,t1_i)} (1 / concurrency(t))
    Returns a pd.Series aligned to df_labels.index
    """
    # t0 / t1
    t0 = pd.to_datetime(df_labels.get(timestamp_col, df_labels.index), utc=True, errors="coerce")
    t1 = _infer_t1(df_labels, horizon_minutes=horizon_minutes)
    mask = t0.notna() & t1.notna() & (t1 > t0)
    t0, t1 = t0[mask], t1[mask]
    idx_keep = df_labels.index[mask]

    # Build event endpoints and concurrency counts on a stepwise timeline
    # Use all unique boundaries to avoid per-second explosion
    bounds = pd.Index(t0.tolist() + t1.tolist()).unique().sort_values()
    if len(bounds) == 0:
        return pd.Series(1.0, index=df_labels.index)

    # Map each event to (start_bin, end_bin)
    start_bin = bounds.get_indexer(t0, method="pad")
    end_bin = bounds.get_indexer(t1, method="pad")
    # Build concurrency per interval via difference array trick
    diff = np.zeros(len(bounds)+1, dtype=np.int64)
    np.add.at(diff, start_bin, 1)
    np.add.at(diff, end_bin, -1)
    conc = np.cumsum(diff[:-1])
    # For each event, average 1/concurrency over its covered bins
    inv = 1.0 / np.maximum(conc.astype(float), 1.0)
    weights = []
    for s, e in zip(start_bin, end_bin):
        if e <= s:
            weights.append(1.0)
        else:
            w = inv[s:e].mean()
            weights.append(w)
    w_series = pd.Series(weights, index=idx_keep, dtype=float)
    # put back to full index
    w_full = pd.Series(1.0, index=df_labels.index, dtype=float)
    w_full.loc[idx_keep] = w_series
    if normalize and w_full.mean() > 0:
        w_full = w_full / w_full.mean()  # keep loss scale stable
    return w_full

