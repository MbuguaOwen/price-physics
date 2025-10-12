import pandas as pd
import numpy as np

def _normalize_timestamp(s):
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    try:
        if np.issubdtype(s.dtype, np.integer):
            vmax = int(s.max())
            if vmax > 10**14:
                return pd.to_datetime(s, unit="ns", utc=True)
            elif vmax > 10**12:
                return pd.to_datetime(s, unit="ms", utc=True)
            else:
                return pd.to_datetime(s, unit="s", utc=True)
        else:
            return pd.to_datetime(s, utc=True)
    except Exception:
        return pd.to_datetime(s, utc=True)

def build_dollar_bars(trades: pd.DataFrame, dollar_value: float = 5e5,
                      time_col: str = "timestamp", price_col: str = "price", qty_col: str = "qty") -> pd.DataFrame:
    """
    Vectorized dollar-bar construction:
      - Compute cumulative notional and use floor(cumsum/threshold) as a group id.
      - Aggregate O/H/L/C, qty sum, notional sum per group.
      - Drop the last (incomplete) group if its notional < threshold.
    """
    if trades.empty:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","qty","notional"])

    df = trades[[time_col, price_col, qty_col]].copy()
    df[time_col] = _normalize_timestamp(df[time_col])
    df = df.sort_values(time_col)

    # Notional per tick and cumulative groups
    v = df[price_col].to_numpy(float) * df[qty_col].to_numpy(float)
    cs = np.cumsum(v)
    bar_id = np.floor((cs - 1e-12) / float(dollar_value)).astype(np.int64)

    # Group & aggregate in one pass
    df = df.assign(notional=v, bar_id=bar_id)
    g = df.groupby("bar_id", sort=True, observed=True)

    bars = g.agg({
        time_col: "last",
        price_col: ["first", "max", "min", "last"],
        qty_col: "sum",
        "notional": "sum",
    }).reset_index(drop=True)

    # Flatten columns and rename
    bars.columns = ["timestamp", "open", "high", "low", "close", "qty", "notional"]

    # Drop trailing incomplete bar (if total notional < threshold)
    if len(bars) and float(bars["notional"].iloc[-1]) < float(dollar_value):
        bars = bars.iloc[:-1].reset_index(drop=True)

    return bars
