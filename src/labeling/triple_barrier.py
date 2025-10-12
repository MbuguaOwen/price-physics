import numpy as np
import pandas as pd

def ewma_vol(returns: pd.Series, span: int = 32) -> pd.Series:
    return returns.ewm(span=span, adjust=False).std().bfill().replace(0, np.nan).ffill()

def atr_vol(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = (h - l).abs().combine((h - prev_c).abs(), max).combine((l - prev_c).abs(), max)
    return tr.rolling(period, min_periods=1).mean().bfill()

def triple_barrier_labels(close: pd.Series, vol: pd.Series, pt_mult: float, sl_mult: float, max_holding_bars: int):
    n = len(close)
    t1 = np.minimum(np.arange(n) + max_holding_bars, n - 1)
    labels = np.zeros(n, dtype=int)
    pt_idx = np.full(n, -1); sl_idx = np.full(n, -1)
    for i in range(n - 1):
        pt = close.iloc[i] * (1 + pt_mult * vol.iloc[i])
        sl = close.iloc[i] * (1 - sl_mult * vol.iloc[i])
        for j in range(i + 1, int(t1[i]) + 1):
            if close.iloc[j] >= pt:
                labels[i] = 1; pt_idx[i] = j; break
            if close.iloc[j] <= sl:
                labels[i] = -1; sl_idx[i] = j; break
    return pd.DataFrame({"label": labels, "t1": t1, "pt_hit": pt_idx, "sl_hit": sl_idx})
