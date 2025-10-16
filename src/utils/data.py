from __future__ import annotations
import pandas as pd
def read_ticks_csv(path: str) -> pd.DataFrame:
    """Return raw ticks as-is; column resolution/type safety happens in scripts/make_bars.py."""
    return pd.read_csv(path)
