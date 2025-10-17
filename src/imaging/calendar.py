import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

def calendar_maps(timestamps, image_size: int):
    ts_in = timestamps
    try:
        s = pd.Series(ts_in)
        if is_datetime64_any_dtype(s) or is_datetime64tz_dtype(s):
            ts = pd.to_datetime(s, utc=True)
        else:
            ts = pd.to_datetime(ts_in, utc=True)
    except Exception:
        ts = pd.to_datetime(ts_in, utc=True)
    tod = ts.dt.hour + ts.dt.minute/60.0 + ts.dt.second/3600.0
    tod_sin = np.sin(2*np.pi * tod/24.0).iloc[-1]
    tod_cos = np.cos(2*np.pi * tod/24.0).iloc[-1]
    dow = ts.dt.dayofweek.iloc[-1]
    dow_sin = np.sin(2*np.pi * dow/7.0)
    dow_cos = np.cos(2*np.pi * dow/7.0)
    maps = []
    for val in [tod_sin, tod_cos, dow_sin, dow_cos]:
        maps.append(np.full((image_size, image_size), float(val), dtype=np.float32))
    return maps
