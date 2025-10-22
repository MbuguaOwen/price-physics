import pandas as pd


def test_output_months_subset_of_snapshot():
    snap = r"data/images_jan_jun/btc_snapshot_2025_01_06_FIXED.csv"
    out  = r"data/images_jan_jul/btc_labels_ticksONLY_GROUPED_m600.csv"
    s = set(pd.read_csv(snap, usecols=["month"])['month'].astype(str).unique())
    d = pd.read_csv(out, usecols=["t0"])
    d["t0"] = pd.to_datetime(d["t0"], errors="coerce", utc=True)
    d["month"] = d["t0"].dt.strftime("%Y-%m")
    m = set(d["month"].dropna().unique())
    assert m.issubset(s), f"Output months {sorted(m - s)} not in snapshot"

