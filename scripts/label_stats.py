from __future__ import annotations
import glob, os, re, sys
import pandas as pd

# Acceptance criteria (QA guidance):
# - Label 0 share should be significantly lower than prior (~86%); target <50% at h240.
# - If still high, test with horizon minutes 360/480.
# - end_reason should show more 'pt'/'sl' and fewer 't1'.
# - Same-bar collisions appear as dropped (or zero if rare).
# - No import errors when run as `python -m scripts.label_stats` or directly.

LABEL_CANDIDATES = ["label","tb_label","tbm_label","y","target","y_cls"]
MONTH_RE = re.compile(r"(20\d{2}-\d{2})")

def find_label_col(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns[::-1]:
        if pd.api.types.is_integer_dtype(df[c]):
            return c
    raise RuntimeError(f"Could not find label column among {LABEL_CANDIDATES}")

def month_from_ts(series):
    s = pd.to_datetime(series, unit="ms", errors="ignore")
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.strftime("%Y-%m").fillna("unknown")

def month_from_path(path):
    m = MONTH_RE.search(path.replace("\\","/"))
    return m.group(1) if m else "unknown"

def main(glob_pattern="data/labels/*.parquet"):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        print(f"No label files for: {glob_pattern}"); sys.exit(2)

    parts = []
    for p in files:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"Skip {p}: {e}")
            continue
        col = find_label_col(df)
        if "timestamp" in df.columns and len(df):
            month = month_from_ts(df["timestamp"]).iloc[0]
        else:
            month = month_from_path(p)
        parts.append(pd.DataFrame({col: df[col], "month":[month]*len(df)}))

    X = pd.concat(parts, ignore_index=True)
    col = find_label_col(X)
    total = len(X)
    vc = X[col].value_counts(dropna=False).sort_index()
    pct = (vc / total * 100).round(2)

    print("\n=== Label distribution (all files) ===")
    print(pd.DataFrame({"count": vc, "percent": pct}))

    # Optional: end_reason distribution if present
    if "end_reason" in X.columns:
        er_vc = X["end_reason"].value_counts(dropna=False).sort_index()
        er_pct = (er_vc / total * 100).round(2)
        print("\n=== End reason distribution (all files) ===")
        print(pd.DataFrame({"count": er_vc, "percent": er_pct}))

    print("\n=== By month ===")
    bym = X.groupby(["month", col]).size().unstack(fill_value=0).sort_index()
    bym["total"] = bym.sum(axis=1)
    for c2 in bym.columns:
        if c2 != "total":
            bym[f"{c2}_pct"] = (bym[c2] / bym["total"] * 100).round(2)
    print(bym)

if __name__ == "__main__":
    glob_pattern = sys.argv[1] if len(sys.argv) > 1 else "data/labels/*.parquet"
    main(glob_pattern)
