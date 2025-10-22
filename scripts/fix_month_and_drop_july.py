import argparse, pandas as pd, numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--drop_month", default="2025-07")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # Parse t0, derive month, guard against NaT rows
    df["t0"] = pd.to_datetime(df.get("t0"), errors="coerce", utc=True)
    before = len(df)
    df = df[df["t0"].notna()].copy()

    # Derive month strictly from t0
    df["month"] = df["t0"].dt.strftime("%Y-%m")

    # Drop the unwanted month explicitly
    df = df[df["month"] != args.drop_month].copy()

    # (Optional) sanity: keep only snapshot months if provided in file
    # pass; the permanent patch in the labeler enforces this

    df.to_csv(args.out_csv, index=False)

    # Robust prints that don’t break on NaN
    months = sorted(df["month"].dropna().astype(str).unique().tolist())
    mix = (df.get("label", pd.Series(dtype=int)).value_counts(normalize=True) * 100.0).round(2).to_dict()

    print(f"[fix] input rows: {before} -> output rows: {len(df)}")
    print(f"[fix] months: {months}")
    print(f"[fix] class mix %: {mix}")


if __name__ == "__main__":
    main()

