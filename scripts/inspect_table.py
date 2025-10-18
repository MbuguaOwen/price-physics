import argparse, pandas as pd, os
from pathlib import Path

def load_any(path, nrows=None):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if ext == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    if ext in (".feather", ".fea"):
        df = pd.read_feather(path)
        return df.head(nrows) if nrows else df
    raise ValueError(f"Unsupported extension: {ext}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--nrows", type=int, default=5)
    args = ap.parse_args()
    if not os.path.exists(args.path):
        print("NOT FOUND:", args.path); return
    df = load_any(args.path, nrows=args.nrows)
    print("FILE:", args.path)
    print("ROWS:", len(df), "COLS:", len(df.columns))
    print("COLUMNS:", list(df.columns))
    with pd.option_context("display.max_columns", 200, "display.width", 140):
        print(df.head(args.nrows).to_string(index=False))

if __name__ == "__main__":
    main()

