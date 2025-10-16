import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import glob
import os
from typing import Dict

import pandas as pd
from tqdm.auto import tqdm

from scripts.make_bars import load_ticks_flexible


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks_glob", required=True, help="Glob pattern for tick CSV files.")
    ap.add_argument("--engine", default="pandas", choices=["pandas"], help="CSV engine to use.")
    ap.add_argument("--chunk_rows", type=int, default=None, help="Optional chunk size for loader.")
    ap.add_argument("--col-timestamp", dest="col_timestamp", help="Override column name for timestamp.")
    ap.add_argument("--col-price", dest="col_price", help="Override column name for price.")
    ap.add_argument("--col-qty", dest="col_qty", help="Override column name for base quantity.")
    ap.add_argument("--col-quote-qty", dest="col_quote_qty", help="Override column name for quote quantity.")
    ap.add_argument("--col-ibm", dest="col_ibm", help="Override column name for is_buyer_maker flag.")
    ap.add_argument("--col-side", dest="col_side", help="Override column name for trade side.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.ticks_glob))
    if not files:
        tqdm.write("No tick files matched.")
        return

    overrides: Dict[str, str] = {}
    if args.col_timestamp:
        overrides["timestamp"] = args.col_timestamp
    if args.col_price:
        overrides["price"] = args.col_price
    if args.col_qty:
        overrides["qty"] = args.col_qty
    if args.col_quote_qty:
        overrides["quote_qty"] = args.col_quote_qty
    if args.col_ibm:
        overrides["is_buyer_maker"] = args.col_ibm
    if args.col_side:
        overrides["side"] = args.col_side

    for path in files:
        header = pd.read_csv(path, nrows=0)
        tqdm.write(f"[AUDIT] {os.path.basename(path)} columns: {list(header.columns)}")
        try:
            load_ticks_flexible(
                path,
                overrides=overrides,
                chunk_rows=args.chunk_rows,
                engine=args.engine,
            )
        except ValueError as exc:
            tqdm.write(f"[ERROR] {os.path.basename(path)}: {exc}")


if __name__ == "__main__":
    main()
