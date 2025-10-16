import argparse
import glob
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from src.bars.dollar_bars import build_dollar_bars

ALIASES: Dict[str, Iterable[str]] = {
    "timestamp": ("timestamp", "ts", "time", "t"),
    "price": ("price", "p"),
    "qty": ("qty", "quantity", "size", "q", "base_qty", "amount_base", "volume"),
    "quote_qty": ("quote_qty", "qv", "amount", "amount_quote", "quote_volume", "quotevolume"),
    "is_buyer_maker": (
        "is_buyer_maker",
        "isbuyer_maker",
        "isbuyer",
        "buyer_maker",
        "isbm",
        "maker_is_buyer",
        "isbuyermaker",
    ),
    "side": ("side", "taker_side", "trade_side"),
}

SELL_MARKERS = {"sell", "ask", "offer", "short", "s", "-1", "sellshort", "sell-side", "sell_side"}
TRUE_MARKERS = {"true", "1", "yes", "y", "t"}
FALSE_MARKERS = {"false", "0", "no", "n", "f"}


def _column_lookup(columns: Iterable[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for col in columns:
        key = col.strip().lower()
        if key not in lookup:
            lookup[key] = col
    return lookup


def _resolve_columns(
    lookup: Dict[str, str], overrides: Dict[str, str]
) -> Tuple[Dict[str, Optional[str]], Dict[str, str]]:
    resolved: Dict[str, Optional[str]] = {
        "timestamp": None,
        "price": None,
        "qty": None,
        "quote_qty": None,
        "is_buyer_maker": None,
        "side": None,
    }
    notes: Dict[str, str] = {}

    for canonical in ("timestamp", "price", "qty", "quote_qty", "is_buyer_maker", "side"):
        override_name = overrides.get(canonical)
        if override_name:
            key = override_name.strip().lower()
            if key not in lookup:
                raise ValueError(f"Override for {canonical!r}='{override_name}' not found in columns.")
            resolved[canonical] = lookup[key]
            notes[canonical] = f"{lookup[key]} (override)"
            continue

        for alias in ALIASES.get(canonical, ()):
            if alias in lookup:
                resolved[canonical] = lookup[alias]
                notes[canonical] = lookup[alias]
                break

    if resolved["timestamp"] is None:
        raise ValueError("Missing required timestamp column; provide an override with --col-timestamp.")
    if resolved["price"] is None:
        raise ValueError("Missing required price column; provide an override with --col-price.")
    if resolved["qty"] is None and resolved["quote_qty"] is None:
        raise ValueError(
            "Missing qty and quote_qty columns; provide an override with --col-qty or --col-quote-qty."
        )

    return resolved, notes


def _coerce_timestamp_to_ms(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_valid = numeric.notna()
    if numeric_valid.sum() >= max(1, int(0.5 * len(series))):
        values = numeric
        med = values.loc[numeric_valid].median()
        if pd.isna(med):
            med = 0.0
        if med < 1e12:
            values = values * 1000.0
        elif 1e15 < med < 1e18:
            values = values / 1000.0
        elif med >= 1e18:
            values = values / 1_000_000.0

        if numeric_valid.sum() != len(series):
            fallback = pd.to_datetime(series.loc[~numeric_valid], utc=True, errors="coerce")
            values.loc[~numeric_valid] = fallback.astype("int64") // 1_000_000
        return values

    datetimes = pd.to_datetime(series, utc=True, errors="coerce")
    return pd.Series(datetimes.astype("int64") // 1_000_000, index=series.index, dtype="float64")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_is_buyer_maker(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="int8")
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        return (numeric > 0).astype("int8")

    text = series.astype("string").str.strip().str.lower()
    mapped = pd.Series(0, index=series.index, dtype="int8")
    mapped[text.isin(TRUE_MARKERS)] = 1
    mapped[text.isin(FALSE_MARKERS)] = 0
    return mapped


def _approx_is_buyer_maker_from_side(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    result = np.zeros(len(series), dtype="int8")
    result[text.isin(SELL_MARKERS)] = 1
    return pd.Series(result, index=series.index, dtype="int8")


def _normalize_chunk(chunk: pd.DataFrame, resolved: Dict[str, Optional[str]]) -> pd.DataFrame:
    rename_map = {
        source: canonical
        for canonical, source in resolved.items()
        if canonical in ("timestamp", "price", "qty", "quote_qty", "is_buyer_maker", "side") and source
    }
    chunk = chunk.rename(columns=rename_map)

    keep_cols = ["timestamp", "price"]
    for optional in ("qty", "quote_qty", "is_buyer_maker", "side"):
        if optional in chunk.columns:
            keep_cols.append(optional)
    chunk = chunk[keep_cols].copy()

    chunk["timestamp"] = _coerce_timestamp_to_ms(chunk["timestamp"])
    chunk["price"] = _coerce_numeric(chunk["price"])

    if "qty" in chunk.columns:
        chunk["qty"] = _coerce_numeric(chunk["qty"])
    if "quote_qty" in chunk.columns:
        chunk["quote_qty"] = _coerce_numeric(chunk["quote_qty"])
        if "qty" not in chunk.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                chunk["qty"] = chunk["quote_qty"] / chunk["price"]

    if "is_buyer_maker" in chunk.columns:
        chunk["is_buyer_maker"] = _coerce_is_buyer_maker(chunk["is_buyer_maker"])

    if "is_buyer_maker" not in chunk.columns or chunk["is_buyer_maker"].isna().all():
        if "side" in chunk.columns:
            chunk["is_buyer_maker"] = _approx_is_buyer_maker_from_side(chunk["side"])
        else:
            chunk["is_buyer_maker"] = 0
    chunk["is_buyer_maker"] = chunk["is_buyer_maker"].fillna(0).astype("int8")

    if "side" in chunk.columns:
        chunk = chunk.drop(columns=["side"])
    if "quote_qty" in chunk.columns:
        chunk = chunk.drop(columns=["quote_qty"])

    mask = (
        chunk["timestamp"].notna()
        & chunk["price"].notna()
        & chunk["qty"].notna()
        & (chunk["price"] > 0)
        & (chunk["qty"] > 0)
    )
    chunk = chunk.loc[mask]
    if chunk.empty:
        return pd.DataFrame(columns=["timestamp", "price", "qty", "is_buyer_maker"])

    chunk["timestamp"] = chunk["timestamp"].round().astype("int64")
    chunk["price"] = chunk["price"].astype("float32")
    chunk["qty"] = chunk["qty"].astype("float32")
    chunk["is_buyer_maker"] = chunk["is_buyer_maker"].astype("int8")

    return chunk[["timestamp", "price", "qty", "is_buyer_maker"]]


def load_ticks_flexible(
    path: str,
    *,
    overrides: Optional[Dict[str, str]] = None,
    chunk_rows: Optional[int] = None,
    engine: str = "pandas",
) -> pd.DataFrame:
    """
    Load a tick CSV with flexible schema and return canonical columns.

    Parameters
    ----------
    path:
        File path to the CSV.
    overrides:
        Optional mapping from canonical names to explicit column names.
    chunk_rows:
        Number of rows per chunk for incremental processing. If None, the entire file is read.
    engine:
        CSV engine to use. Currently only 'pandas' is supported.
    """
    if engine != "pandas":
        raise ValueError(f"Unsupported engine '{engine}'. Only 'pandas' is available.")

    overrides = overrides or {}
    # Use pandas' default CSV engine when our logical engine is 'pandas'.
    _engine = None if engine == "pandas" else engine
    if _engine is None:
        header = pd.read_csv(path, nrows=0)
    else:
        header = pd.read_csv(path, nrows=0, engine=_engine)
    lookup = _column_lookup(header.columns)
    resolved, notes = _resolve_columns(lookup, overrides)

    read_kwargs = {}
    if _engine is not None:
        read_kwargs["engine"] = _engine
    if chunk_rows:
        read_kwargs["chunksize"] = int(chunk_rows)
    reader = pd.read_csv(path, **read_kwargs)

    if chunk_rows:
        chunks = (chunk for chunk in reader)
    else:
        chunks = (reader,)

    frames = []
    total_rows = 0
    for chunk in chunks:
        total_rows += len(chunk)
        normalised = _normalize_chunk(chunk, resolved)
        if not normalised.empty:
            frames.append(normalised)

    if not frames:
        df = pd.DataFrame(columns=["timestamp", "price", "qty", "is_buyer_maker"])
    else:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    qty_note = notes.get("qty")
    if resolved["qty"] is None and resolved["quote_qty"]:
        qty_note = f"{resolved['quote_qty']} (quote/price)"
    ibm_note = notes.get("is_buyer_maker")
    if not ibm_note:
        if resolved["side"]:
            ibm_note = f"{resolved['side']} (approx side)"
        else:
            ibm_note = "filled with 0"

    log_parts = [
        f"timestamp<-{notes.get('timestamp', resolved['timestamp'])}",
        f"price<-{notes.get('price', resolved['price'])}",
        f"qty<-{qty_note}",
        f"is_buyer_maker<-{ibm_note}",
        f"rows_kept={len(df):,}/{total_rows:,}",
    ]
    tqdm.write(f"[INFO] {os.path.basename(path)}: " + ", ".join(filter(None, log_parts)))
    return df


def run_synthetic_smoke_tests(dollar_value: float) -> None:
    from tempfile import TemporaryDirectory

    schemas = [
        (
            "basic.csv",
            pd.DataFrame(
                {
                    "timestamp": [1700000000, 1700000001, 1700000002],
                    "price": [100.0, 101.0, 102.0],
                    "qty": [12.5, 10.0, 9.5],
                    "is_buyer_maker": [1, 0, 1],
                }
            ),
        ),
        (
            "alt_names.csv",
            pd.DataFrame(
                {
                    "ts": [1700001000, 1700001001, 1700001002],
                    "p": [150.0, 149.5, 150.5],
                    "quantity": [8.0, 7.5, 8.5],
                    "isbuyer": ["true", "false", "true"],
                }
            ),
        ),
        (
            "quote_side.csv",
            pd.DataFrame(
                {
                    "time": [1700002000, 1700002001, 1700002002],
                    "price": [200.0, 201.0, 202.0],
                    "quote_qty": [2200.0, 900.0, 950.0],
                    "side": ["sell", "buy", "SELL"],
                }
            ),
        ),
    ]

    with TemporaryDirectory() as tmpdir:
        for name, df in schemas:
            path = os.path.join(tmpdir, name)
            df.to_csv(path, index=False)
            ticks = load_ticks_flexible(path)
            if ticks.empty:
                raise AssertionError(f"Smoke test {name} produced no ticks.")
            bars = build_dollar_bars(ticks, dollar_value=dollar_value)
            if bars.empty:
                raise AssertionError(f"Smoke test {name} produced no dollar bars.")
        tqdm.write("[SMOKE] Synthetic schema tests passed.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks_glob", help="Glob pattern for tick CSV files.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/bars.yaml")
    ap.add_argument("--bar_type", default="dollar", choices=["dollar"])
    ap.add_argument("--dollar_value", type=float, default=None)
    ap.add_argument("--adapt_dollar_monthly", action="store_true", help="Adapt dollar_value monthly to target bars/day.")
    ap.add_argument("--target_bars_per_day", type=float, default=None, help="Target bars/day when adapting.")
    ap.add_argument("--clip_factor", type=float, default=2.0, help="Cap monthly dollar_value change by this factor.")
    ap.add_argument("--engine", default="pandas", choices=["pandas"], help="CSV engine to use.")
    ap.add_argument("--chunk_rows", type=int, default=None, help="Optional chunk size for CSV loading.")
    ap.add_argument("--col-timestamp", dest="col_timestamp", help="Override column name for timestamp.")
    ap.add_argument("--col-price", dest="col_price", help="Override column name for price.")
    ap.add_argument("--col-qty", dest="col_qty", help="Override column name for base quantity.")
    ap.add_argument("--col-quote-qty", dest="col_quote_qty", help="Override column name for quote quantity.")
    ap.add_argument("--col-ibm", dest="col_ibm", help="Override column name for is_buyer_maker flag.")
    ap.add_argument("--col-side", dest="col_side", help="Override column name for trade side.")
    ap.add_argument(
        "--run-smoke-tests",
        action="store_true",
        help="Generate synthetic CSVs and validate loader/bar builder.",
    )
    args = ap.parse_args()

    if args.run_smoke_tests:
        dv = args.dollar_value if args.dollar_value else 1000.0
        run_synthetic_smoke_tests(dv)
        return

    if not args.ticks_glob:
        ap.error("--ticks_glob is required unless --run-smoke-tests is set.")

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(args.ticks_glob))
    if not files:
        tqdm.write("No tick files found.")
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

    threshold = args.dollar_value if args.dollar_value else float(cfg.get("dollar_value", 5e5))
    # YAML-based adapt block
    adapt_cfg = (cfg.get("adapt_dollar_value") or {}) if isinstance(cfg, dict) else {}
    adapt_flag = bool(args.adapt_dollar_monthly or adapt_cfg.get("enabled", False))
    target_bpd = float(args.target_bars_per_day or adapt_cfg.get("target_bars_per_day", 0) or 0)
    clip_factor = float(args.clip_factor or adapt_cfg.get("clip_factor", 2.0))

    def _estimate_dollar_value_for_file(csv_path: str, fallback: float) -> float:
        if not adapt_flag or target_bpd <= 0:
            return fallback
        try:
            # Light scan of file for daily notional
            tmp = pd.read_csv(csv_path, usecols=None)
            # Best-effort column mapping
            lookup = _column_lookup(tmp.columns)
            ts_col = lookup.get("timestamp") or next((lookup[a] for a in ALIASES["timestamp"] if a in lookup), None)
            p_col = lookup.get("price") or next((lookup[a] for a in ALIASES["price"] if a in lookup), None)
            q_col = lookup.get("qty") or next((lookup[a] for a in ALIASES["qty"] if a in lookup), None)
            if not (ts_col and p_col and q_col):
                return fallback
            ts = tmp[ts_col]
            vmax = int(pd.to_numeric(ts, errors="coerce").max() or 0)
            unit = "ns" if vmax > 10 ** 14 else ("ms" if vmax > 10 ** 12 else "s")
            t = pd.to_datetime(ts, unit=unit, utc=True)
            notional = pd.to_numeric(tmp[p_col], errors="coerce").fillna(0) * pd.to_numeric(tmp[q_col], errors="coerce").fillna(0)
            daily = notional.groupby(t.dt.date).sum()
            base = float(daily.median()) if len(daily) else 0.0
            dv = base / float(target_bpd) if target_bpd > 0 else base
            # Clip vs provided fallback
            lo, hi = fallback / clip_factor, fallback * clip_factor
            if dv <= 0:
                return fallback
            return float(max(min(dv, hi), lo))
        except Exception:
            return fallback
    total_files = len(files)

    for idx, path in enumerate(tqdm(files, desc="Building bars", unit="file"), start=1):
        sym = os.path.splitext(os.path.basename(path))[0]
        outp = os.path.join(args.out_dir, f"{sym}.parquet")
        if os.path.exists(outp) and os.path.getsize(outp) > 0:
            # Enforce unique timestamps if an existing file has duplicates (monotonic bump)
            try:
                df_existing = pd.read_parquet(outp)
                if "timestamp" in df_existing.columns:
                    ts = pd.to_numeric(df_existing["timestamp"], errors="coerce").astype("int64").to_numpy()
                    last = np.int64(-(2**62))
                    changed = False
                    for i in range(len(ts)):
                        if ts[i] <= last:
                            ts[i] = last + 1
                            changed = True
                        last = ts[i]
                    if changed:
                        df_existing["timestamp"] = ts.astype("int64")
                        df_existing.to_parquet(outp, index=False)
                        tqdm.write(f"[FIX] Deduped timestamps (monotonic) in {outp}")
            except Exception:
                pass
            tqdm.write(f"[SKIP] {os.path.basename(path)} -> {outp} (exists)")
            continue
        tqdm.write(f"Processing {os.path.basename(path)} [{idx}/{total_files}]")
        try:
            ticks = load_ticks_flexible(
                path,
                overrides=overrides,
                chunk_rows=args.chunk_rows,
                engine=args.engine,
            )
        except ValueError as exc:
            tqdm.write(f"[ERROR] {os.path.basename(path)}: {exc}")
            continue

        if ticks.empty:
            tqdm.write(f"No valid ticks for {path}.")
            continue

        dv_use = _estimate_dollar_value_for_file(path, threshold)
        bars = build_dollar_bars(ticks, dollar_value=dv_use)
        if bars.empty:
            tqdm.write(f"No bars for {path}")
            continue

        cols = ["timestamp", "open", "high", "low", "close"]
        for _c in ("qty", "notional"):
            if _c in bars.columns and _c not in cols:
                cols.append(_c)
        bars = bars[cols].copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").astype("int64") // 1_000_000
        # Enforce per-file unique timestamps (ms); bump duplicates by +1, +2, ... ms
        ts = pd.to_numeric(bars["timestamp"], errors="coerce").astype("int64").to_numpy()
        last = np.int64(-(2**62))
        changed = False
        for i in range(len(ts)):
            if ts[i] <= last:
                ts[i] = last + 1
                changed = True
            last = ts[i]
        if changed:
            bars["timestamp"] = ts.astype("int64")
        for col in ("open", "high", "low", "close"):
            bars[col] = bars[col].astype("float32")

        bars.to_parquet(outp, index=False)
        tqdm.write(f"Wrote {outp} {len(bars)}")


if __name__ == "__main__":
    main()
