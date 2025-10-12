import argparse
import glob
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import yaml


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize label distributions and termination reasons from Parquet label files."
        )
    )
    ap.add_argument(
        "--labels_glob",
        required=True,
        help="Glob pattern for label parquet files (e.g., data/labels/*_labels.parquet)",
    )
    ap.add_argument(
        "--config",
        default="configs/tbm.yaml",
        help="YAML config path for label_classes (default: configs/tbm.yaml)",
    )
    ap.add_argument(
        "--save_csv",
        default=None,
        help="Optional path to save per-file breakdown CSV",
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="Optional path to save JSON summary (global, per_file, per_symbol, class_weights)",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress normal output to stdout")
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose errors with stack traces on unexpected exceptions",
    )
    return ap.parse_args()


def load_label_classes(config_path: str) -> List[int]:
    default = [1, 0, -1]
    try:
        if not os.path.exists(config_path):
            return default
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        classes = cfg.get("label_classes", default)
        # Normalize to list of ints if possible
        if not isinstance(classes, list):
            return default
        try:
            return [int(x) for x in classes]
        except Exception:
            return default
    except Exception:
        # On any config read issue, just fall back to default
        return default


def display_label(v: int) -> str:
    try:
        vi = int(v)
    except Exception:
        return str(v)
    if vi > 0:
        return f"+{vi}"
    return str(vi)


def symbol_from_path(path: str) -> str:
    stem = Path(path).stem  # e.g., BTCUSDT-ticks-2025-01_labels
    # symbol is before the first '-' in the stem
    return stem.split("-", 1)[0]


def read_labels_parquet(path: str) -> Tuple[pd.DataFrame, bool]:
    """Read parquet attempting to load only needed columns.

    Returns (df, term_available) where term_available indicates whether
    pt_hit and sl_hit columns are present.
    """
    # Try reading with columns including termination columns
    try:
        df = pd.read_parquet(path, columns=["label", "pt_hit", "sl_hit"])
        # Some engines might return successfully but columns missing -> guard
        term_available = all(c in df.columns for c in ["pt_hit", "sl_hit"])
        if not term_available:
            # Fallback to label only
            df = pd.read_parquet(path, columns=["label"])  # type: ignore
        return df, term_available
    except Exception as e_all:
        # Fallback: try just label
        try:
            df = pd.read_parquet(path, columns=["label"])  # type: ignore
            return df, False
        except Exception as e_label:
            # Re-raise the original more informative exception
            raise e_all


def format_count(n: int) -> str:
    return f"{int(n):,}"


def format_pct(p: float) -> str:
    # 0.421 -> 42.10%
    return f"{p * 100:.2f}%"


def format_ratio(p: float) -> str:
    # 0.421 -> 0.42
    return f"{p:.2f}"


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def summarize(labels_glob: str, label_classes: List[int], quiet: bool) -> Dict[str, Any]:
    files = sorted(glob.glob(labels_glob))
    if not files:
        eprint(f"ERROR: No label files matched pattern: {labels_glob}")
        sys.exit(2)

    processed = 0
    any_term_missing = False
    all_term_missing = True

    # Aggregators
    global_known_counts = Counter()  # counts for known classes only
    global_unknown_counts = Counter()  # counts for unknown classes
    global_N = 0
    term_counts_global = Counter(pt=0, sl=0, timeout=0)
    term_N_global = 0

    per_file_rows = []
    # For symbol summary aggregators
    symbol_known_counts: Dict[str, Counter] = defaultdict(Counter)
    symbol_N: Dict[str, int] = defaultdict(int)

    for path in files:
        try:
            df, term_available = read_labels_parquet(path)
        except Exception as e:
            eprint(f"ERROR: Failed to read {path}: {e}")
            continue

        processed += 1
        if not term_available:
            any_term_missing = True
        else:
            all_term_missing = False

        # Ensure label column exists
        if "label" not in df.columns:
            eprint(f"ERROR: File missing 'label' column: {path}")
            continue

        labels = df["label"]
        # Compute per-file counts
        counts = labels.value_counts(dropna=False).to_dict()
        known_counts = {int(c): int(counts.get(c, 0)) for c in label_classes}
        # Safely detect unknown labels (including possible NaN or non-int types)
        unknown_counts: Dict[Any, int] = {}
        for k, v in counts.items():
            try:
                kk = int(k)
            except Exception:
                kk = k
            if kk not in label_classes:
                unknown_counts[kk] = int(v)

        # Update global
        for k, v in known_counts.items():
            global_known_counts[k] += v
        for k, v in unknown_counts.items():
            try:
                kk = int(k)
            except Exception:
                kk = k  # keep as-is if non-int
            global_unknown_counts[kk] += v
        N = int(len(labels))
        global_N += N

        # Termination per-file
        pt = sl = timeout = None
        pt_r = sl_r = to_r = None
        if term_available:
            pt = int((df["pt_hit"] != -1).sum())
            sl = int((df["sl_hit"] != -1).sum())
            timeout = int(((df["pt_hit"] == -1) & (df["sl_hit"] == -1)).sum())
            # Accumulate global termination (only from files with columns)
            term_counts_global["pt"] += pt
            term_counts_global["sl"] += sl
            term_counts_global["timeout"] += timeout
            term_N_global += N
            # Ratios
            if N > 0:
                pt_r = pt / N
                sl_r = sl / N
                to_r = timeout / N

        # Build per-file row
        file_stem = Path(path).stem
        row: Dict[str, Any] = {"file": file_stem}
        for cls in label_classes:
            row[display_label(cls)] = int(known_counts.get(cls, 0))
        row.update(
            {
                "PT%": (format_ratio(pt_r) if pt_r is not None else ""),
                "SL%": (format_ratio(sl_r) if sl_r is not None else ""),
                "Timeout%": (format_ratio(to_r) if to_r is not None else ""),
                "N": N,
            }
        )
        per_file_rows.append(row)

        # Symbol aggregations
        sym = symbol_from_path(path)
        for cls in label_classes:
            symbol_known_counts[sym][cls] += int(known_counts.get(cls, 0))
        symbol_N[sym] += N

    if processed == 0:
        eprint(
            "ERROR: All files failed to read; no statistics could be produced."
        )
        sys.exit(3)

    # Prepare outputs
    per_file_df = pd.DataFrame(per_file_rows)
    # Stable column order: file, each class in order, PT%, SL%, Timeout%, N
    class_cols = [display_label(c) for c in label_classes]
    ordered_cols = ["file"] + class_cols + ["PT%", "SL%", "Timeout%", "N"]
    # Some columns may be missing if label_classes changed; guard
    existing_cols = [c for c in ordered_cols if c in per_file_df.columns]
    per_file_df = per_file_df[existing_cols]

    # Build per-symbol summary DataFrame
    sym_rows = []
    for sym in sorted(symbol_N.keys()):
        N = symbol_N[sym]
        row: Dict[str, Any] = {"symbol": sym, "N": N}
        for cls in label_classes:
            cnt = int(symbol_known_counts[sym].get(cls, 0))
            row[display_label(cls)] = cnt
            row[f"{display_label(cls)}%"] = format_ratio(cnt / N) if N > 0 else "0.00"
        sym_rows.append(row)
    per_symbol_df = pd.DataFrame(sym_rows)
    # Stable per-symbol column order
    sym_cols: List[str] = ["symbol", "N"]
    for cls in label_classes:
        sym_cols.append(display_label(cls))
        sym_cols.append(f"{display_label(cls)}%")
    per_symbol_df = per_symbol_df[sym_cols] if not per_symbol_df.empty else per_symbol_df

    # Unknown label warning (aggregate)
    had_unknown = len(global_unknown_counts) > 0

    # Frequencies and weights (known classes only)
    known_total = sum(global_known_counts.get(c, 0) for c in label_classes)
    freqs: List[float] = [
        (global_known_counts.get(c, 0) / known_total) if known_total > 0 else 0.0
        for c in label_classes
    ]
    inv = [1.0 / f if f > 0 else 0.0 for f in freqs]
    inv_sum = sum(inv)
    weights = [w / inv_sum if inv_sum > 0 else 0.0 for w in inv]

    # Prepare JSON structure
    json_payload: Dict[str, Any] = {
        "global": {
            "classes": label_classes,
            "counts": {int(c): int(global_known_counts.get(c, 0)) for c in label_classes},
            "percent": {int(c): (global_known_counts.get(c, 0) / known_total if known_total > 0 else 0.0) for c in label_classes},
            "unknown_counts": {str(k): int(v) for k, v in sorted(global_unknown_counts.items(), key=lambda kv: str(kv[0]))},
        },
        "per_file": per_file_rows,
        "per_symbol": sym_rows,
        "class_weights": {
            "classes": label_classes,
            "freq": freqs,
            "weights_norm": weights,
        },
    }

    # Add termination to JSON if any available
    if term_N_global > 0:
        json_payload["global"]["termination"] = {
            "counts": {
                "pt": int(term_counts_global["pt"]),
                "sl": int(term_counts_global["sl"]),
                "timeout": int(term_counts_global["timeout"]),
            },
            "percent": {
                "pt": (term_counts_global["pt"] / term_N_global),
                "sl": (term_counts_global["sl"] / term_N_global),
                "timeout": (term_counts_global["timeout"] / term_N_global),
            },
            "N": int(term_N_global),
        }
    else:
        json_payload["global"]["termination"] = None

    # Printing (respect --quiet for normal output)
    if not quiet:
        # Global label distribution
        print("=== GLOBAL LABEL DISTRIBUTION ===")
        for cls in label_classes:
            cnt = int(global_known_counts.get(cls, 0))
            pct = (cnt / known_total) if known_total > 0 else 0.0
            print(f"label {cls:>2}: {format_count(cnt):>10}  ({format_pct(pct)})")
        print()

        # Global termination reasons
        print("=== GLOBAL TERMINATION REASONS ===")
        if term_N_global == 0:
            print("(not available)")
        else:
            pt = int(term_counts_global["pt"]) ; sl = int(term_counts_global["sl"]) ; to = int(term_counts_global["timeout"])  # noqa: E702
            print(f"     pt: {format_count(pt):>10}  ({format_pct(pt/term_N_global)})")
            print(f"     sl: {format_count(sl):>10}  ({format_pct(sl/term_N_global)})")
            print(f"timeout: {format_count(to):>10}  ({format_pct(to/term_N_global)})")
        print()

        # Per-file breakdown
        print("=== PER-FILE BREAKDOWN ===")
        if not per_file_df.empty:
            # Ensure consistent dtypes for pretty printing
            print(per_file_df.to_string(index=False))
        else:
            print("(no data)")
        print()

        # Per-symbol summary
        print("=== PER-SYMBOL SUMMARY ===")
        if not per_symbol_df.empty:
            print(per_symbol_df.to_string(index=False))
        else:
            print("(no data)")
        print()

        # Class order, freq, weights
        print(f"Class order: {label_classes}")
        print(
            "Freq: [" + ", ".join(f"{x:.3f}" for x in freqs) + "]"
        )
        print(
            "Weights (norm.): [" + ", ".join(f"{x:.3f}" for x in weights) + "]"
        )

    # Warnings (print regardless of quiet)
    if had_unknown:
        # Present unknowns with counts
        unknown_str_items = [f"{k}: {v}" for k, v in sorted(global_unknown_counts.items(), key=lambda kv: str(kv[0]))]
        eprint("WARNING: Unknown labels found: {" + ", ".join(unknown_str_items) + "}")

    if any_term_missing and not all_term_missing and not quiet:
        eprint(
            "INFO: Some files are missing pt_hit/sl_hit; global termination uses available files only."
        )
    if all_term_missing and not quiet:
        eprint(
            "INFO: No files contain pt_hit/sl_hit columns; termination breakdown not available."
        )

    return {
        "per_file_df": per_file_df,
        "per_symbol_df": per_symbol_df,
        "json_payload": json_payload,
    }


def main():
    args = parse_args()
    try:
        label_classes = load_label_classes(args.config)
        results = summarize(args.labels_glob, label_classes, args.quiet)

        # Save CSV if requested
        if args.save_csv:
            try:
                ensure_parent_dir(args.save_csv)
                # Save as numeric where possible; per-file df may contain formatted strings for % columns
                results["per_file_df"].to_csv(args.save_csv, index=False)
            except Exception as e:
                eprint(f"ERROR: Failed to write CSV to {args.save_csv}: {e}")

        # Save JSON if requested
        if args.json_out:
            try:
                ensure_parent_dir(args.json_out)
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump(results["json_payload"], f, indent=2)
            except Exception as e:
                eprint(f"ERROR: Failed to write JSON to {args.json_out}: {e}")

        # Success if we reached here with some processed files
        sys.exit(0)

    except SystemExit:
        # Already handled exit codes for expected conditions
        raise
    except Exception as e:
        eprint(f"ERROR: Unexpected exception: {e}")
        if getattr(args, "verbose", False):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
