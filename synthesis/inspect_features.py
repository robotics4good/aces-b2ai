"""
Inspect a handful of columns in `features/static_features.tsv`.

Run (recommended):
  ./.venv/bin/python3 synthesis/inspect_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_ROOT = Path("b2ai_adult_dataset/3.0.0")
DEFAULT_STATIC_TSV = DEFAULT_DATASET_ROOT / "features/static_features.tsv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--static-tsv", type=Path, default=DEFAULT_STATIC_TSV)
    args = p.parse_args()

    path: Path = args.static_tsv
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    df = pd.read_csv(path, sep="\t")
    print(f"Loaded: {path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")

    cols_of_interest = [
        "jitterLocal_sma3nz_amean",
        "shimmerLocaldB_sma3nz_amean",
        "HNRdBACF_sma3nz_amean",
        "F0semitoneFrom27.5Hz_sma3nz_amean",  # F0 in semitones — important
    ]

    # Print which of these columns actually exist
    print("\n=== COLUMN CHECK ===")
    for col in cols_of_interest:
        print(f"  {col}: {'EXISTS' if col in df.columns else 'MISSING'}")

    # Candidate label columns (usually none in this table)
    print("\n=== DIAGNOSIS COLUMN ===")
    diag_cols = [
        c
        for c in df.columns
        if ("diag" in c.lower() or "condition" in c.lower() or "label" in c.lower())
    ]
    print(f"  Candidate diagnosis columns: {diag_cols}")

    existing = [c for c in cols_of_interest if c in df.columns]
    if not existing:
        print("\nNo requested columns exist in this TSV.")
        return

    # Coerce to numeric for stats (non-numeric becomes NaN)
    x = df[existing].apply(pd.to_numeric, errors="coerce")

    print("\n=== FEATURE DESCRIPTIVE STATS (numeric-coerced) ===")
    print(x.describe().to_string())

    print("\n=== FIRST 5 RAW VALUES ===")
    print(df[existing].head().to_string())

    print("\n=== NaN RATES (after numeric coercion) ===")
    for col in existing:
        nan_rate = float(x[col].isna().mean())
        print(f"  {col}: {nan_rate:.1%} NaN")


if __name__ == "__main__":
    main()