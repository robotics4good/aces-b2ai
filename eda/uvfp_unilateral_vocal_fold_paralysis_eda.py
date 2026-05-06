from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


DEFAULT_PATH = Path(
    "b2ai_adult_dataset/3.0.0/phenotype/diagnosis/unilateral_vocal_fold_paralysis.tsv"
)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    i = int(round((len(values) - 1) * p))
    return values[i]


def eda_stdlib(path: Path) -> None:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []

        n = 0
        pid_seen: set[str] = set()
        pid_dupes = 0

        empty_counts: Counter[str] = Counter()

        cat_cols = [
            "diagnosis_uvfp_ds",
            "diagnosis_uvfp_treatment",
            "diagnosis_uvfp_dse",
            "diagnosis_uvfp_etiology",
            "diagnosis_uvfp_gold_standard_diagnosis",
            "diagnosis_uvfp_iatrogenic",
            "diagnosis_uvfp_tumor",
        ]
        cat_cols = [c for c in cat_cols if c in cols]
        value_counts = {c: Counter() for c in cat_cols}

        flag_cols = [
            c
            for c in cols
            if (
                "treatment_select___" in c
                or "treatment_surgery___" in c
                or "treatment_surgery_thyroplasty___" in c
                or "treatment_surgery_vfia___" in c
            )
        ]
        flag_nonzero: Counter[str] = Counter()

        degree_cols = [
            c for c in cols if c.startswith("diagnosis_degree_") and not c.endswith("_2")
        ]
        degree_vals = {c: [] for c in degree_cols}

        def is_empty(v: str | None) -> bool:
            return v is None or v.strip() == ""

        for row in reader:
            n += 1
            for c in cols:
                if is_empty(row.get(c)):
                    empty_counts[c] += 1

            pid = (row.get("participant_id") or "").strip()
            if pid and pid in pid_seen:
                pid_dupes += 1
            pid_seen.add(pid)

            for c in cat_cols:
                v = (row.get(c) or "").strip()
                value_counts[c][v] += 1

            for c in flag_cols:
                v = (row.get(c) or "").strip()
                try:
                    x = float(v) if v else 0.0
                except ValueError:
                    x = 0.0
                if x != 0.0:
                    flag_nonzero[c] += 1

            for c in degree_cols:
                v = (row.get(c) or "").strip()
                if not v:
                    continue
                try:
                    degree_vals[c].append(float(v))
                except ValueError:
                    pass

    print(f"path: {path}")
    print(f"rows: {n}")
    print(f"cols: {len(cols)}")
    print(f"participant_id unique (incl blank as one value): {len(pid_seen)}")
    print(f"participant_id duplicates (non-blank): {pid_dupes}")

    miss = [(c, empty_counts[c] / n) for c in cols] if n else []
    miss.sort(key=lambda x: x[1], reverse=True)
    print("\nTop missingness (fraction empty):")
    for c, frac in miss[:20]:
        print(f"  {c}: {frac:.3f} ({empty_counts[c]}/{n})")

    print("\nKey categorical distributions (top values):")
    for c in cat_cols:
        ctr = value_counts[c]
        blanks = ctr.get("", 0)
        print(f"\n- {c} (blanks {blanks}/{n})")
        for val, k in ctr.most_common(8):
            label = val if val else "<blank>"
            print(f"    {label}: {k} ({k/n:.1%})")

    if flag_cols:
        prev = [(c, flag_nonzero[c] / n, flag_nonzero[c]) for c in flag_cols]
        prev.sort(key=lambda x: x[1], reverse=True)
        print("\nTreatment-related flags (non-zero prevalence):")
        for c, frac, k in prev[:25]:
            print(f"  {c}: {k} ({frac:.1%})")

    print("\nDegree numeric summaries (non-null):")
    for c, vals in degree_vals.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        print(
            f"  {c}: n={len(vals)}, mean={mean:.2f}, min={min(vals):.2f}, "
            f"p50={_pct(vals, 0.50):.2f}, p95={_pct(vals, 0.95):.2f}, max={max(vals):.2f}"
        )


def eda_pandas(path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    print("path:", path)
    print("rows:", len(df), "cols:", df.shape[1])
    print("\nparticipant_id:")
    print("  nulls:", int(df["participant_id"].isna().sum()))
    print("  unique:", int(df["participant_id"].nunique(dropna=True)))
    print("  duplicates:", int(df["participant_id"].duplicated().sum()))

    missing_frac = df.isna().mean().sort_values(ascending=False)
    print("\nTop missingness (fraction empty):")
    print(missing_frac.head(25).to_string())

    for col in [
        "diagnosis_uvfp_ds",
        "diagnosis_uvfp_treatment",
        "diagnosis_uvfp_dse",
        "diagnosis_uvfp_etiology",
        "diagnosis_uvfp_gold_standard_diagnosis",
        "diagnosis_uvfp_iatrogenic",
        "diagnosis_uvfp_tumor",
    ]:
        if col not in df.columns:
            continue
        vc = df[col].fillna("<NA>").value_counts(dropna=False)
        print(f"\n{col}:")
        print(vc.head(15).to_string())

    degree_cols = [c for c in df.columns if c.startswith("diagnosis_degree_")]
    deg = df[degree_cols].apply(pd.to_numeric, errors="coerce")
    print("\nDegree numeric summary:")
    print(deg.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"File not found: {args.path}")

    try:
        import pandas  # noqa: F401

        eda_pandas(args.path)
    except Exception:
        eda_stdlib(args.path)


if __name__ == "__main__":
    main()

