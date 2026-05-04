#!/usr/bin/env python3
"""Build a training manifest (CSV) for binary Yes/No classification.

Each row is one (participant_id, task_name) clip that has extracted features.
The final column ``label`` is 1 = positive class, 0 = negative class,
chosen by ``--target``.

Usage examples
--------------
List all available targets and their class balance::

    python scripts/build_training_manifest.py --list-configs

Voice disorder vs. healthy (no illness, ages 6-17)::

    python scripts/build_training_manifest.py \\
        --target voice_any \\
        --exclude-ill \\
        --age-min 6 --age-max 17 \\
        --tasks long-sounds \\
        --out /tmp/voice_manifest.csv

Cochlear implant vs. all others::

    python scripts/build_training_manifest.py \\
        --target hearing_cochlear \\
        --exclude-ill \\
        --out /tmp/cochlear_manifest.csv

ADHD vs. healthy only (exclude other psych conditions)::

    python scripts/build_training_manifest.py \\
        --target psych_adhd \\
        --negative-class healthy_only \\
        --exclude-ill \\
        --out /tmp/adhd_manifest.csv

Available targets
-----------------
Voice:      voice_any, voice_reflux, voice_nodules, voice_scarring,
            voice_paralysis, voice_stenosis_webs
Hearing:    hearing_any, hearing_cochlear, hearing_mild, hearing_deaf
Breathing:  breathing_asthma, breathing_stridor, breathing_apnea,
            breathing_intubated
Psychiatric: psych_adhd, psych_anxiety, psych_asd, psych_depression, psych_ocd
Other:      genetic_syndrome, chronic_condition, healthy_vs_all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths (can be overridden via --dataset-dir)
# ---------------------------------------------------------------------------

DEFAULT_DATASET_DIR = Path("~/Downloads/bridge2ai-voice-pediatric-dataset-1.0.0").expanduser()
DEFAULT_FEATURES_DIR = DEFAULT_DATASET_DIR / "features"


# ---------------------------------------------------------------------------
# Label definitions
# All targets are defined here — add new ones freely.
# ---------------------------------------------------------------------------

def _flag(mc: pd.DataFrame, col: str, val: str | None = None) -> pd.Series:
    """Return boolean Series for a binary Yes/No or substring-match column."""
    if col not in mc.columns:
        return pd.Series(False, index=mc.index)
    s = mc[col].fillna("")
    if val is None:
        return s.str.strip() == "Yes"
    return s.str.contains(val, case=False, na=False)


def build_label_matrix(mc: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of all binary condition columns for every participant.

    All columns are int (0/1) except ``participant_id``.
    """
    labels: dict[str, pd.Series] = {
        # ── Voice ────────────────────────────────────────────────────────────
        "voice_any":            _flag(mc, "peds_mc_v_dis"),
        "voice_reflux":         _flag(mc, "peds_mc_voice_disorders", "reflux")
                                | _flag(mc, "peds_mc_reflux"),
        "voice_nodules":        _flag(mc, "peds_mc_voice_disorders", "nodule|polyp"),
        "voice_scarring":       _flag(mc, "peds_mc_voice_disorders", "scarring"),
        "voice_paralysis":      _flag(mc, "peds_mc_voice_disorders", "paralysis"),
        "voice_stenosis_webs":  _flag(mc, "peds_mc_voice_disorders", "stenosis|web"),
        # ── Hearing ──────────────────────────────────────────────────────────
        "hearing_any":          _flag(mc, "peds_mc_hl"),
        "hearing_mild":         _flag(mc, "peds_mc_hearing_loss", "mild hearing loss"),
        "hearing_deaf":         _flag(mc, "peds_mc_hearing_loss", "deafness"),
        "hearing_cochlear":     _flag(mc, "peds_mc_hearing_loss", "cochlear"),
        "hearing_ear_tubes":    _flag(mc, "peds_mc_hearing_loss", "ear tubes"),
        "hearing_cholesteatoma":_flag(mc, "peds_mc_hearing_loss", "cholesteatoma"),
        # ── Breathing ────────────────────────────────────────────────────────
        "breathing_asthma":     _flag(mc, "peds_mc_breathing_conditions", "asthma"),
        "breathing_stridor":    _flag(mc, "peds_mc_stridor"),
        "breathing_apnea":      _flag(mc, "peds_mc_snoring_apnea", "obstructive sleep apnea"),
        "breathing_intubated":  _flag(mc, "peds_mc_breathing_conditions", "intubated"),
        "breathing_trach":      _flag(mc, "peds_mc_breathing_conditions", "tracheostomy"),
        "breathing_cough":      _flag(mc, "peds_mc_breathing_conditions", "cough"),
        # ── Psychiatric / neurodevelopmental ──────────────────────────────────
        "psych_adhd":           _flag(mc, "peds_mc_psych_disorders", "adhd"),
        "psych_anxiety":        _flag(mc, "peds_mc_psych_disorders", "anxiety"),
        "psych_asd":            _flag(mc, "peds_mc_psych_disorders", "asd"),
        "psych_depression":     _flag(mc, "peds_mc_psych_disorders", "depression"),
        "psych_ocd":            _flag(mc, "peds_mc_psych_disorders", "ocd"),
        "psych_eating_disorder":_flag(mc, "peds_mc_psych_disorders", "eating disorder"),
        # ── Neurological ─────────────────────────────────────────────────────
        "neuro_any":            _flag(mc, "peds_mc_neurological_disorders"),
        # ── Surgeries ────────────────────────────────────────────────────────
        "surgery_tonsil":       _flag(mc, "peds_mc_tonsillectomy"),
        "surgery_adenoid":      _flag(mc, "peds_mc_adenoidectomy"),
        "surgery_vocal_cord":   _flag(mc, "peds_mc_surgery_t_vc_a"),
        # ── Systemic ─────────────────────────────────────────────────────────
        "genetic_syndrome":     _flag(mc, "peds_mc_genetic_syndromes"),
        "chronic_condition":    _flag(mc, "peds_mc_chronic_medical_condition"),
        "congenital_heart":     _flag(mc, "peds_mc_conditions", "congenital heart"),
        "allergies":            _flag(mc, "peds_mc_allergies"),
        # ── Swallowing ───────────────────────────────────────────────────────
        "swallowing_difficulty":_flag(mc, "peds_mc_dif_swallowing"),
        "feeding_tube":         _flag(mc, "peds_mc_ft"),
        # ── Illness at recording (confound flag, not a target) ───────────────
        "ill_at_recording":     mc["peds_mc_l2w"].notna() & (mc["peds_mc_l2w"].str.strip() != ""),
    }

    df = pd.DataFrame({k: v.astype(int) for k, v in labels.items()})
    df.insert(0, "participant_id", mc["participant_id"].values)

    # Healthy: zero across all clinical condition columns (excluding ill_at_recording)
    clinical_cols = [c for c in labels if c != "ill_at_recording"]
    df["healthy_vs_all"] = (df[clinical_cols].sum(axis=1) == 0).astype(int)

    return df


# Human-readable descriptions for --list-configs
TARGET_DESCRIPTIONS: dict[str, str] = {
    "voice_any":             "Any voice disorder (GERD, nodules, paralysis, webs…)",
    "voice_reflux":          "Acid reflux / GERD affecting voice",
    "voice_nodules":         "Vocal nodules or polyps",
    "voice_scarring":        "Vocal cord scarring",
    "voice_paralysis":       "Vocal cord paralysis (uni- or bilateral)",
    "voice_stenosis_webs":   "Airway stenosis or vocal cord webs",
    "hearing_any":           "Any hearing impairment (peds_mc_hl = Yes)",
    "hearing_mild":          "Mild hearing loss specifically",
    "hearing_deaf":          "Deafness (severe/profound)",
    "hearing_cochlear":      "Cochlear implant user (uni- or bilateral)",
    "hearing_ear_tubes":     "Tympanostomy (ear tubes)",
    "hearing_cholesteatoma": "Cholesteatoma (destructive ear cyst)",
    "breathing_asthma":      "Asthma",
    "breathing_stridor":     "Stridor (structural airway noise)",
    "breathing_apnea":       "Obstructive sleep apnea (confirmed)",
    "breathing_intubated":   "History of intubation",
    "breathing_trach":       "Tracheostomy",
    "breathing_cough":       "Chronic cough",
    "psych_adhd":            "ADHD",
    "psych_anxiety":         "Anxiety disorder (any type)",
    "psych_asd":             "Autism Spectrum Disorder",
    "psych_depression":      "Depression",
    "psych_ocd":             "OCD",
    "psych_eating_disorder": "Eating disorder",
    "neuro_any":             "Any neurological disorder (n=2, too rare)",
    "surgery_tonsil":        "History of tonsillectomy",
    "surgery_adenoid":       "History of adenoidectomy",
    "surgery_vocal_cord":    "Vocal cord / airway surgery",
    "genetic_syndrome":      "Genetic syndrome (specific syndrome not recorded)",
    "chronic_condition":     "Any chronic medical condition",
    "congenital_heart":      "Congenital heart disease",
    "allergies":             "Allergies",
    "swallowing_difficulty": "Difficulty swallowing",
    "feeding_tube":          "Feeding tube (ever)",
    "healthy_vs_all":        "Healthy (no flags at all) — useful as negative class",
}


# ---------------------------------------------------------------------------
# Core manifest builder
# ---------------------------------------------------------------------------

def build_manifest(
    *,
    dataset_dir: Path,
    target: str,
    negative_class: str = "all",
    age_min: float | None = None,
    age_max: float | None = None,
    exclude_ill: bool = False,
    tasks: list[str] | None = None,
    feature_source: str = "sparc_ema",
) -> pd.DataFrame:
    """Build and return the training manifest DataFrame.

    Parameters
    ----------
    target:
        Column name in the label matrix to use as positive class (label=1).
    negative_class:
        ``"all"``          — everything not in positive class is label=0.
        ``"healthy_only"`` — only participants with zero clinical flags are label=0;
                             participants with other conditions are dropped.
    age_min / age_max:
        Inclusive age bounds for filtering participants.
    exclude_ill:
        Drop participants who had illness symptoms within 2 weeks of recording.
    tasks:
        List of task name prefixes to keep.  Prefix-matched.  None = all tasks.
    feature_source:
        Which parquet file to use for the clip roster (determines task_name rows).
        Use "sparc_ema", "torchaudio_mel_spectrogram", etc.
    """
    phenotype_dir = dataset_dir / "phenotype"
    features_dir  = dataset_dir / "features"

    # ── 1. Load phenotype ────────────────────────────────────────────────────
    mc  = pd.read_csv(phenotype_dir / "pediatric/pediatric_medical_conditions.tsv",
                      sep="\t", dtype=str)
    dem = pd.read_csv(phenotype_dir / "pediatric/pediatric_demographics.tsv",
                      sep="\t", dtype=str)

    # De-duplicate mc: keep most informative row per participant
    def _rank(row: pd.Series) -> int:
        if pd.notna(row.get("peds_mc_voice_disorders")) and \
                str(row["peds_mc_voice_disorders"]).strip() not in ("", "nan"):
            return 0
        if str(row.get("peds_mc_v_dis", "No")).strip() == "Yes":
            return 1
        return 2

    mc["_rank"] = mc.apply(_rank, axis=1)
    mc = mc.sort_values("_rank").drop_duplicates(subset=["participant_id"], keep="first").copy()
    mc.drop(columns=["_rank"], inplace=True)

    # ── 2. Build label matrix ────────────────────────────────────────────────
    label_df = build_label_matrix(mc)

    # ── 3. Merge demographics for age ───────────────────────────────────────
    dem_age = dem[["participant_id", "age"]].copy()
    dem_age["age"] = pd.to_numeric(dem_age["age"], errors="coerce")
    label_df = label_df.merge(dem_age, on="participant_id", how="left")

    # ── 4. Apply filters ─────────────────────────────────────────────────────
    mask = pd.Series(True, index=label_df.index)

    if exclude_ill:
        mask &= (label_df["ill_at_recording"] == 0)

    if age_min is not None:
        mask &= (label_df["age"] >= age_min)
    if age_max is not None:
        mask &= (label_df["age"] <= age_max)

    label_df = label_df[mask].copy()

    # ── 5. Define positive / negative class ──────────────────────────────────
    if target not in label_df.columns:
        available = [c for c in label_df.columns
                     if c not in ("participant_id", "age", "ill_at_recording")]
        raise ValueError(
            f"Unknown target '{target}'. "
            f"Available: {sorted(available)}"
        )

    pos_mask = label_df[target] == 1

    if negative_class == "healthy_only":
        # Negative class = participants with ZERO clinical flags.
        # Exclude derived/meta columns so only raw condition flags are summed.
        _exclude = {"participant_id", "ill_at_recording", "healthy_vs_all", "age", target}
        clinical_cols = [c for c in label_df.columns if c not in _exclude]
        neg_mask = label_df[clinical_cols].sum(axis=1) == 0
        keep = pos_mask | neg_mask
        label_df = label_df[keep].copy()
        label_df["label"] = pos_mask[keep].astype(int)
    else:
        label_df["label"] = pos_mask.astype(int)

    # ── 6. Load clip roster from feature parquet ──────────────────────────────
    parquet_path = features_dir / f"{feature_source}.parquet"
    if not parquet_path.exists():
        # Fall back to any available parquet for the clip list
        parquets = list(features_dir.glob("*.parquet"))
        if not parquets:
            raise FileNotFoundError(
                f"No parquet files found in {features_dir}. "
                "Run feature extraction first."
            )
        parquet_path = parquets[0]
        print(f"  [warn] {feature_source}.parquet not found, using {parquet_path.name} for clip roster")

    clips = pd.read_parquet(parquet_path, columns=["participant_id", "task_name"])
    clips = clips.drop_duplicates()

    # ── 7. Filter tasks ───────────────────────────────────────────────────────
    if tasks:
        task_mask = clips["task_name"].apply(
            lambda t: any(t.startswith(pfx) for pfx in tasks)
        )
        clips = clips[task_mask]

    # ── 8. Join labels onto clips ─────────────────────────────────────────────
    manifest = clips.merge(
        label_df[["participant_id", "age", "label"] +
                 [c for c in label_df.columns
                  if c not in ("participant_id", "age", "label")]],
        on="participant_id",
        how="inner",
    )

    # Sort for reproducibility
    manifest = manifest.sort_values(["participant_id", "task_name"]).reset_index(drop=True)

    return manifest


# ---------------------------------------------------------------------------
# --list-configs helper
# ---------------------------------------------------------------------------

def list_configs(dataset_dir: Path, exclude_ill: bool = False) -> None:
    """Print all targets with their class balance to stdout."""
    phenotype_dir = dataset_dir / "phenotype"
    mc = pd.read_csv(phenotype_dir / "pediatric/pediatric_medical_conditions.tsv",
                     sep="\t", dtype=str)

    def _rank(row: pd.Series) -> int:
        if pd.notna(row.get("peds_mc_voice_disorders")) and \
                str(row["peds_mc_voice_disorders"]).strip() not in ("", "nan"):
            return 0
        if str(row.get("peds_mc_v_dis", "No")).strip() == "Yes":
            return 1
        return 2

    mc["_rank"] = mc.apply(_rank, axis=1)
    mc = mc.sort_values("_rank").drop_duplicates(subset=["participant_id"], keep="first").copy()

    label_df = build_label_matrix(mc)
    N_total = len(label_df)

    if exclude_ill:
        label_df = label_df[label_df["ill_at_recording"] == 0].copy()
        N_total = len(label_df)

    print(f"\n{'─'*78}")
    print(f"  Bridge2AI Pediatric — available classification targets")
    print(f"  N = {N_total} participants{' (illness excluded)' if exclude_ill else ''}")
    print(f"{'─'*78}")
    print(f"  {'Target':<30}  {'Pos':>5}  {'Neg':>5}  {'Ratio':>6}  Description")
    print(f"  {'─'*28}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*35}")

    target_cols = [c for c in label_df.columns
                   if c not in ("participant_id", "ill_at_recording", "healthy_vs_all")]
    # Add healthy_vs_all at end
    target_cols.append("healthy_vs_all")

    for col in target_cols:
        pos = int(label_df[col].sum())
        neg = N_total - pos
        ratio = f"1:{neg//pos}" if pos > 0 else "  n/a"
        desc = TARGET_DESCRIPTIONS.get(col, "")[:45]
        flag = " *** too rare" if pos < 5 else ""
        print(f"  {col:<30}  {pos:>5}  {neg:>5}  {ratio:>6}  {desc}{flag}")

    print(f"\n{'─'*78}")
    print("  Age ranges available: 4–17  (use --age-min / --age-max)")
    print("  Illness confound:      61 participants had acute symptoms at recording")
    print("  Negative class modes:  --negative-class all (default) | healthy_only")
    print(f"{'─'*78}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Root of the unpacked Bridge2AI dataset.",
    )
    parser.add_argument(
        "--list-configs", action="store_true",
        help="Print all available targets with class balance and exit.",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Condition to use as positive class (label=1). See --list-configs.",
    )
    parser.add_argument(
        "--negative-class", choices=["all", "healthy_only"],
        default="all",
        help=(
            "'all'          → everything not positive is negative.\n"
            "'healthy_only' → only participants with zero clinical flags are negative.\n"
            "                 Others (different condition) are dropped."
        ),
    )
    parser.add_argument(
        "--age-min", type=float, default=None,
        help="Minimum age (inclusive). Default: no lower bound.",
    )
    parser.add_argument(
        "--age-max", type=float, default=None,
        help="Maximum age (inclusive). Default: no upper bound.",
    )
    parser.add_argument(
        "--exclude-ill", action="store_true",
        help=(
            "Exclude participants who had acute illness symptoms "
            "(cough, sneezing, ear infection, stridor) within 2 weeks of recording. "
            "Removes 61 / 300 participants."
        ),
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help=(
            "Task name prefixes to keep. E.g. --tasks long-sounds rainbow-passage. "
            "Default: all tasks."
        ),
    )
    parser.add_argument(
        "--feature-source", default="sparc_ema",
        help=(
            "Parquet file (without .parquet) used to enumerate (participant, task) clips. "
            "Default: sparc_ema. Options: sparc_ema, torchaudio_mel_spectrogram, "
            "torchaudio_mfcc, sparc_pitch, etc."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path. Default: <target>_manifest.csv in the current directory.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics but do not write the CSV.",
    )

    args = parser.parse_args(argv)

    # ── --list-configs ────────────────────────────────────────────────────────
    if args.list_configs:
        list_configs(args.dataset_dir, exclude_ill=args.exclude_ill)
        return

    if args.target is None:
        parser.error("--target is required unless --list-configs is used.")

    # ── Build manifest ────────────────────────────────────────────────────────
    print(f"\nBuilding manifest for target: {args.target}")
    print(f"  Negative class : {args.negative_class}")
    print(f"  Age range      : {args.age_min or 'any'} – {args.age_max or 'any'}")
    print(f"  Exclude ill    : {args.exclude_ill}")
    print(f"  Tasks filter   : {args.tasks or 'all'}")
    print(f"  Feature source : {args.feature_source}.parquet")

    manifest = build_manifest(
        dataset_dir=args.dataset_dir,
        target=args.target,
        negative_class=args.negative_class,
        age_min=args.age_min,
        age_max=args.age_max,
        exclude_ill=args.exclude_ill,
        tasks=args.tasks,
        feature_source=args.feature_source,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pos = int((manifest["label"] == 1).sum())
    n_neg = int((manifest["label"] == 0).sum())
    n_participants = manifest["participant_id"].nunique()
    n_clips = len(manifest)

    print(f"\nManifest summary:")
    print(f"  Participants : {n_participants}")
    print(f"  Clips        : {n_clips}")
    print(f"  Positive (1) : {n_pos} clips  ({n_pos/n_clips*100:.1f}%)")
    print(f"  Negative (0) : {n_neg} clips  ({n_neg/n_clips*100:.1f}%)")
    print(f"  Class ratio  : 1:{n_neg//n_pos if n_pos else '∞'}")

    if args.age_min is not None or args.age_max is not None:
        ages = manifest["age"].dropna()
        print(f"  Age range (actual) : {ages.min():.0f} – {ages.max():.0f}  "
              f"(mean {ages.mean():.1f})")

    tasks_in = sorted(manifest["task_name"].unique())
    print(f"  Tasks ({len(tasks_in)}): {tasks_in}")

    # Print per-task class balance
    print(f"\n  Per-task balance:")
    for task, grp in manifest.groupby("task_name"):
        p = int((grp["label"] == 1).sum())
        n = int((grp["label"] == 0).sum())
        print(f"    {task:<40} pos={p:>4}  neg={n:>4}")

    if args.dry_run:
        print("\n[dry-run] CSV not written.")
        return

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = args.out or Path(f"{args.target}_manifest.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  ({len(manifest)} rows × {len(manifest.columns)} cols)")
    print(f"Columns: {list(manifest.columns)}\n")


if __name__ == "__main__":
    main()
