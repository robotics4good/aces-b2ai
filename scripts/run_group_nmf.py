#!/usr/bin/env python3
"""Multi-group NMF / matrix factorisation on the Bridge2AI pediatric dataset.

Runs independent PCA + NMF on three feature subsets — SPARC, TorchAudio, and
Static — then saves per-group diagnostic plots and a cross-group comparison.

Usage
-----
::

    python scripts/run_group_nmf.py \\
        --dataset-dir ~/Downloads/bridge2ai-voice-pediatric-dataset-1.0.0 \\
        --out-dir /tmp/nmf_results \\
        --groups sparc torchaudio static \\
        --n-components 20 \\
        --tasks long-sounds \\
        --aggregate-level participant

Key flags
---------
--tasks
    Comma-separated or space-separated task names to include.
    Default: all tasks.  Recommended: ``long-sounds`` for phonation tasks.
--aggregate-level
    ``"clip"``        → one row per (participant, task) clip.
    ``"participant"`` → mean-pool all clips per participant (default).
--groups
    Which subsets to analyse.  Choices: sparc  torchaudio  static.
    Default: all three.
--include-spectrogram
    Include the 201-bin linear spectrogram in the TorchAudio group.
    Off by default (adds 1 005 aggregated cols; slow to load).
--no-nmf
    Skip NMF (PCA + correlation only).  Useful for a fast first pass.
--n-components
    Number of PCA / NMF components per group.  Default: 20.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress noisy warnings from parquet / scikit-learn during aggregation.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


# ---------------------------------------------------------------------------
# Label builder (mirrors the UMAP canvas logic from previous analyses)
# ---------------------------------------------------------------------------

def build_labels(dataset_dir: Path) -> pd.DataFrame:
    """Return a DataFrame[participant_id, label_binary, label_multiclass].

    Binary:     0 = Healthy, 1 = Pathological
    Multiclass: specific disorder name (or "Healthy")
    """
    mc = pd.read_csv(
        dataset_dir / "phenotype" / "pediatric" / "pediatric_medical_conditions.tsv",
        sep="\t",
        dtype=str,
    )

    # Deduplicate: keep the row with the most specific diagnosis.
    # Rank: specific voice disorder > reflux > neurological > healthy.
    def _rank(row: pd.Series) -> int:
        if pd.notna(row.get("peds_mc_voice_disorders")) and str(row["peds_mc_voice_disorders"]).strip():
            return 0
        if str(row.get("peds_mc_v_dis", "No")).strip() == "Yes":
            return 1
        if str(row.get("peds_mc_reflux", "No")).strip() == "Yes":
            return 2
        if str(row.get("peds_mc_neurological_disorders", "No")).strip() == "Yes":
            return 3
        return 4

    mc["_rank"] = mc.apply(_rank, axis=1)
    mc = (
        mc.sort_values("_rank")
        .drop_duplicates(subset=["participant_id"], keep="first")
        .copy()
    )

    def _multiclass(row: pd.Series) -> str:
        vd = str(row.get("peds_mc_voice_disorders", "")).strip()
        if vd and vd not in ("nan", ""):
            return vd.title()
        if str(row.get("peds_mc_v_dis", "No")).strip() == "Yes":
            return "Voice Disorder (unspecified)"
        if str(row.get("peds_mc_reflux", "No")).strip() == "Yes":
            return "Acid Reflux"
        if str(row.get("peds_mc_neurological_disorders", "No")).strip() == "Yes":
            spec = str(row.get("peds_mc_neurological_disorders_specified", "")).strip()
            return f"Neurological: {spec}" if spec and spec != "nan" else "Neurological"
        return "Healthy"

    mc["label_multiclass"] = mc.apply(_multiclass, axis=1)
    mc["label_binary"] = (mc["label_multiclass"] != "Healthy").astype(int)

    return mc[["participant_id", "label_binary", "label_multiclass"]].copy()


# ---------------------------------------------------------------------------
# Parquet aggregation helpers
# ---------------------------------------------------------------------------

def _load_and_aggregate(
    features_dir: Path,
    ts_names: list[str],
    id_cols: list[str],
) -> pd.DataFrame | None:
    """Load and aggregate a list of time-series parquet files.

    Returns a single merged DataFrame keyed on ``id_cols``, or ``None``
    if none of the files were found.
    """
    from aces_b2ai.analysis import TIMESERIES_BY_NAME, FeatureAnalyzer

    merged: pd.DataFrame | None = None

    for name in ts_names:
        entry = TIMESERIES_BY_NAME.get(name)
        if entry is None:
            print(f"  [skip] {name}: not in manifest")
            continue

        parquet_path = features_dir / entry.parquet_file
        if not parquet_path.exists():
            print(f"  [skip] {name}: {parquet_path.name} not found")
            continue

        t0 = time.time()
        print(f"  Aggregating {name} ({parquet_path.name}) ...", end=" ", flush=True)
        try:
            agg = FeatureAnalyzer.aggregate_timeseries(parquet_path, entry, id_cols)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue
        print(f"{len(agg)} rows, {agg.shape[1] - len(id_cols)} cols  [{time.time()-t0:.1f}s]")

        if merged is None:
            merged = agg
        else:
            merged = merged.merge(agg, on=id_cols, how="outer")

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("~/Downloads/bridge2ai-voice-pediatric-dataset-1.0.0").expanduser(),
        help="Root of the unpacked Bridge2AI dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/nmf_results"),
        help="Directory to write plots and CSV summaries.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["sparc", "torchaudio", "static"],
        default=["sparc", "torchaudio", "static"],
        help="Which feature groups to analyse.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=(
            "Task name prefixes to keep.  Prefix matching is used, so "
            "'long-sounds' matches 'long-sounds-1' through 'long-sounds-6'.  "
            "Default: all tasks."
        ),
    )
    parser.add_argument(
        "--aggregate-level",
        choices=["clip", "participant"],
        default="participant",
        help=(
            "clip: one row per (participant, task). "
            "participant: mean-pool across all clips per participant."
        ),
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=20,
        help="Number of PCA / NMF components per group.",
    )
    parser.add_argument(
        "--include-spectrogram",
        action="store_true",
        default=False,
        help="Include the 201-bin spectrogram in the TorchAudio group (slow).",
    )
    parser.add_argument(
        "--no-nmf",
        action="store_true",
        default=False,
        help="Skip NMF (PCA + correlation analysis only).",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Spearman |r| above which pairs are flagged redundant.",
    )
    args = parser.parse_args(argv)

    dataset_dir: Path = args.dataset_dir.expanduser().resolve()
    features_dir = dataset_dir / "features"
    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_dir.exists():
        sys.exit(f"ERROR: features dir not found: {features_dir}")

    # --- Imports (deferred so argparse --help works without heavy deps) -----
    from aces_b2ai.analysis import (
        FeatureAnalyzerConfig,
        MultiGroupFeatureAnalyzer,
        ANALYSIS_GROUPS,
    )

    print("=" * 60)
    print("Bridge2AI Multi-Group NMF Analysis")
    print("=" * 60)
    print(f"Dataset : {features_dir}")
    print(f"Output  : {out_dir}")
    print(f"Groups  : {args.groups}")
    print(f"Tasks   : {args.tasks or 'all'}")
    print(f"Agg level: {args.aggregate_level}")
    print(f"Components: {args.n_components}")
    print()

    id_cols = ["participant_id", "session_id", "task_name"]

    # -----------------------------------------------------------------------
    # 1. Load static features
    # -----------------------------------------------------------------------
    print("Loading static_features.tsv ...")
    static_path = features_dir / "static_features.tsv"
    df_static = pd.read_csv(static_path, sep="\t", dtype={"participant_id": str})
    print(f"  {len(df_static)} rows, {df_static.shape[1]} cols")

    # -----------------------------------------------------------------------
    # 2. Aggregate SPARC parquets
    # -----------------------------------------------------------------------
    sparc_ts = ["sparc_loudness", "sparc_periodicity", "sparc_pitch", "ema"]
    torchaudio_ts = ["torchaudio_pitch", "mfcc", "mel", "ppg"]
    if args.include_spectrogram:
        torchaudio_ts.append("spectrogram")

    print("\nAggregating SPARC parquets ...")
    df_sparc = _load_and_aggregate(features_dir, sparc_ts, id_cols)

    print("\nAggregating TorchAudio parquets ...")
    df_torchaudio = _load_and_aggregate(features_dir, torchaudio_ts, id_cols)

    # -----------------------------------------------------------------------
    # 3. Merge all feature tables
    # -----------------------------------------------------------------------
    print("\nMerging tables ...")
    df = df_static.copy()
    for extra in [df_sparc, df_torchaudio]:
        if extra is not None:
            df = df.merge(extra, on=id_cols, how="left")

    print(f"  Combined: {len(df)} rows × {df.shape[1]} cols")

    # -----------------------------------------------------------------------
    # 4. Task filtering
    # -----------------------------------------------------------------------
    if args.tasks:
        before = len(df)
        # Prefix matching: "long-sounds" matches "long-sounds-1", "long-sounds-2", …
        mask = df["task_name"].apply(
            lambda t: any(str(t).startswith(prefix) for prefix in args.tasks)
        )
        df = df[mask].copy()
        print(f"  After task filter (prefixes={args.tasks}): {len(df)} / {before} rows")

    if len(df) == 0:
        sys.exit("ERROR: No rows remaining after task filter.")

    # -----------------------------------------------------------------------
    # 5. Load and join diagnosis labels
    # -----------------------------------------------------------------------
    print("\nBuilding diagnosis labels ...")
    labels_df = build_labels(dataset_dir)
    print(f"  Label distribution:\n{labels_df['label_multiclass'].value_counts().to_string()}")

    df["participant_id"] = df["participant_id"].astype(str)
    labels_df["participant_id"] = labels_df["participant_id"].astype(str)
    df = df.merge(labels_df, on="participant_id", how="left")

    # -----------------------------------------------------------------------
    # 6. Participant-level aggregation (optional)
    # -----------------------------------------------------------------------
    if args.aggregate_level == "participant":
        print("\nMean-pooling to participant level ...")
        non_feat_cols = ["participant_id", "session_id", "task_name",
                         "label_binary", "label_multiclass", "transcription"]
        feat_cols = [c for c in df.columns if c not in non_feat_cols]
        meta = df[["participant_id", "label_binary", "label_multiclass"]].drop_duplicates("participant_id")
        df_agg = (
            df[["participant_id"] + feat_cols]
            .groupby("participant_id", as_index=False)
            .mean(numeric_only=True)
        )
        df = df_agg.merge(meta, on="participant_id", how="left")
        print(f"  {len(df)} participants")

    # Save the assembled feature matrix for reference.
    matrix_csv = out_dir / "feature_matrix.csv"
    df.to_csv(matrix_csv, index=False)
    print(f"\nFeature matrix saved → {matrix_csv}")

    # -----------------------------------------------------------------------
    # 7. Adjust group definitions based on flags
    # -----------------------------------------------------------------------
    group_defs = {k: v for k, v in ANALYSIS_GROUPS.items() if k in args.groups}
    if not args.include_spectrogram and "torchaudio" in group_defs:
        # Remove spectrogram from the torchaudio group ts list if not requested.
        ts_list = [t for t in group_defs["torchaudio"]["timeseries_to_include"]
                   if t != "spectrogram"]
        group_defs["torchaudio"] = {
            **group_defs["torchaudio"],
            "timeseries_to_include": ts_list,
        }

    # Patch ANALYSIS_GROUPS temporarily so MultiGroupFeatureAnalyzer picks up
    # the adjusted definitions.
    import aces_b2ai.analysis.feature_analyzer as _fa_mod
    _orig_groups = _fa_mod.ANALYSIS_GROUPS.copy()
    _fa_mod.ANALYSIS_GROUPS.update(group_defs)

    # -----------------------------------------------------------------------
    # 8. Run multi-group analysis
    # -----------------------------------------------------------------------
    print(f"\nRunning multi-group factorisation (n_components={args.n_components}) ...")
    base_cfg = FeatureAnalyzerConfig(
        n_components=args.n_components,
        run_nmf=not args.no_nmf,
        run_sparse_pca=False,
        variance_threshold=0.001,
        correlation_threshold=args.correlation_threshold,
        impute_strategy="median",
        min_nonmissing_fraction=0.3,
        save_dir=str(out_dir),
    )

    mg = MultiGroupFeatureAnalyzer(
        base_cfg=base_cfg,
        groups=args.groups,
    )

    t0 = time.time()
    result = mg.fit(df, label_col="label_multiclass")
    print(f"  Done in {time.time()-t0:.1f}s")

    # Restore original group definitions.
    _fa_mod.ANALYSIS_GROUPS.clear()
    _fa_mod.ANALYSIS_GROUPS.update(_orig_groups)

    # -----------------------------------------------------------------------
    # 9. Print summary and save outputs
    # -----------------------------------------------------------------------
    print()
    print(result.summary())

    # Per-group CSV exports.
    for group_name, res in result.groups.items():
        group_dir = out_dir / group_name
        group_dir.mkdir(exist_ok=True)

        # Selected features list.
        pd.Series(res.selected_features, name="feature").to_csv(
            group_dir / "selected_features.csv", index=False
        )
        # PCA scores with labels.
        scores_df = pd.DataFrame(
            res.pca_scores,
            columns=[f"PC{i+1}" for i in range(res.pca_scores.shape[1])],
        )
        if res.labels is not None:
            scores_df.insert(0, "label", res.labels)
        scores_df.to_csv(group_dir / "pca_scores.csv", index=False)

        # NMF activations (W matrix).
        if res.nmf_scores is not None:
            nmf_df = pd.DataFrame(
                res.nmf_scores,
                columns=[f"NMF{i}" for i in range(res.nmf_scores.shape[1])],
            )
            if res.labels is not None:
                nmf_df.insert(0, "label", res.labels)
            nmf_df.to_csv(group_dir / "nmf_scores.csv", index=False)

        # NMF component loadings (H matrix) — which features load on each component.
        if res.nmf_components is not None:
            H_df = pd.DataFrame(
                res.nmf_components,
                index=[f"NMF{i}" for i in range(res.nmf_components.shape[0])],
                columns=res.nmf_feature_names,
            )
            H_df.to_csv(group_dir / "nmf_components.csv")

        # Redundant pairs.
        if res.redundant_pairs:
            pd.DataFrame(
                res.redundant_pairs, columns=["feat_a", "feat_b", "spearman_r"]
            ).to_csv(group_dir / "redundant_pairs.csv", index=False)

        print(f"  [{group_name}] CSVs saved → {group_dir}/")

    # Generate all plots.
    print("\nGenerating plots ...")
    try:
        result.plot_all(save_dir=str(out_dir))
        print(f"  Plots saved → {out_dir}/")
    except Exception as exc:
        print(f"  WARNING: plot generation failed: {exc}")

    print("\nDone.")
    print(f"Results in: {out_dir}")


if __name__ == "__main__":
    main()
