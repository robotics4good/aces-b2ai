# ACES B2AI — Pediatric Voice & Secondary Features

Team ACES, [2026 Voice AI Symposium Hackathon](https://www.eventsquid.com/event.cfm?id=29517) (May 6, 2026 · St. Petersburg, FL).

## Research question

What acoustic features in pediatric voice recordings are associated with specific pathological conditions (for example airway and breathing-related histories), and can machine learning models reliably identify acoustic correlates—especially when voice is treated as a **dynamical system** across tasks and over time within a clip?

This repository combines **dataset documentation**, **EDA scripts**, and a **Python package** (`aces_b2ai`) for **secondary (derived) features** on top of Bridge2AI tensor exports (Torchaudio mel/MFCC/spectrogram/pitch) with optional SPARC voicing and external stress labels.

---

## Dataset

**Bridge2AI-Voice Pediatric Dataset v1.0.0**  
PhysioNet: [10.13026/y7mp-eh56](https://physionet.org/content/b2ai-voice-pediatric/1.0.0/)  
~300 participants · ages 2–18 · tens of thousands of clips · multiple sites.

> **Access:** PhysioNet credentialing and the Bridge2AI Voice DUA are required.  
> See [docs/data_access.md](docs/data_access.md). **Do not commit downloaded data** (use a local `data/` folder; it should stay gitignored).

### Citation

Bensoussan Y, et al. (2025). Bridge2AI-Voice Pediatric Dataset (version 1.0.0). PhysioNet. DOI: [10.13026/y7mp-eh56](https://doi.org/10.13026/y7mp-eh56).

---

## Repository structure

High-level layout (paths are from the repo root):

| Path | Purpose |
|------|---------|
| [src/aces_b2ai/](src/aces_b2ai/) | **Installable package**: alignment, voicing mask, secondary-feature extractors, pipeline, CLI, task/stress helpers. |
| [tests/](tests/) | `pytest` suite for alignment, pipeline, tasks, stress label I/O. |
| [pyproject.toml](pyproject.toml) | Package metadata and dependencies (`numpy`, `pandas`, `pyarrow`, `scipy`). |
| [environment.yml](environment.yml) | Conda environment (Python 3.11, Jupyter, ML stack, **pinned** [b2aiprep](https://github.com/robotics4good/b2aiprep) fork). |
| [docs/data_access.md](docs/data_access.md) | How to obtain PhysioNet access and local data layout. |
| [FEATURES.MD](FEATURES.MD) | Authoritative breakdown of **primary** feature modalities (static TSV, Torchaudio Parquet, SPARC, PPG), derivations, and fusion notes. |
| [FEATURES_BREAKDOWN_AND_DERIVATIONS.md](FEATURES_BREAKDOWN_AND_DERIVATIONS.md) | Same lineage as `FEATURES.MD` (exported plan); use `FEATURES.MD` as the main link. |
| [BRIDGE2AI_COMBINED.MD](BRIDGE2AI_COMBINED.MD) | Long-form working reference: phenotype joins, breathing EDA, signal-processing curriculum, dynamical-feature ideas, implementation notes. |
| [bridge2ai-voice-pediatric-dataset-1.0.0/](bridge2ai-voice-pediatric-dataset-1.0.0/) | **Bundled dataset slice**: JSON schemas under `features/` and `phenotype/`, analysis scripts, and generated reports (when present). |
| `bridge2ai-voice-pediatric-dataset-1.0.0/features/` | JSON sidecars describing Parquet tensor columns (names, shapes, extraction config). |
| `bridge2ai-voice-pediatric-dataset-1.0.0/scripts/` | [eda_airway_comprehensive.py](bridge2ai-voice-pediatric-dataset-1.0.0/scripts/eda_airway_comprehensive.py), [torchaudio_breathing_analysis.py](bridge2ai-voice-pediatric-dataset-1.0.0/scripts/torchaudio_breathing_analysis.py). |
| `bridge2ai-voice-pediatric-dataset-1.0.0/analysis_outputs/` | EDA tables/figures and [TORCHAUDIO_BREATHING_REPORT.md](bridge2ai-voice-pediatric-dataset-1.0.0/analysis_outputs/TORCHAUDIO_BREATHING_REPORT.md), [EDA_AIRWAY_FINDINGS.md](bridge2ai-voice-pediatric-dataset-1.0.0/analysis_outputs/eda/EDA_AIRWAY_FINDINGS.md). |
| `notebooks/`, `results/figures/` | Described in older README stubs; create as needed for your workflow. |
| `data/` | **Local only** (gitignored): downloaded PhysioNet files and optional raw audio. |

**In-package reference:** [src/aces_b2ai/INVENTORY.md](src/aces_b2ai/INVENTORY.md) lists primary keys, common Parquet column names, and frame-rate notes for cross-modal alignment.

---

## Team

| Name | Role |
|------|------|
| Akshay | Project Lead |
| Pranav | ML Engineer |
| Michael | Data Engineer |

---

## Setup for contributors

### 1. Conda environment (recommended for full stack + b2aiprep)

```bash
conda env create -f environment.yml
conda activate aces-b2ai
```

The environment installs the team **b2aiprep** fork from Git (commit pinned in `environment.yml`). After activation: `b2aiprep-cli --help`.

### 2. Install this repo’s `aces_b2ai` package (editable)

From the repository root:

```bash
pip install -e ".[dev]"
```

- **`[dev]`** adds `pytest` for running tests.
- Without conda, use Python **3.11+** and `pip install -e ".[dev]"` only if you already have compatible scientific stack wheels.

### 3. Run tests

```bash
pytest -q
```

---

## Using the `aces_b2ai` package

### Concepts

1. **Primary features** — Rows from Bridge2AI exports: e.g. Torchaudio `mel_spectrogram`, `mfcc`, `spectrograms`, `pitch` in Parquet; optional SPARC `periodicity`; static scalars in `static_features.tsv`.
2. **Preprocessing** — One reference time length (usually mel/MFCC width), linear resampling of pitch (and periodicity) onto that grid, and a **single voicing mask** (SPARC periodicity threshold if available, else pitch in-band).
3. **Secondary features** — Scalar summaries per clip (F0 depletion, mel energy depletion, modulation bands, MFCC dynamics, optional dynamical metrics, cross-modal coupling).

### Minimal Python example

```python
import numpy as np
from aces_b2ai import FeatureExtractionPipeline, PipelineConfig

# Synthetic stand-in for one clip: mel (60, T), mfcc (60, T), pitch (T,) or longer (will be resampled)
T = 200
mel = np.random.randn(60, T).astype(np.float32) * 0.1
mfcc = np.random.randn(60, T).astype(np.float32) * 0.05
pitch = np.full(T, 220.0, dtype=np.float32)

pipe = FeatureExtractionPipeline()
result = pipe.extract(
    participant_id="001",
    session_id="sessionA",
    task_name="long-sounds",
    mel=mel,
    mfcc=mfcc,
    pitch=pitch,
)

# Scalar features dict
print(sorted(result.features.keys())[:10], "...")

# Soft issues (e.g. too few voiced frames for optional blocks)
print(result.warnings)

# Config + provenance echo
print(result.metadata.get("pipeline", {}).keys())
```

### Pipeline with optional dynamical features

Dynamical extractors (phase portrait, sample entropy, Takens spread) are **off by default** (they need enough voiced frames and good pitch tracks):

```python
from aces_b2ai import FeatureExtractionPipeline
from aces_b2ai.config import PipelineConfig

cfg = PipelineConfig(enable_dynamical=True, dynamical_min_voiced_frames=15)
pipe = FeatureExtractionPipeline(cfg=cfg)
result = pipe.extract(
    participant_id="001",
    session_id="sessionA",
    task_name="long-sounds",
    mel=mel,
    mfcc=mfcc,
    pitch=pitch,
    periodicity_sparc=periodicity_array,  # optional, same logical length as raw pitch before align
)
```

### From a Torchaudio Parquet row dict

If you already have a row from `pyarrow` / `pandas` with nested lists:

```python
from aces_b2ai.pipeline import FeatureExtractionPipeline

row = {
    "participant_id": "...",
    "session_id": "...",
    "task_name": "...",
    "mel_spectrogram": [...],  # nested list 60 x T
    "mfcc": [...],
    "spectrograms": None,      # optional
    "pitch": [...],            # length may differ from T; will be resampled
}
result = FeatureExtractionPipeline.from_torchaudio_parquet_row(row)
```

Column names match those used in [torchaudio_breathing_analysis.py](bridge2ai-voice-pediatric-dataset-1.0.0/scripts/torchaudio_breathing_analysis.py): `mel_spectrogram`, `mfcc`, `spectrograms`, `pitch`.

---

## Command-line batch extraction

Run the package module from the repo root (after `pip install -e .`):

```bash
python -m aces_b2ai \
  --mel-parquet /path/to/torchaudio_mel_spectrogram.parquet \
  --mfcc-parquet /path/to/torchaudio_mfcc.parquet \
  --pitch-parquet /path/to/torchaudio_pitch.parquet \
  --output-csv /path/to/secondary_features.csv \
  --features-dir /path/to/bridge2ai-voice-pediatric-dataset-1.0.0/features
```

| Flag | Meaning |
|------|---------|
| `--mel-parquet`, `--mfcc-parquet`, `--pitch-parquet` | Paths to Parquet files (inner-join on `participant_id`, `session_id`, `task_name`). |
| `--output-csv` | Wide CSV: IDs + all scalar secondary columns + `warnings_json`. |
| `--features-dir` | Optional: directory containing `torchaudio_*.json`; SHA256 prefixes of those files are stored as provenance metadata per run. |
| `--max-rows` | Limit rows read from each file (smoke tests). |
| `--sustained-phonation-only` | Keep only tasks in the sustained-phonation-like set (see Tasks section). |
| `--enable-dynamical` | Register dynamical extractors (slower / stricter). |

**Note:** Clips missing from **any** of the three Parquets are skipped (inner join).

---

## Module and API reference (`src/aces_b2ai`)

### Public exports ([`__init__.py`](src/aces_b2ai/__init__.py))

| Symbol | Description |
|--------|-------------|
| `FeatureExtractionPipeline` | Orchestrates preprocess + registered extractors for one clip. |
| `ExtractionResult` | `features: dict[str, float]`, `metadata: dict`, `warnings: list[str]`. |
| `PipelineConfig` | Tunables: voicing thresholds, nominal 2D frame rate for modulation, `enable_dynamical`, `dynamical_min_voiced_frames`. |
| `is_sustained_phonation_task`, `task_theme`, `filter_rows_by_tasks`, `aggregate_secondary_by_task` | Task taxonomy helpers for stratified analysis. |
| `load_stress_labels_csv`, `merge_stress_labels` | External stress label table (not from core phenotype). |

### [`config.py`](src/aces_b2ai/config.py)

| Type | Role |
|------|------|
| `TorchaudioFeatureConfig` | Documents nominal STFT / band settings aligned with dataset JSON (sample rate, hop, mel/MFCC bins, pitch band). |
| `VoicingConfig` | `periodicity_threshold`, `f0_min_hz`, `f0_max_hz` for the hierarchical mask. |
| `PipelineConfig` | Bundles the above + `torchaudio_2d_frame_rate_hz` (used for modulation FFT scaling) + dynamical flags. |

### [`context.py`](src/aces_b2ai/context.py)

`ClipContext` holds identifiers, raw tensors (`mel`, `mfcc`, `spectrogram`, `pitch_torch`, `periodicity_sparc`, …), optional static scalars, and **post-`prepare_clip_context`** fields: `pitch_aligned`, `mfcc_aligned`, `mel_mean_energy`, `voiced_mask`, `frame_rate_hz`, `extras` (e.g. `voicing_mask_source`).

### [`align.py`](src/aces_b2ai/align.py)

| Function | Purpose |
|----------|---------|
| `reference_length(ctx)` | Chooses `T` from mel, else mfcc, else spectrogram, else pitch length. |
| `prepare_clip_context(ctx, cfg)` | Resamples pitch and periodicity to length `T`, builds `mfcc_aligned` / `mel_mean_energy`, sets `voiced_mask` and metadata. |
| `voiced_mask_from_pitch(pitch, cfg)` | Boolean mask from finite pitch within configured Hz bounds. |
| `voiced_mask_hierarchical(...)` | Prefers SPARC periodicity on the aligned grid; falls back to pitch bounds. |

### [`loaders.py`](src/aces_b2ai/loaders.py)

| Function | Purpose |
|----------|---------|
| `iter_parquet_rows(path, columns=..., batch_size=..., max_rows=...)` | Yield one dict per Parquet row. |
| `parquet_row_to_numpy_1d` / `parquet_row_to_numpy_2d` | Convert list / nested list cells to `numpy` arrays. |
| `load_static_row(static_tsv, participant_id, session_id, task_name)` | Return numeric static scalars for one clip row, or `None`. |
| `json_checksum(path)` | Short SHA256 prefix of a JSON sidecar (provenance). |

### [`pipeline.py`](src/aces_b2ai/pipeline.py)

| Function / class | Purpose |
|------------------|---------|
| `default_registry(cfg)` | `Pitch`, `Mfcc`, `Mel`, `Crossmodal` extractors; appends `Dynamical` if `enable_dynamical`. |
| `FeatureExtractionPipeline.extract(...)` | Full clip API; passes `provenance` into metadata if provided. |
| `FeatureExtractionPipeline.from_torchaudio_parquet_row(row)` | Convenience wrapper assuming standard Torchaudio column names. |

### [`core/`](src/aces_b2ai/core/)

| Module | Purpose |
|--------|---------|
| `base.py` | `ExtractionResult`, abstract `BaseExtractor`. |
| `registry.py` | `ExtractorRegistry`: runs extractors in order, merges feature dicts, collects warnings. |
| `validation.py` | Small helpers for ndarray shape checks (extend as needed). |

### [`extractors/`](src/aces_b2ai/extractors/)

| Module | Typical output keys (examples) | Clinical / modeling relevance |
|--------|-------------------------------|----------------------------------|
| `pitch_secondary.py` | `pitch_voiced_fraction_aligned`, `pitch_f0_mean_hz_masked`, `pitch_f0_std_hz_masked`, `pitch_f0_entropy_masked`, `pitch_depletion_slope_hz_per_frame`, `pitch_voiced_run_count`, `pitch_mean_voiced_run_len_frames`, `pitch_max_voiced_run_len_frames`, `pitch_frame_diff_std_hz` | Breath support and laryngeal stability: sustained phonation breaks, F0 drift, variability. |
| `mel_secondary.py` | `mel_energy_depletion_ratio`, `mel_phonation_stability_cv`, `mel_modulation_abs_mean`, `mel_mod_band_*` | Broadband noise / instability proxies; complements static spectral measures in EDA. |
| `mfcc_secondary.py` | `mfcc_delta_coeff{0..3}_abs_mean`, `mfcc_segment_drift_l2_c0_3`, `mfcc{c}_autocorr_lag{L}` | Slow spectral drift vs rapid fluctuation within a clip. |
| `crossmodal.py` | `cross_mfcc0_f0_coupling_pearson` | Simple envelope–pitch coupling on voiced frames. |
| `dynamical.py` | `dyn_phase_hull_area`, `dyn_phase_centroid_drift_l2`, `dyn_pitch_sample_entropy_m2`, `dyn_takens_spread_tau3` | Explicit “dynamical system” summaries on F0 trajectory (optional). |

Exact key sets may evolve; inspect `result.features` after an upgrade.

### [`tasks.py`](src/aces_b2ai/tasks.py)

| Symbol | Purpose |
|--------|---------|
| `SUSTAINED_PHONATION_LIKE_TASKS` | Frozen set of `task_name` strings (e.g. `long-sounds`, `silly-sounds`, `noisy-sounds`) for prolonged-voice-style activities. |
| `is_sustained_phonation_task(name)` | Normalizes case and tests membership. |
| `task_theme(name)` | Coarse bucket: sustained vs read speech vs conversation vs other. |
| `filter_rows_by_tasks(rows, allowed_tasks=..., sustained_phonation_only=...)` | Filter iterable of row dicts. |
| `aggregate_secondary_by_task(df, feature_cols=..., task_col="task_name")` | Per-task mean of numeric columns (pandas). |

Extend `SUSTAINED_PHONATION_LIKE_TASKS` if your cohort uses different `task_name` spellings.

### [`stress_labels.py`](src/aces_b2ai/stress_labels.py)

Bridge2AI pediatric phenotype JSONs in this repo do **not** define a standard stress column. This module documents and loads a **team CSV** with columns including `participant_id`, `session_id`, `stress_level` in **[0, 1]**, and `protocol_id`. Use only with appropriate IRB/DUA governance.

---

## Tests ([`tests/`](tests/))

| File | Intent |
|------|--------|
| `test_align_prepare.py` | Reference length, resampling, pitch-only mask, periodicity-driven mask. |
| `test_pipeline_synthetic.py` | End-to-end pipeline on synthetic arrays; mel depletion ordering; dynamical flag behavior. |
| `test_tasks.py` | Sustained task detection, filters, aggregation. |
| `test_stress_labels.py` | CSV load, validation, merge with a toy feature frame. |

---

## Data compatibility checklist

| Item | Status |
|------|--------|
| Primary key `(participant_id, session_id, task_name)` | **Supported** — matches dataset tensor layout. |
| Torchaudio Parquet columns used by pipeline/CLI | **`mel_spectrogram`, `mfcc`, `spectrograms`, `pitch`** — matches existing analysis script. |
| Pitch length ≠ mel time length | **Handled** — linear resampling to mel/MFCC reference length inside `prepare_clip_context`. |
| SPARC `periodicity` | **Optional** — if provided on `ClipContext` and resampled to same `T` as mel, it drives voicing; else pitch bounds. |
| Static `static_features.tsv` | **Optional** — use `loaders.load_static_row` and pass `static_features=dict` into `extract` if you merge manually (not auto-joined in pipeline today). |
| Stress condition | **External CSV only** — see `stress_labels.py`. |
| Raw WAV | **Out of scope** for `aces_b2ai` — package expects **precomputed** tensors; raw audio requires separate access and feature extraction pipelines. |

For modality semantics and secondary-feature ideas beyond this package, see [FEATURES.MD](FEATURES.MD) and [BRIDGE2AI_COMBINED.MD](BRIDGE2AI_COMBINED.MD).

---

## B2AI Prep (`b2aiprep`)

Team fork: [github.com/robotics4good/b2aiprep](https://github.com/robotics4good/b2aiprep). Source, `b2aiprep-cli`, and `external_scripts/` live there; this repo pins a commit in `environment.yml`. Bump the hash only when the team intentionally upgrades.

---

## License and data terms

Dataset use is governed by PhysioNet and the Bridge2AI Voice DUA. Code in this repository is for research and hackathon use unless otherwise noted; do not redistribute participant-level exports.
