# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ACES Team — Bridge2AI Voice Pediatric Dataset Hackathon Project**

Research Question: What acoustic features in pediatric voice recordings are associated with specific pathological conditions (especially airway and breathing-related histories), and can ML models reliably identify these acoustic correlates—especially when voice is treated as a **dynamical system** across tasks and over time within a clip?

Dataset: Bridge2AI-Voice Pediatric Dataset v1.0.0 (300 participants, ages 2-18, 22,620 recordings)

## Environment Setup

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate aces-b2ai

# 2. Install the aces_b2ai Python package (editable mode)
pip install -e ".[dev]"

# 3. Run tests to verify installation
pytest -q
```

The environment includes:
- Pinned b2aiprep fork: `git+https://github.com/robotics4good/b2aiprep.git@56d22b1...`
- `aces_b2ai` package: Secondary feature extraction pipeline
- Test suite via pytest

## Data Access & DUA Compliance

**CRITICAL**: This project uses protected health data under a Data Use Agreement (DUA).

- Data files are stored locally in `data/` directory (gitignored)
- **NEVER commit**: `.parquet`, `.csv`, `.tsv`, `.wav`, `.mp3`, `.flac`, or any participant-level data
- Data access requires PhysioNet credentialing + signing the Bridge2AI Voice DUA
- See [docs/data_access.md](docs/data_access.md) for complete access instructions
- When working with data, always verify files are in gitignored paths

## Repository Structure

```
aces-b2ai/
├── src/aces_b2ai/           # Python package: secondary feature extraction
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI entry point
│   ├── pipeline.py          # Feature extraction pipeline
│   ├── align.py             # Time-series alignment (50Hz SPARC vs 100Hz pitch)
│   ├── loaders.py           # Parquet/TSV reading utilities
│   ├── config.py            # Pipeline configuration
│   ├── context.py           # ClipContext data structure
│   ├── tasks.py             # Task taxonomy and filtering
│   ├── stress_labels.py     # External stress annotation support
│   ├── core/                # Base classes and registry
│   ├── extractors/          # Feature extractors (pitch, mel, mfcc, dynamical, crossmodal)
│   └── INVENTORY.md         # Frame rate alignment reference
├── tests/                   # pytest test suite
├── data/                    # LOCAL ONLY - participant data (never committed)
├── phenotype/               # JSON schemas for phenotype data
│   ├── pediatric/          # Demographics, medical conditions schemas
│   ├── pediatric_questionnaire/  # Voice quality surveys schemas
│   └── task/               # Session, acoustic task, recording schemas
├── features/                # JSON schemas for audio features
├── docs/                    # Data access documentation
├── *.ipynb                  # Analysis notebooks (exploratory)
├── pyproject.toml           # Package metadata
├── environment.yml          # Conda environment
├── FEATURES.MD              # Primary feature modalities documentation
├── BRIDGE2AI_COMBINED.MD    # Long-form reference
└── CLAUDE.md                # This file
```

## Dataset Architecture

### Data Organization

The dataset has a hierarchical structure:

1. **Participant** (`participant_id`) → Demographics, medical history
2. **Session** (`session_id`) → Data collection session for a participant
3. **Acoustic Task** (`acoustic_task_id`) → Specific voice recording task (e.g., sustained vowel, reading passage)
4. **Recording** (`recording_id`) → Individual audio file for a task
5. **Features** → Extracted acoustic features linked by `[participant_id, session_id, task_name]`

### Key Data Files (stored in local `data/` directory)

**Phenotype Data** (TSV format):
- `phenotype/pediatric/pediatric_demographics.tsv` — Age, gender, language, functional status
- `phenotype/pediatric/pediatric_medical_conditions.tsv` — Voice disorders, hearing loss, breathing conditions, surgical history (50 columns)
- `phenotype/pediatric_questionnaire/pediatric_vhi10.tsv` — Pediatric Voice Handicap Index-10
- `phenotype/task/session.tsv` — Session metadata
- `phenotype/task/acoustic_task.tsv` — Task types and completion status
- `phenotype/task/recording.tsv` — Recording metadata and technical details

**Audio Features** (Parquet format with nested arrays):
- `features/static_features.tsv` — Aggregate acoustic features (F0, formants, jitter, shimmer, HNR, MFCCs, loudness, speaking rate)
- `features/sparc_ema.parquet` — Estimated articulatory kinematics (12 channels @ 50Hz)
- `features/sparc_pitch.parquet` — F0 via CREPE (50Hz)
- `features/sparc_periodicity.parquet` — Voice periodicity confidence (50Hz)
- `features/sparc_loudness.parquet` — Amplitude-based loudness (50Hz)
- `features/torchaudio_mel_spectrogram.parquet` — Mel spectrogram (60 bins, subsampled for privacy)
- `features/torchaudio_mfcc.parquet` — MFCCs (60 coefficients)
- `features/torchaudio_spectrogram.parquet` — Spectrogram (201 frequency bins)
- `features/torchaudio_pitch.parquet` — F0 via autocorrelation
- `features/ppgs.parquet` — Phonetic posteriorgrams (40 phoneme categories @ 100Hz)

**Schema Files**: Each data file has a corresponding `.json` schema in the repository (not gitignored) documenting field definitions, data types, and provenance.

### Linking Tables

Join tables using:
- Participant-level: `participant_id`
- Session-level: `[participant_id, session_id]`
- Recording-level: `[participant_id, session_id, task_name]` or `[participant_id, session_id, acoustic_task_id]`

### Common Acoustic Tasks

The dataset includes 19 task types across categories:
- **Conversational Speech**: ready-for-school, favorite-show-movie-game, favorite-food, outside-of-school
- **Word Naming/Fluency**: naming-animals, naming-food
- **Sustained Phonation**: long-sounds (vowels) — *sustained phonation tasks tracked in `tasks.py:SUSTAINED_PHONATION_LIKE_TASKS`*
- **Non-Speech**: silly-sounds, noisy-sounds
- **Picture Tasks**: picture, picture-description
- **Reading**: sentence, passage
- **Speech Repetition**: repeat-words

### Frame Rate Alignment (Critical for Multi-Modal Fusion)

**See [src/aces_b2ai/INVENTORY.md](src/aces_b2ai/INVENTORY.md) for details**

| Source | Rate | Notes |
|--------|------|-------|
| Torchaudio mel/MFCC/spectrogram | ~50 Hz | After `::2` subsampling (hop 320 @ 16kHz) |
| Torchaudio pitch | ~100 Hz | NO subsampling — **must resample to mel grid** |
| SPARC (pitch/periodicity/loudness) | 50 Hz | Aligned with each other |
| PPG | 100 Hz | 40 phoneme categories |

The `aces_b2ai.align` module handles resampling pitch and periodicity to a common reference length (typically mel/MFCC width).

## Working with Data

### Reading Data Files

```python
import pandas as pd

# Phenotype data (TSV)
demographics = pd.read_csv('data/phenotype/pediatric/pediatric_demographics.tsv', sep='\t')
med_conditions = pd.read_csv('data/phenotype/pediatric/pediatric_medical_conditions.tsv', sep='\t')

# Audio features (Parquet with nested arrays)
static_features = pd.read_csv('data/features/static_features.tsv', sep='\t')
mel_spec = pd.read_parquet('data/features/torchaudio_mel_spectrogram.parquet')
```

### Using the `aces_b2ai` Package for Secondary Features

The `aces_b2ai` package extracts **secondary (derived) features** from primary Bridge2AI tensors:

```python
import numpy as np
from aces_b2ai import FeatureExtractionPipeline, PipelineConfig

# Load your data (example with synthetic data)
mel = np.random.randn(60, 200).astype(np.float32)  # 60 mel bins × 200 frames
mfcc = np.random.randn(60, 200).astype(np.float32)  # 60 coefficients × 200 frames
pitch = np.full(200, 220.0, dtype=np.float32)  # 200 frames (will be resampled if needed)

# Create pipeline
pipe = FeatureExtractionPipeline()

# Extract features
result = pipe.extract(
    participant_id="458172",
    session_id="c6030d0b",
    task_name="long-sounds",
    mel=mel,
    mfcc=mfcc,
    pitch=pitch,
)

# Access secondary features (dict of scalars)
print(result.features.keys())
# Example keys: 'pitch_f0_mean_hz_masked', 'mel_energy_depletion_ratio',
#               'mfcc_delta_coeff0_abs_mean', 'cross_mfcc0_f0_coupling_pearson'
```

**CLI batch processing:**
```bash
python -m aces_b2ai \
  --mel-parquet data/features/torchaudio_mel_spectrogram.parquet \
  --mfcc-parquet data/features/torchaudio_mfcc.parquet \
  --pitch-parquet data/features/torchaudio_pitch.parquet \
  --output-csv results/secondary_features.csv \
  --sustained-phonation-only  # Optional: filter to sustained tasks only
```

**Enabling dynamical features** (phase portraits, sample entropy, Takens embedding):
```python
cfg = PipelineConfig(enable_dynamical=True, dynamical_min_voiced_frames=20)
pipe = FeatureExtractionPipeline(cfg=cfg)
```

### Key Python Libraries

Per [environment.yml](environment.yml):
- **Package dependencies** (via pyproject.toml): numpy, pandas, pyarrow, scipy
- **Audio**: librosa, parselmouth, opensmile, b2aiprep (pinned Git fork)
- **ML**: scikit-learn, xgboost, lightgbm, torch
- **Visualization**: matplotlib, seaborn, plotly
- **Interpretability**: shap
- **Testing**: pytest (dev dependency)

### Important Data Notes

1. **Privacy**: TorchAudio spectral features are temporally subsampled (every 2nd frame) for privacy
2. **Missing Data**: Features may contain NaN when audio quality is insufficient or task was incomplete
3. **Multi-select Fields**: Medical conditions use comma-separated values (e.g., `peds_mc_breathing_conditions`)
4. **Sampling Rates**: SPARC features @ 50Hz, PPGs @ 100Hz, TorchAudio spectral (post-subsample) @ 50Hz
5. **Voicing Mask**: Pipeline uses SPARC periodicity (if available) or pitch bounds to create voiced/unvoiced masks

## Development Workflow

### Running Jupyter Notebooks

```bash
# Start Jupyter
conda activate aces-b2ai
jupyter notebook
```

Current notebooks:
- `age_distribution_analysis.ipynb` — Participant age demographics
- `medical_conditions_visualization.ipynb` — Medical condition prevalence analysis

### Key Medical Conditions to Analyze

From exploratory analysis, high-prevalence conditions include:
- Hearing loss (37.4% of participants)
- Ear tube placement (25.9%)
- Previous hospitalization (24.9%)
- Speech therapy history (24.4%)
- Allergies (23.6%)
- Specific conditions: asthma, vocal nodules/polyps, obstructive sleep apnea, GERD

### Typical Analysis Pattern

**Option 1: Using Pre-extracted Features (Static + Secondary)**
1. Load demographics and medical conditions
2. Load static features TSV and/or secondary features CSV (from CLI pipeline)
3. Join tables on `participant_id`, `session_id`, `task_name`
4. Filter for specific medical conditions or task types
5. Train/evaluate ML models
6. Generate interpretability analyses (SHAP)

**Option 2: Custom Feature Extraction**
1. Load primary features (mel, MFCC, pitch) from Parquet files
2. Use `aces_b2ai.FeatureExtractionPipeline` to compute secondary features
3. Combine with phenotype data
4. Filter and analyze

**Option 3: Task-Specific Analysis**
```python
from aces_b2ai import filter_rows_by_tasks, is_sustained_phonation_task

# Filter to sustained phonation tasks only
sustained_rows = filter_rows_by_tasks(
    all_rows,
    sustained_phonation_only=True
)

# Or check individual tasks
if is_sustained_phonation_task("long-sounds"):
    # Analyze sustained phonation metrics
    pass
```

## Important Considerations

- **Age Range**: Participants are ages 4-17, with peak enrollment at 14 (37 participants)
- **Task Completion**: ~3,800 task recordings across 300 participants (~13 tasks/participant)
- **Feature Dimensionality**: Time-series features are nested arrays requiring special handling
- **Coordinate Systems**: EMA articulator positions are relative 2D coordinates, not absolute anatomical
- **Quality Metrics**: Static features include STOI, PESQ, and SI-SDR for audio quality assessment

## Citation

When publishing results, cite:
```
Bensoussan Y, et al. (2025). Bridge2AI-Voice Pediatric Dataset (version 1.0.0).
PhysioNet. DOI: 10.13026/y7mp-eh56
```

## Package API Reference

**Key Modules** (see [README.md](README.md) for full API documentation):
- `pipeline.py` — `FeatureExtractionPipeline`, `default_registry()`
- `align.py` — Time-series resampling, voicing mask creation
- `config.py` — `PipelineConfig`, `VoicingConfig`, `TorchaudioFeatureConfig`
- `context.py` — `ClipContext` data structure
- `tasks.py` — `SUSTAINED_PHONATION_LIKE_TASKS`, task filtering
- `loaders.py` — Parquet/TSV reading utilities
- `extractors/` — Individual feature extractors:
  - `pitch_secondary.py` — F0 depletion, voiced runs, entropy
  - `mel_secondary.py` — Energy depletion, modulation bands
  - `mfcc_secondary.py` — Delta statistics, autocorrelation
  - `crossmodal.py` — Inter-feature correlations
  - `dynamical.py` — Sample entropy, phase portraits (optional)

**Test Coverage:**
```bash
pytest -v  # Run all tests with verbose output
pytest tests/test_pipeline_synthetic.py -v  # Run specific test file
```

## Documentation References

- **Primary features**: [FEATURES.MD](FEATURES.MD) — Authoritative breakdown of all feature modalities
- **Long-form reference**: [BRIDGE2AI_COMBINED.MD](BRIDGE2AI_COMBINED.MD) — Phenotype joins, breathing EDA, signal processing
- **Frame rates**: [src/aces_b2ai/INVENTORY.md](src/aces_b2ai/INVENTORY.md) — Critical alignment table
- **Dataset documentation**: [features/features.md](features/features.md) — Original feature schemas
- **Data access**: [docs/data_access.md](docs/data_access.md)
- **PhysioNet page**: https://physionet.org/content/b2ai-voice-pediatric/1.0.0/
- **Package README**: [README.md](README.md) — Complete usage guide and API reference

## Secondary Features Overview

The `aces_b2ai` package computes these secondary feature categories:

| Category | Example Features | Clinical Relevance |
|----------|------------------|-------------------|
| **Pitch/F0** | `pitch_f0_mean_hz_masked`, `pitch_depletion_slope_hz_per_frame`, `pitch_voiced_run_count` | Breath support, laryngeal stability, phonation breaks |
| **Mel Energy** | `mel_energy_depletion_ratio`, `mel_phonation_stability_cv`, `mel_mod_band_*` | Broadband noise, instability proxies |
| **MFCC Dynamics** | `mfcc_delta_coeff0_abs_mean`, `mfcc_segment_drift_l2_c0_3` | Spectral drift vs rapid fluctuation |
| **Cross-modal** | `cross_mfcc0_f0_coupling_pearson` | Envelope-pitch coupling |
| **Dynamical** (opt) | `dyn_phase_hull_area`, `dyn_pitch_sample_entropy_m2` | Trajectory complexity, system-level analysis |

See [README.md](README.md) Module Reference section for complete feature key lists.
