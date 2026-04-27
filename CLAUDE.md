# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ACES Team — Bridge2AI Voice Pediatric Dataset Hackathon Project**

Research Question: What acoustic features in pediatric voice recordings are associated with specific pathological conditions, and can ML models reliably identify these acoustic correlates?

Dataset: Bridge2AI-Voice Pediatric Dataset v1.0.0 (300 participants, ages 2-18, 22,620 recordings)

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate aces-b2ai
```

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
├── data/                    # LOCAL ONLY - participant data (never committed)
├── phenotype/               # JSON schemas for phenotype data
│   ├── pediatric/          # Demographics, medical conditions schemas
│   ├── pediatric_questionnaire/  # Voice quality surveys schemas
│   └── task/               # Session, acoustic task, recording schemas
├── features/                # JSON schemas for audio features
├── docs/                    # Data access documentation
├── *.ipynb                  # Analysis notebooks (exploratory)
└── environment.yml          # Python dependencies
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
- **Sustained Phonation**: long-sounds (vowels)
- **Non-Speech**: silly-sounds, noisy-sounds
- **Picture Tasks**: picture, picture-description
- **Reading**: sentence, passage
- **Speech Repetition**: repeat-words

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

### Key Python Libraries

Per [environment.yml](environment.yml):
- **Data**: pandas, numpy, pyarrow, fastparquet
- **Audio**: librosa, parselmouth, opensmile, b2aiprep==3.1.0
- **ML**: scikit-learn, xgboost, lightgbm, torch
- **Visualization**: matplotlib, seaborn, plotly
- **Interpretability**: shap

### Important Data Notes

1. **Privacy**: TorchAudio spectral features are temporally subsampled (every 2nd frame) for privacy
2. **Missing Data**: Features may contain NaN when audio quality is insufficient or task was incomplete
3. **Multi-select Fields**: Medical conditions use comma-separated values (e.g., `peds_mc_breathing_conditions`)
4. **Sampling Rates**: SPARC features @ 50Hz, PPGs @ 100Hz, TorchAudio spectral (post-subsample) @ 50Hz

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

1. Load demographics and medical conditions
2. Load relevant feature set(s) based on research question
3. Join tables on `participant_id` and `session_id`
4. Filter for specific tasks or conditions
5. Extract acoustic features
6. Train/evaluate ML models
7. Generate interpretability analyses (SHAP)

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

## Documentation References

- Dataset documentation: [features/features.md](features/features.md)
- Data access: [docs/data_access.md](docs/data_access.md)
- PhysioNet page: https://physionet.org/content/b2ai-voice-pediatric/1.0.0/
