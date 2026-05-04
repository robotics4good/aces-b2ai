# bridge2ai_ssast

SSAST Transfer Learning Library for Bridge2AI Voice Datasets

## Overview

This library provides a complete pipeline for training and fine-tuning SSAST (Self-Supervised Audio Spectrogram Transformer) models on Bridge2AI adult and pediatric voice datasets to study acoustic biomarker generalization across age groups.

**Research Question**: How well do spectrogram-based representations learned from adult voice data generalize to pediatric populations for medical condition detection, and which acoustic biomarkers are shared versus age-specific?

## Installation

```bash
# Install in development mode
pip install -e .
```

## Three-Stage Transfer Learning Workflow

```
SSAST-Tiny-Frame-400 → Adult Supervised → Pediatric Supervised
(AudioSet+Librispeech)   (16,738 rec.)      (22,620 rec.)
```

## Quick Start

### 1. Preprocessing

**Convert linear spectrograms to 128-bin mel:**
```bash
# Adult dataset (subsample 100Hz → 50Hz)
python scripts/convert_linear_to_128mel.py --dataset adult

# Pediatric dataset (already 50Hz)
python scripts/convert_linear_to_128mel.py --dataset pediatric
```

**Parse pediatric multi-select conditions:**
```bash
python scripts/parse_pediatric_conditions.py
```

**Compute normalization statistics:**
```bash
python scripts/compute_mel128_normalization.py --dataset adult
python scripts/compute_mel128_normalization.py --dataset pediatric
```

### 2. Gradient Verification (Sanity Check)

```bash
python scripts/sanity_check_gradients.py --dataset adult
```

Verifies:
- Loss decreases over 10 iterations
- Gradients are stable (0.01-100 range)
- No NaN/Inf values

### 3. Stage 1: Adult Supervised Training

```bash
python scripts/stage1_train_adult.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir results/stage1_adult
```

Outputs:
- `best_model.pth` - Best checkpoint (highest val F1)
- `training_log.csv` - Per-epoch metrics
- `loss_curves.png` - Training/validation curves

### 4. Stage 2: Pediatric Fine-Tuning

```bash
python scripts/stage2_train_pediatric.py \
    --adult-checkpoint results/stage1_adult/best_model.pth \
    --epochs 30 \
    --batch-size 32 \
    --lr 5e-5 \
    --output-dir results/stage2_pediatric
```

Uses participant-level stratified splitting to prevent data leakage.

Outputs:
- `best_model.pth` - Best checkpoint
- `test_results.csv` - Test set performance
- `training_log.csv` - Per-epoch metrics

## Python API

### Adult Dataset

```python
from bridge2ai_ssast.datasets import AdultBridge2AIDataset
import numpy as np

# Load normalization stats
stats = np.load('adult_mel128_stats.npy', allow_pickle=True).item()
mean, std = stats['mean'], stats['std']

# Condition mapping
condition_mapping = {
    'asthma': 'asthma',
    'allergies': 'seasonal_allergies',
    'hearing_loss': 'hearing_loss',
    'voice_disorder': 'voice_disorder',
    'neurological': 'neurological_disorder',
}

# Create dataset
dataset = AdultBridge2AIDataset(
    mel_parquet_path='adult/.../mel_128bin_50hz.parquet',
    phenotype_path='adult/.../phenotype.tsv',
    condition_mapping=condition_mapping,
    target_length=200,  # ~4 seconds @ 50Hz
    normalize=True,
    mean=mean,
    std=std,
)

# Access samples
mel, labels = dataset[0]
# mel: torch.Tensor [200, 128] (time, freq)
# labels: torch.Tensor [5] (binary condition labels)
```

### Pediatric Dataset

```python
from bridge2ai_ssast.datasets import PediatricBridge2AIDataset
import numpy as np

# Load normalization stats
stats = np.load('pediatric_mel128_stats.npy', allow_pickle=True).item()
mean, std = stats['mean'], stats['std']

# Condition mapping (must match adult)
condition_mapping = {
    'asthma': 'has_asthma',
    'allergies': 'had_allergies',
    'hearing_loss': 'has_hearing_loss',
    'voice_disorder': 'has_voice_disorder',
    'neurological': 'has_neurological_disorder',
}

# Create dataset with participant filtering
train_pids = ['458172', '123456', ...]  # From stratified split

dataset = PediatricBridge2AIDataset(
    mel_parquet_path='features/mel_128bin_50hz.parquet',
    conditions_expanded_path='phenotype/pediatric/pediatric_conditions_expanded.tsv',
    condition_mapping=condition_mapping,
    target_length=200,
    normalize=True,
    mean=mean,
    std=std,
    participant_ids=train_pids,  # Optional filtering
)
```

## Package Structure

```
bridge2ai_ssast/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   ├── adult_dataset.py      # AdultBridge2AIDataset
│   ├── pediatric_dataset.py  # PediatricBridge2AIDataset
│   └── utils.py              # pad_or_truncate, normalize_mel
├── models/                   # (future: SSAST wrappers)
├── training/                 # (future: training loops)
├── preprocessing/            # (future: audio processing)
└── evaluation/               # (future: metrics, visualization)

scripts/
├── convert_linear_to_128mel.py
├── parse_pediatric_conditions.py
├── compute_mel128_normalization.py
├── sanity_check_gradients.py
├── stage1_train_adult.py
└── stage2_train_pediatric.py
```

## Key Features

### Multi-Select Condition Parsing

Pediatric conditions are stored as comma-separated strings:
```
peds_mc_breathing_conditions: "asthma, chronic_cough"
peds_mc_hearing_loss: "hearing_loss, ear tubes"
```

`parse_pediatric_conditions.py` expands these into binary columns:
```
has_asthma: 1
has_chronic_cough: 1
has_hearing_loss: 1
has_ear_tubes: 1
```

### Participant-Level Stratified Splitting

Pediatric dataset uses participant-level splits to prevent data leakage:
- Same participant never appears in both train and validation sets
- Maintains recording-level independence

### Frame Rate Alignment

- **Adult**: 201-bin linear @ 100Hz → subsample to 50Hz → convert to 128 mel bins
- **Pediatric**: 201-bin linear @ 50Hz (privacy subsampled) → convert to 128 mel bins

### SSAST Normalization

Uses SSAST's normalization scheme: `(x - mean) / (std * 2)`

## Condition Overlap (Adult ↔ Pediatric)

Identified overlapping conditions:
1. **Asthma** (respiratory/breathing)
2. **Allergies** (general medical)
3. **Hearing loss** (ENT)
4. **Voice disorder** (laryngeal/voice)
5. **Neurological disorders** (neurological)

## Output File Formats

### mel_128bin_50hz.parquet
```python
{
    'participant_id': str,
    'session_id': str,
    'task_name': str,
    'mel_spectrogram': List[List[float]],  # [128, T]
    'n_frames': int
}
```

### pediatric_conditions_expanded.tsv
```
participant_id  has_asthma  has_hearing_loss  had_allergies  ...
458172          1           0                 1              ...
```

### training_log.csv
```
epoch  train_loss  train_f1  val_loss  val_f1  val_auroc  val_auprc
1      0.6234      0.4512    0.5891    0.4823  0.7234     0.6123
```

## Citations

If you use this library, please cite:

```
Bensoussan Y, et al. (2025). Bridge2AI-Voice Pediatric Dataset (version 1.0.0).
PhysioNet. DOI: 10.13026/y7mp-eh56

Gong Y, et al. (2022). SSAST: Self-Supervised Audio Spectrogram Transformer.
AAAI 2022.
```
