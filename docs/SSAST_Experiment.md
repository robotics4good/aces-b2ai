# SSAST Transfer Learning Experiment: Adult → Pediatric Voice Biomarkers

## Research Question

**How well do spectrogram-based representations learned from adult voice data generalize to pediatric populations for medical condition detection, and which acoustic biomarkers are shared versus age-specific?**

---

## Dataset Overview

### Adult Dataset (Bridge2AI Voice v2.0.1)
- **Source**: `adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/`
- **Size**: 16,738 recordings
- **Features**: 201-bin linear spectrogram @ 100 Hz
  - n_fft: 400, hop_length: 160 samples @ 16kHz
  - **No privacy subsampling** (full temporal resolution)
- **Phenotype**: `phenotype.tsv` with medical conditions, demographics
- **Conditions**: Neurodegenerative focus (Alzheimer's, Parkinson's, ALS), plus general (asthma, COPD, allergies, hearing, voice disorders)

### Pediatric Dataset (Bridge2AI Voice Pediatric v1.0.0)
- **Source**: `phenotype/` (various subdirectories)
- **Size**: 22,620 recordings (300 participants, ages 4-17)
- **Features**: 201-bin linear spectrogram @ 50 Hz
  - n_fft: 400, hop_length: 160 @ 16kHz → 100 Hz native
  - **Privacy subsampling**: `data[:, ::2]` → **50 Hz final**
- **Phenotype**: `phenotype/pediatric/pediatric_medical_conditions.tsv`
- **Conditions**: Respiratory (asthma, croup, RSV), ENT (hearing loss, ear tubes, tonsillectomy), voice disorders, neurological, psychiatric

### Key Difference
- **Adult**: 100 Hz temporal resolution
- **Pediatric**: 50 Hz (privacy-protected)
- **→ Solution**: Subsample adult to 50 Hz for consistency

---

## Multi-Select Condition Fields in Pediatric Dataset

### Critical Discovery
Pediatric medical conditions are stored in **9 multi-select fields** as **comma-separated strings**.

#### Example Data Format:
```
peds_mc_breathing_conditions: "asthma, chronic_cough"
peds_mc_hearing_loss: "hearing_loss, ear tubes"
peds_mc_psych_disorders: "anxiety disorder, adhd, social anxiety disorder"
```

#### Multi-Select Fields (from JSON schema):

1. **`peds_mc_breathing_conditions`** (10 choices):
   - `asthma`, `recurrent_croup`, `bronchiolitis`, `rsv`, `chronic_cough`, `subglottic_stenosis`, `vocal_cord_webs`, `airway_surgery`, `intubated`, `tracheostomy`

2. **`peds_mc_hearing_loss`** (18 choices):
   - `hearing_loss`, `mild_hearing_loss`, `moderate_hearing_loss`, `severe_hearing_loss`
   - `conductive_hearing_loss`, `sensorineural_hearing_loss`
   - `unilateral_hearing_loss`, `bilateral_hearing_loss`
   - `hearing_aid`, `cochlear_implantation_unilateral`, `cochlear_implantation_bilateral`
   - `ear_tubes`, `cholesteatoma`, ...

3. **`peds_mc_voice_disorders`** (7 choices):
   - `vocal_nodules_or_polyps`, `unilateral_vocal_cord_paralysis`, `bilateral_vocal_cord_paralysis`
   - `acid_reflux_disease` (GERD), `scarring`, `airway_stenosis`, `vocal_cord_webs`

4. **`peds_mc_psych_disorders`** (choices in data, not JSON):
   - `adhd`, `anxiety_disorder`, `social_anxiety_disorder`, `depression`, ...

5. **`peds_mc_conditions`** (general medical - TBD from data)

6-9. **Other multi-select fields**: Surgery history, etc.

### Parsing Strategy

**Before**: Treat entire field as binary (e.g., "has any breathing condition")

**After (Recommended)**: Extract granular conditions

```python
def parse_multi_select_field(value_str):
    """Parse comma-separated condition string"""
    if pd.isna(value_str):
        return []
    return [c.strip() for c in value_str.split(',')]

# Example usage
breathing_conditions = parse_multi_select_field(row['peds_mc_breathing_conditions'])
# Returns: ['asthma', 'chronic_cough']

# Create binary labels
labels['has_asthma'] = 1 if 'asthma' in breathing_conditions else 0
labels['has_chronic_cough'] = 1 if 'chronic_cough' in breathing_conditions else 0
```

**Output**: `pediatric_conditions_expanded.tsv` with 30-50 binary condition columns

---

## Overlapping Conditions (Adult ↔ Pediatric)

### Identified Overlaps

| Condition | Pediatric Field | Adult Field | Match Quality |
|-----------|-----------------|-------------|---------------|
| **Asthma** | `peds_mc_breathing_conditions` ⊃ `"asthma"` | `asthma` (column 35) | ✅ Direct |
| **Allergies** | `peds_mc_allergies` (yes/no) | `seasonal_allergies` (column 453) | ✅ Direct |
| **Hearing Loss** | `peds_mc_hearing_loss` ⊃ `"hearing_loss"` | `hearing` (column 392) | ✅ Good |
| **Voice Disorder** | `peds_mc_voice_disorders` (multi-select) | `voice_quality_perception` (cols 591-605) | ✅ Good |
| **Sleep Apnea** | `peds_mc_snoring_apnea` (yes/no) | `respiratory_conditions___?` (cols 504-511) | ⚠️ Need to verify |
| **Speech Therapy** | `peds_mc_a_therapy` (yes/no) | `throat_med_history___?` (cols 473-482) | ⚠️ Need to verify |
| **Tonsillectomy** | `peds_mc_tonsillectomy` (yes/no) | `throat_surgical___?` (cols 491-494) | ⚠️ Need to verify |
| **Neurological** | `peds_mc_neurological_disorders` (yes/no) | `current_neuro_dx` (column 503) | ✅ Good |

### Recommended Conditions for Initial Experiment

**Tier 1 (Confirmed Overlaps - 5 conditions):**
1. **Asthma** - Respiratory, affects voice
2. **Allergies** - Respiratory, voice quality
3. **Hearing Loss** - Speech production feedback
4. **Voice Disorder** - Direct pathology
5. **Neurological** - Speech/voice motor control

**Tier 2 (Pending Adult Column Verification - 3 conditions):**
6. **Sleep Apnea** - Respiratory/airway
7. **Speech Therapy History** - Indicates voice/speech issues
8. **Tonsillectomy** - Airway intervention

**Start with Tier 1 (5 conditions), expand to Tier 2 after adult data verification.**

---

## Experiment Design

### Overview: Three-Stage Transfer Learning

```
AudioSet + Librispeech (2M+ samples)
           ↓ (SSL pretraining)
    SSAST-Tiny-Frame-400
           ↓ (supervised fine-tuning)
    Adult Classifier (16,738 samples)
           ↓ (supervised fine-tuning)
    Pediatric Classifier (300 participants)
```

---

### Stage 0: Data Preparation & Infrastructure

#### 0.1: Download Pretrained Model

```bash
cd ssast/pretrained_model/
wget https://www.dropbox.com/s/rx7g60ruzawffzv/SSAST-Tiny-Frame-400.pth?dl=1 -O SSAST-Tiny-Frame-400.pth
```

**Model Specifications:**
- **Name**: SSAST-Tiny-Frame-400
- **Architecture**: Frame-based (good for speech)
  - `fshape=128, tshape=2` (frequency × time patch size)
  - `fstride=128, tstride=2` (non-overlapping in pretraining)
- **Input**: **128 mel bins** × variable time frames
- **Pretrained on**: AudioSet-2M + Librispeech-960 (SSL masked reconstruction)
- **Size**: 6M parameters (appropriate for pediatric n=300)
- **Performance**: Avg speech AUROC 47.8% (untested), suitable for fine-tuning

#### 0.2: Convert 201-bin Linear Spec → 128-bin Mel Spec

**Challenge**: Both datasets have 201-bin linear spectrograms, but SSAST-Tiny-Frame-400 expects **128 mel bins**.

**Solution**: Compute mel filterbank from existing linear spectrograms (no need for raw audio).

**Script**: `scripts/convert_linear_to_128mel.py`

```python
import torch
import torchaudio
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

def convert_linear_to_mel(linear_spec, n_mels=128):
    """
    Convert 201-bin linear spectrogram to 128-bin mel spectrogram

    Args:
        linear_spec: [201, T] - power/magnitude spectrogram
    Returns:
        mel_spec: [128, T] - mel spectrogram
    """
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=16000,
        f_min=0.0,
        f_max=8000.0,
        n_stft=201  # input has 201 frequency bins
    )

    linear_tensor = torch.tensor(linear_spec, dtype=torch.float32)
    mel_spec = mel_scale(linear_tensor)  # [128, T]

    return mel_spec.numpy()

# Process adult dataset
print("Converting adult linear → 128-bin mel...")
adult_table = pq.read_table('adult/.../spectrogram.parquet')
adult_df = adult_table.to_pandas()

adult_mel_rows = []
for idx, row in tqdm(adult_df.iterrows(), total=len(adult_df)):
    linear_spec = np.array(row['spectrogram'])  # [201, T]

    # Subsample time axis: 100Hz → 50Hz to match pediatric
    linear_spec = linear_spec[:, ::2]

    mel_spec = convert_linear_to_mel(linear_spec, n_mels=128)

    adult_mel_rows.append({
        'participant_id': row['participant_id'],
        'session_id': row['session_id'],
        'task_name': row['task_name'],
        'mel_spectrogram': mel_spec.tolist(),
        'n_frames': mel_spec.shape[1]
    })

# Save
import pandas as pd
adult_mel_df = pd.DataFrame(adult_mel_rows)
adult_mel_df.to_parquet('adult/.../mel_128bin_50hz.parquet', compression='zstd')

# Process pediatric dataset (already 50Hz)
print("Converting pediatric linear → 128-bin mel...")
# ... similar process, no time subsampling needed ...
```

**Output:**
- `adult/.../mel_128bin_50hz.parquet` - 16,738 adult spectrograms [128, T] @ 50Hz
- `phenotype/features/mel_128bin_50hz.parquet` - 22,620 pediatric spectrograms [128, T] @ 50Hz

#### 0.3: Parse Multi-Select Condition Fields

**Script**: `scripts/parse_pediatric_conditions.py`

```python
import pandas as pd

def parse_multi_select_field(value_str):
    """Parse comma-separated condition string"""
    if pd.isna(value_str):
        return []
    return [c.strip() for c in value_str.split(',')]

# Load pediatric medical conditions
df = pd.read_csv('phenotype/pediatric/pediatric_medical_conditions.tsv', sep='\t')

# Initialize binary condition columns
all_conditions = set()

# Collect all unique conditions from multi-select fields
for field in ['peds_mc_breathing_conditions', 'peds_mc_hearing_loss',
              'peds_mc_voice_disorders', 'peds_mc_psych_disorders']:
    for value in df[field].dropna():
        conditions = parse_multi_select_field(value)
        all_conditions.update(conditions)

print(f"Found {len(all_conditions)} unique conditions across multi-select fields")

# Create binary columns for each condition
expanded_df = pd.DataFrame({'participant_id': df['participant_id']})

for condition in sorted(all_conditions):
    col_name = f'has_{condition}'
    expanded_df[col_name] = 0

    # Check each multi-select field
    for field in ['peds_mc_breathing_conditions', 'peds_mc_hearing_loss',
                  'peds_mc_voice_disorders', 'peds_mc_psych_disorders']:
        for idx, value in enumerate(df[field]):
            if pd.notna(value) and condition in parse_multi_select_field(value):
                expanded_df.loc[idx, col_name] = 1

# Add binary fields (yes/no questions)
expanded_df['had_allergies'] = (df['peds_mc_allergies'] == 'yes').astype(int)
expanded_df['had_speech_therapy'] = (df['peds_mc_a_therapy'] == 'yes').astype(int)
expanded_df['had_tonsillectomy'] = (df['peds_mc_tonsillectomy'] == 'yes').astype(int)
expanded_df['has_neurological'] = (df['peds_mc_neurological_disorders'] == 'yes').astype(int)

# Save
expanded_df.to_csv('phenotype/pediatric/pediatric_conditions_expanded.tsv', sep='\t', index=False)

# Generate prevalence report
prevalence = expanded_df.drop('participant_id', axis=1).mean().sort_values(ascending=False)
prevalence.to_csv('condition_prevalence_report.csv')
print("\nTop 10 most prevalent conditions:")
print(prevalence.head(10))
```

**Output:**
- `pediatric_conditions_expanded.tsv` - 30-50 binary condition columns
- `condition_prevalence_report.csv` - Condition prevalence for selection

#### 0.4: Compute Normalization Statistics

**Script**: `scripts/compute_mel128_normalization.py`

```python
import pyarrow.parquet as pq
import numpy as np

def compute_normalization_stats(parquet_path):
    """Compute mean/std for 128-bin mel spectrograms"""
    df = pq.read_table(parquet_path).to_pandas()

    all_values = []
    for mel in df['mel_spectrogram']:
        mel_array = np.array(mel)  # [128, T]
        all_values.append(mel_array.flatten())

    all_values = np.concatenate(all_values)
    mean = np.mean(all_values)
    std = np.std(all_values)

    return mean, std

# Adult
adult_mean, adult_std = compute_normalization_stats('adult/.../mel_128bin_50hz.parquet')
print(f"Adult: mean={adult_mean:.4f}, std={adult_std:.4f}")
np.save('adult_mel128_stats.npy', {'mean': adult_mean, 'std': adult_std})

# Pediatric
ped_mean, ped_std = compute_normalization_stats('phenotype/features/mel_128bin_50hz.parquet')
print(f"Pediatric: mean={ped_mean:.4f}, std={ped_std:.4f}")
np.save('pediatric_mel128_stats.npy', {'mean': ped_mean, 'std': ped_std})
```

#### 0.5: Verify Adult Condition Column Names

**TODO**: Manually inspect adult `phenotype.tsv` to confirm exact column names for overlapping conditions.

```bash
# Check respiratory conditions
head -1 adult/.../phenotype.tsv | tr '\t' '\n' | nl | grep -i "respiratory"

# Check throat/voice columns
head -1 adult/.../phenotype.tsv | tr '\t' '\n' | nl | grep -i "throat\|voice"

# Verify asthma, allergies, hearing columns exist
head -1 adult/.../phenotype.tsv | tr '\t' '\n' | nl | grep -E "^\\s*(35|392|453)\\s"
```

**Create mapping file**: `docs/CONDITION_MAPPING.md`

#### 0.6: Sanity Check - Gradient Verification

**Purpose**: Verify model architecture and training logic on small subsample before full training runs to catch bugs early.

**Script**: `scripts/sanity_check_gradients.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from ssast.src.models.ast_models import ASTModel

# Initialize model with pretrained weights
model = ASTModel(
    label_dim=5,  # 5 overlapping conditions
    fshape=128,
    tshape=2,
    fstride=128,
    tstride=1,
    input_fdim=128,
    input_tdim=200,
    model_size='tiny',
    pretrain_stage=False,
    load_pretrained_mdl_path='ssast/pretrained_model/SSAST-Tiny-Frame-400.pth'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Use only 50 samples from adult dataset
small_dataset = Subset(adult_dataset, range(50))
small_loader = DataLoader(small_dataset, batch_size=10, shuffle=True)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# Track metrics
losses = []
gradient_norms = []

print("Running gradient verification on 50-sample subsample...")
print("=" * 60)

# Train for 10 iterations
model.train()
for i, (mels, labels) in enumerate(small_loader):
    if i >= 10:
        break

    mels, labels = mels.to(device), labels.to(device)

    # Forward pass
    outputs = model(mels, task='ft_avgtok')
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Track gradient norm across all parameters
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    gradient_norms.append(total_norm)
    losses.append(loss.item())

    optimizer.step()

    print(f"Iter {i+1}/10: Loss={loss.item():.4f}, Gradient Norm={total_norm:.4f}")

print("=" * 60)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(losses, marker='o', linewidth=2)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Loss (BCEWithLogits)', fontsize=12)
ax1.set_title('Loss Over 10 Iterations (50 Samples)', fontsize=14)
ax1.grid(True, alpha=0.3)

# Gradient norm
ax2.plot(gradient_norms, marker='o', linewidth=2, color='orange')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Gradient Norm (L2)', fontsize=12)
ax2.set_title('Gradient Flow Verification', fontsize=14)
ax2.axhline(y=0.1, color='red', linestyle='--', label='Min threshold (0.1)')
ax2.axhline(y=10, color='red', linestyle='--', label='Max threshold (10)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_verification.png', dpi=150)
print(f"\n✅ Plot saved to gradient_verification.png")

# Verification checks
print("\n" + "=" * 60)
print("VERIFICATION RESULTS:")
print("=" * 60)

loss_decreasing = losses[-1] < losses[0]
grad_stable = all(0.01 < g < 100 for g in gradient_norms)
no_nan = not any(np.isnan(losses)) and not any(np.isnan(gradient_norms))

print(f"✅ Loss decreasing: {loss_decreasing} (start={losses[0]:.4f}, end={losses[-1]:.4f})")
print(f"✅ Gradients stable (0.01-100): {grad_stable} (mean={np.mean(gradient_norms):.4f})")
print(f"✅ No NaN/Inf values: {no_nan}")

if not loss_decreasing:
    print("\n⚠️  WARNING: Loss not decreasing!")
    print("    Possible causes:")
    print("    - Learning rate too low")
    print("    - Data normalization incorrect")
    print("    - Labels incorrectly encoded")

if not grad_stable:
    print("\n⚠️  WARNING: Gradient instability detected!")
    if np.mean(gradient_norms) < 0.1:
        print("    Issue: Vanishing gradients")
        print("    Solutions: Check normalization stats, layer initialization")
    else:
        print("    Issue: Exploding gradients")
        print("    Solutions: Reduce learning rate, add gradient clipping")

if no_nan and loss_decreasing and grad_stable:
    print("\n🎉 All checks passed! Ready for full training.")
else:
    print("\n❌ Some checks failed. Debug before full training.")

print("=" * 60)
```

**Success Criteria:**
- ✅ Loss decreases over 10 iterations on 50-sample subsample
- ✅ Gradient norms are stable (typically 0.1-10 range, broader 0.01-100 acceptable)
- ✅ No NaN or Inf values in loss or gradients
- ✅ Model outputs reasonable predictions (not all 0s or 1s)

**If Issues Found:**

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| **Vanishing gradients** (norm < 0.1) | Layers not learning | Check normalization stats, verify pretrained weights loaded |
| **Exploding gradients** (norm > 100) | Training instability | Reduce learning rate to 1e-5, add gradient clipping (`torch.nn.utils.clip_grad_norm_`) |
| **Static loss** (no decrease) | Model not training | Verify labels correctly encoded (0/1 for multi-label), check data preprocessing |
| **NaN loss** | Numerical instability | Check for division by zero, invalid log operations, extreme normalization values |
| **All predictions → 0** | Dead ReLU or wrong activation | Inspect model output layer, verify BCEWithLogitsLoss (no sigmoid before loss) |

**Expected Output:**
- Console log with 10 iterations of loss/gradient metrics
- `gradient_verification.png` with dual plots (loss curve + gradient norms)
- Pass/fail verification report

**Time**: 5-10 minutes

---

### Stage 1: Fine-Tune on Adult Dataset (Supervised Classification)

#### Model Configuration

```python
from ssast.src.models.ast_models import ASTModel

model = ASTModel(
    label_dim=5,  # 5 overlapping conditions
    fshape=128,   # full frequency axis (frame-based)
    tshape=2,
    fstride=128,  # no overlap in pretraining
    tstride=1,    # frame-level for fine-tuning
    input_fdim=128,
    input_tdim=200,  # ~4 seconds @ 50Hz (adjust based on median task length)
    model_size='tiny',
    pretrain_stage=False,
    load_pretrained_mdl_path='ssast/pretrained_model/SSAST-Tiny-Frame-400.pth'
)
```

#### Dataloader

**File**: `bridge2ai_ssast/datasets/adult_dataset.py`

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

class AdultBridge2AIDataset(Dataset):
    def __init__(self, mel_parquet_path, phenotype_path,
                 condition_mapping, target_length=200, normalize=True):
        """
        Args:
            mel_parquet_path: Path to adult mel_128bin_50hz.parquet
            phenotype_path: Path to adult phenotype.tsv
            condition_mapping: Dict mapping condition names to column names
                Example: {'asthma': 'asthma', 'allergies': 'seasonal_allergies', ...}
            target_length: Number of time frames to pad/truncate to
            normalize: Apply dataset normalization
        """
        # Load mel spectrograms
        self.mels = pq.read_table(mel_parquet_path).to_pandas()

        # Load phenotype data
        self.phenotype = pd.read_csv(phenotype_path, sep='\t')

        # Merge on participant_id and session_id
        self.data = self.mels.merge(self.phenotype,
                                     on=['participant_id', 'session_id'])

        self.conditions = list(condition_mapping.keys())
        self.condition_cols = condition_mapping
        self.target_length = target_length

        # Load normalization stats
        if normalize:
            stats = np.load('adult_mel128_stats.npy', allow_pickle=True).item()
            self.mean = stats['mean']
            self.std = stats['std']
        else:
            self.mean = 0.0
            self.std = 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get mel spectrogram [128, T]
        mel = np.array(row['mel_spectrogram'])

        # Transpose to [T, 128] for SSAST
        mel = mel.T

        # Pad or truncate to target_length
        if mel.shape[0] < self.target_length:
            mel = np.pad(mel, ((0, self.target_length - mel.shape[0]), (0, 0)), mode='constant')
        else:
            mel = mel[:self.target_length, :]

        # Normalize (SSAST convention: (x - mean) / (std * 2))
        mel = (mel - self.mean) / (self.std * 2)

        # Multi-label encoding
        labels = np.zeros(len(self.conditions), dtype=np.float32)
        for i, cond_name in enumerate(self.conditions):
            adult_col = self.condition_cols[cond_name]
            value = row.get(adult_col, None)

            # Handle different formats (yes/no vs 1/0)
            if value == 'yes' or value == 1 or value == True:
                labels[i] = 1.0

        return torch.tensor(mel, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
```

#### Training Script

**File**: `scripts/stage1_adult_supervised.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from bridge2ai_ssast.datasets.adult_dataset import AdultBridge2AIDataset
from ssast.src.models.ast_models import ASTModel

# Define 5 overlapping conditions
condition_mapping = {
    'asthma': 'asthma',  # Column name in adult phenotype.tsv
    'allergies': 'seasonal_allergies',
    'hearing_loss': 'hearing',
    'voice_disorder': 'voice_quality_perception',  # or appropriate column
    'neurological': 'current_neuro_dx'
}

# Create dataset
dataset = AdultBridge2AIDataset(
    mel_parquet_path='adult/.../mel_128bin_50hz.parquet',
    phenotype_path='adult/.../phenotype.tsv',
    condition_mapping=condition_mapping,
    target_length=200
)

# Train/val split (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=4)

# Create model
model = ASTModel(
    label_dim=5,
    fshape=128, tshape=2, fstride=128, tstride=1,
    input_fdim=128, input_tdim=200,
    model_size='tiny',
    pretrain_stage=False,
    load_pretrained_mdl_path='ssast/pretrained_model/SSAST-Tiny-Frame-400.pth'
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model).to(device)

# Optimizer & loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-7)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(25):
    model.train()
    train_loss = 0.0

    for batch_idx, (mels, labels) in enumerate(train_loader):
        mels = mels.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(mels, task='ft_avgtok')  # Use average token pooling
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mels, labels in val_loader:
            mels = mels.to(device)
            labels = labels.to(device)
            outputs = model(mels, task='ft_avgtok')
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f'Epoch {epoch} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Save checkpoint
torch.save(model.state_dict(), 'checkpoints/adult_5cond_tiny.pth')
print("Adult model saved to checkpoints/adult_5cond_tiny.pth")
```

**Expected Outcome:**
- Adult validation AUROC: 0.65-0.75 (depends on condition prevalence and quality)
- Checkpoint ready for pediatric fine-tuning

---

### Stage 2: Fine-Tune on Pediatric Dataset (Transfer from Adult)

#### Dataloader

**File**: `bridge2ai_ssast/datasets/pediatric_dataset.py`

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

class PediatricBridge2AIDataset(Dataset):
    def __init__(self, mel_parquet_path, conditions_expanded_path,
                 condition_mapping, tasks=None, target_length=200, normalize=True):
        """
        Args:
            mel_parquet_path: Path to pediatric mel_128bin_50hz.parquet
            conditions_expanded_path: Path to pediatric_conditions_expanded.tsv
            condition_mapping: Dict mapping condition names to expanded column names
                Example: {'asthma': 'has_asthma', 'allergies': 'had_allergies', ...}
            tasks: List of task names to filter (e.g., ['long-sounds'])
            target_length: Number of frames to pad/truncate to
        """
        # Load mel spectrograms
        self.mels = pq.read_table(mel_parquet_path).to_pandas()

        # Filter by task if specified
        if tasks is not None:
            self.mels = self.mels[self.mels['task_name'].isin(tasks)]

        # Load expanded conditions
        self.conditions_df = pd.read_csv(conditions_expanded_path, sep='\t')

        # Merge on participant_id
        self.data = self.mels.merge(self.conditions_df, on='participant_id')

        self.conditions = list(condition_mapping.keys())
        self.condition_cols = condition_mapping
        self.target_length = target_length

        # Load normalization stats
        if normalize:
            stats = np.load('pediatric_mel128_stats.npy', allow_pickle=True).item()
            self.mean = stats['mean']
            self.std = stats['std']
        else:
            self.mean = 0.0
            self.std = 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get mel spectrogram [128, T]
        mel = np.array(row['mel_spectrogram'])
        mel = mel.T  # → [T, 128]

        # Pad/truncate
        if mel.shape[0] < self.target_length:
            mel = np.pad(mel, ((0, self.target_length - mel.shape[0]), (0, 0)))
        else:
            mel = mel[:self.target_length, :]

        # Normalize
        mel = (mel - self.mean) / (self.std * 2)

        # Multi-label encoding from expanded conditions
        labels = np.zeros(len(self.conditions), dtype=np.float32)
        for i, cond_name in enumerate(self.conditions):
            expanded_col = self.condition_cols[cond_name]
            labels[i] = float(row.get(expanded_col, 0))

        return torch.tensor(mel, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
```

#### Training Script

**File**: `scripts/stage2_pediatric_from_adult.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bridge2ai_ssast.datasets.pediatric_dataset import PediatricBridge2AIDataset
from ssast.src.models.ast_models import ASTModel
from sklearn.model_selection import train_test_split

# Define 5 overlapping conditions (SAME as adult)
condition_mapping = {
    'asthma': 'has_asthma',  # From expanded conditions
    'allergies': 'had_allergies',
    'hearing_loss': 'has_hearing_loss',
    'voice_disorder': 'has_vocal_nodules_or_polyps',  # or appropriate voice disorder
    'neurological': 'has_neurological'
}

# Create dataset (filter to sustained phonation)
dataset = PediatricBridge2AIDataset(
    mel_parquet_path='phenotype/features/mel_128bin_50hz.parquet',
    conditions_expanded_path='phenotype/pediatric/pediatric_conditions_expanded.tsv',
    condition_mapping=condition_mapping,
    tasks=['long-sounds'],  # Focus on sustained phonation
    target_length=200
)

# Participant-level stratified split
participant_ids = dataset.data['participant_id'].unique()
train_pids, test_pids = train_test_split(participant_ids, test_size=0.2, random_state=42)
train_pids, val_pids = train_test_split(train_pids, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

train_indices = dataset.data[dataset.data['participant_id'].isin(train_pids)].index.tolist()
val_indices = dataset.data[dataset.data['participant_id'].isin(val_pids)].index.tolist()
test_indices = dataset.data[dataset.data['participant_id'].isin(test_pids)].index.tolist()

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Load adult-trained model
model = ASTModel(
    label_dim=5,  # SAME as adult (5 conditions)
    fshape=128, tshape=2, fstride=128, tstride=1,
    input_fdim=128, input_tdim=200,
    model_size='tiny',
    pretrain_stage=False,
    load_pretrained_mdl_path='checkpoints/adult_5cond_tiny.pth'  # Load adult checkpoint
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model).to(device)

# Fine-tune ALL parameters (backbone + head)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-7)  # Lower LR
criterion = nn.BCEWithLogitsLoss()

# Early stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# Training loop
for epoch in range(50):
    model.train()
    train_loss = 0.0

    for mels, labels in train_loader:
        mels = mels.to(device)
        labels = labels.to(device)

        outputs = model(mels, task='ft_avgtok')
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mels, labels in val_loader:
            mels = mels.to(device)
            labels = labels.to(device)
            outputs = model(mels, task='ft_avgtok')
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f'Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'checkpoints/pediatric_from_adult_5cond_tiny.pth')
        print("  → Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print("Pediatric model training complete")
```

---

### Baseline Comparisons

#### Baseline 1: Pediatric from Scratch (No Transfer)

**File**: `scripts/baseline_pediatric_scratch.py`

```python
# Same as Stage 2, but initialize model without pretrained weights

model = ASTModel(
    label_dim=5,
    fshape=128, tshape=2, fstride=128, tstride=1,
    input_fdim=128, input_tdim=200,
    model_size='tiny',
    pretrain_stage=False,
    load_pretrained_mdl_path=None  # No pretrained weights
)

# Train on pediatric from random initialization
# ... rest of training code same as Stage 2 ...
```

#### Baseline 2: Pediatric from SSL (Skip Adult)

**File**: `scripts/baseline_pediatric_ssl.py`

```python
# Load SSAST-Tiny-Frame-400 directly, skip adult supervised training

model = ASTModel(
    label_dim=5,
    fshape=128, tshape=2, fstride=128, tstride=1,
    input_fdim=128, input_tdim=200,
    model_size='tiny',
    pretrain_stage=False,
    load_pretrained_mdl_path='ssast/pretrained_model/SSAST-Tiny-Frame-400.pth'  # SSL pretrained
)

# Train on pediatric
# ... same as Stage 2 ...
```

---

## Evaluation & Analysis

### Metrics

**Per-Condition Metrics:**
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- Precision, Recall, F1 at optimal threshold

**Aggregate Metrics:**
- Macro-average AUROC
- Micro-average AUROC
- Per-condition breakdown

### Comparison Table

**File**: `scripts/evaluate_all_models.py`

| Model | Transfer Path | Pediatric AUROC (Macro) | Transfer Gain | Best Conditions | Worst Conditions |
|-------|---------------|-------------------------|---------------|-----------------|------------------|
| From Scratch | None | ? | - | ? | ? |
| SSL → Pediatric | AudioSet+Librispeech → Pediatric | ? | +X% | ? | ? |
| **SSL → Adult → Pediatric** | AudioSet+Librispeech → Adult → Pediatric | ? | +Y% | ? | ? |

**Per-Condition Breakdown:**

| Condition | From Scratch | SSL → Ped | SSL → Adult → Ped | Transfer Gain |
|-----------|--------------|-----------|-------------------|---------------|
| Asthma | ? | ? | ? | ? |
| Allergies | ? | ? | ? | ? |
| Hearing Loss | ? | ? | ? | ? |
| Voice Disorder | ? | ? | ? | ? |
| Neurological | ? | ? | ? | ? |

### Analysis Scripts

**File**: `scripts/analyze_condition_transfer.py`

```python
# For each condition, compute:
# 1. Prevalence in dataset
# 2. AUROC improvement from transfer
# 3. Confidence intervals (bootstrap)

# Hypothesis tests:
# - Does adult transfer help significantly? (paired t-test)
# - Which conditions benefit most? (correlation analysis)
# - Is transfer gain related to condition prevalence?
```

**File**: `scripts/visualize_attention_maps.py`

```python
# Extract attention weights from transformer layers
# Overlay on mel spectrograms
# Compare adult vs pediatric attention patterns
```

---

## Expected Outcomes

### Hypothesis

**Adult self-supervised + supervised transfer will improve pediatric classification by 5-15% over from-scratch, but transfer effectiveness will vary by condition.**

### Shared vs Age-Specific Biomarkers

**Shared (Expected to transfer well):**
- **Asthma**: Breathiness, prolonged exhalation patterns (respiratory physiology similar)
- **Allergies**: Nasal resonance, congestion effects (anatomy scales)
- **Neurological**: Motor control deficits in speech timing (pathophysiology similar)

**Age-Specific (Expected to transfer poorly):**
- **Voice Disorder**: F0 range differs drastically (children vs adults)
- **Hearing Loss**: Cochlear implant usage patterns (more common in children)

### Success Criteria

1. ✅ Adult→Pediatric transfer improves AUROC by ≥5% over from-scratch on ≥3/5 conditions
2. ✅ Identify at least 2 conditions with shared biomarkers (transfer gain >10%)
3. ✅ Identify at least 1 condition with age-specific biomarkers (transfer gain <2%)
4. ✅ Publish findings + open-source `bridge2ai_ssast` library

---

## Implementation Timeline

### Week 1: Data Preparation
- [ ] Download SSAST-Tiny-Frame-400 pretrained model
- [ ] Convert linear → 128-bin mel for adult dataset
- [ ] Convert linear → 128-bin mel for pediatric dataset
- [ ] Parse pediatric multi-select conditions → expanded TSV
- [ ] Manually verify adult condition column names
- [ ] Create condition mapping documentation
- [ ] Compute normalization statistics

### Week 2: Adult Training
- [ ] Implement `AdultBridge2AIDataset` dataloader
- [ ] Implement training script with BCEWithLogitsLoss
- [ ] Train on 5 overlapping conditions
- [ ] Evaluate on adult validation set
- [ ] Save checkpoint: `adult_5cond_tiny.pth`

### Week 3: Pediatric Training
- [ ] Implement `PediatricBridge2AIDataset` dataloader
- [ ] Implement participant-level stratified split
- [ ] Train from adult checkpoint (main experiment)
- [ ] Train from scratch baseline
- [ ] Train from SSL baseline

### Week 4: Analysis & Writeup
- [ ] Compute AUROC/AUPRC for all models
- [ ] Per-condition transfer analysis
- [ ] Age-stratified analysis (4-10 vs 11-17)
- [ ] Attention visualization
- [ ] Bootstrap confidence intervals
- [ ] Write findings report
- [ ] Prepare presentation for professor

---

## Building `bridge2ai_ssast` Library

### Library Scope

A lightweight Python package that:
1. Loads Bridge2AI Parquet features into SSAST-compatible format
2. Handles multi-select condition parsing
3. Provides simple training/fine-tuning APIs
4. Works with both adult and pediatric datasets

### API Design

```python
from bridge2ai_ssast import SSASTFineTuner

# Initialize fine-tuner
finetuner = SSASTFineTuner(
    pretrained_path='ssast/pretrained_model/SSAST-Tiny-Frame-400.pth',
    mel_parquet_path='phenotype/features/mel_128bin_50hz.parquet',
    conditions_path='phenotype/pediatric/pediatric_conditions_expanded.tsv',
    conditions=['has_asthma', 'had_allergies', 'has_hearing_loss'],
    tasks=['long-sounds'],
    target_length=200
)

# Train
results = finetuner.train(
    epochs=50,
    batch_size=16,
    lr=5e-5,
    early_stopping_patience=10
)

# Evaluate
test_metrics = finetuner.evaluate(test_loader)
print(f"Test AUROC: {test_metrics['auroc_macro']:.3f}")

# Analyze
finetuner.plot_per_condition_auroc()
finetuner.visualize_attention(sample_idx=0)
```

### Repository Structure

```
bridge2ai_ssast/
├── __init__.py
├── datasets/
│   ├── adult_dataset.py
│   ├── pediatric_dataset.py
│   └── utils.py (multi-select parsing, splits)
├── models/
│   └── ssast_wrapper.py
├── training/
│   ├── trainer.py
│   └── callbacks.py
├── evaluation/
│   ├── metrics.py
│   └── visualization.py
└── configs/
    └── default_config.yaml
```

---

## References

**SSAST Paper:**
- Gong, Yuan, et al. "SSAST: Self-Supervised Audio Spectrogram Transformer." AAAI 2022.

**Bridge2AI Voice Dataset:**
- Bensoussan, Y., et al. "Bridge2AI-Voice Pediatric Dataset (version 1.0.0)." PhysioNet, 2025.
- Bensoussan, Y., et al. "Bridge2AI-Voice Dataset (version 2.0.1)." PhysioNet, 2025.

**Code:**
- SSAST GitHub: https://github.com/YuanGongND/ssast
- This experiment: `docs/SSAST_Experiment.md`
