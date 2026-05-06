
# Vocal Pathology Fingerprinting & Synthetic Pathological Voice Transformation

## Project Overview

This notebook explores a research-oriented pipeline for transforming clean or AI-generated speech into speech that acoustically resembles real pathological voices.

The core idea is:

> Instead of manually designing “voice effects,” derive transformation targets directly from real pathology datasets and optimize synthetic speech so that its measurable acoustic properties statistically align with pathological populations.

The notebook evolved through several major iterations and pivots. Many early sections contain failed or partially successful experiments that informed later approaches.

The final successful methodology is represented primarily by:

- **Trial 4**
- **Final Breath Addition Section**

These sections combine:
- pathology fingerprint extraction
- temporal instability modeling
- WORLD vocoder decomposition/resynthesis
- acoustic-statistical optimization
- fatigue/out-of-breath simulation

The project is effectively a:
- speech pathology analysis pipeline
- acoustic biomarker discovery workflow
- pathological voice synthesis framework

---

# Main Research Objective

The notebook attempts to answer:

> Can we learn measurable acoustic fingerprints from real pathological speech and then generate synthetic speech that statistically resembles those pathologies?

Instead of creating arbitrary “robotic” or “damaged” effects, the system tries to reproduce:
- pitch instability
- aperiodicity
- spectral degradation
- breathiness
- temporal fatigue
- loudness instability

using statistics derived from real patients.

---

# Dataset Structure

The notebook works with an `adult/` dataset containing:

## Phenotype Data
Located under:

```python
adult/phenotype/
```

Includes:
- diagnosis labels
- demographics
- task metadata
- confounders

Examples:
- Parkinson’s disease
- cognitive impairment
- vocal fold paralysis
- ALS
- laryngeal disorders
- controls

---

## Feature Data

Located under:

```python
adult/features/
```

Includes:
- static acoustic features
- mel spectrograms
- pitch trajectories
- loudness trajectories
- periodicity trajectories

Important files:

```python
static_features.tsv
torchaudio_mel_spectrogram.parquet
torchaudio_pitch.parquet
sparc_periodicity.parquet
sparc_loudness.parquet
```

---

# Overall Notebook Structure

The notebook evolved in several stages:

| Stage | Purpose |
|---|---|
| Dataset parsing | Load diagnoses and acoustic features |
| Pathology labeling | Build participant-level labels |
| Cohort matching | Age/sex matching between pathology and controls |
| Fingerprint extraction | Identify acoustic biomarkers |
| Temporal analysis | Analyze trajectories over time |
| Mel analysis | Compare spectrogram structure |
| Trial 1-3 | Early synthetic voice experiments |
| Trial 4 | Final successful WORLD-based transformation |
| Breath section | Adds fatigue/out-of-breath behavior |

---

# Stage 1 — Dataset Parsing & Cleaning

## Goal

Build a unified pathology dataset from many diagnosis TSV files.

---

## Key Utility Functions

### `read_tsv(path)`

Robust TSV reader with fallback parsing.

Handles:
- malformed TSVs
- inconsistent separators
- broken rows

---

### `clean_cols(df)`

Standardizes column names:
- lowercase
- underscores
- removes spaces/hyphens

---

### `standardize_ids(df)`

Normalizes identifier columns into:

```python
participant_id
session_id
recording_id
```

This is necessary because different dataset files use inconsistent naming conventions.

---

### `booleanize(s)`

Converts textual diagnostic values into binary labels.

Maps values like:

```python
yes, positive, diagnosed -> 1
no, negative, control -> 0
```

---

# Stage 2 — Diagnosis Aggregation

## Goal

Convert dozens of diagnosis files into a single participant-level pathology table.

---

## Method

Every diagnosis TSV is interpreted as:

> “If a participant appears in this file, they have the condition.”

For each condition:

```python
dx_<condition_name>
```

is created.

Example:

```python
dx_parkinsons_disease
dx_cognitive_impairment
dx_unilateral_vocal_fold_paralysis
```

All diagnosis tables are merged together.

---

## Multi-Condition Participants

Many participants contain multiple diagnoses.

The notebook computes:

```python
n_conditions
```

to quantify comorbidity overlap.

---

# Stage 3 — Primary Label Assignment

## Goal

Assign a single dominant pathology label to each participant.

---

## Label Priority System

The notebook defines:

```python
LABEL_PRIORITY = [...]
```

This resolves conflicts when a participant has multiple diagnoses.

Example priority:
1. Parkinson’s disease
2. ALS
3. Cognitive impairment
4. Depression
5. Anxiety
6. Vocal disorders
7. Control

The first matching diagnosis becomes the participant’s:
```python
primary_label
```

---

# Stage 4 — Demographic Matching

## Goal

Reduce confounding between pathology and control cohorts.

---

# Focus Pathology

The notebook eventually narrows toward:

```python
unilateral_vocal_fold_paralysis
```

because:
- it has clear acoustic manifestations
- effects are audible
- it strongly affects voice production mechanisms

---

# Matching Procedure

The notebook:
1. loads VF participants
2. loads controls
3. merges demographics
4. compares age distributions
5. filters age ranges
6. compares sex distributions

---

## Why This Matters

Without demographic matching:

- older participants may naturally sound weaker
- sex differences affect pitch/formants
- pathology effects become confounded

The notebook attempts to isolate pathology-specific effects.

---

# Stage 5 — Acoustic Fingerprinting

This is the core scientific section.

---

# Goal

Identify acoustic biomarkers that distinguish pathological voices from controls.

---

# Static Feature Analysis

The notebook loads:

```python
static_features.tsv
```

This contains many engineered acoustic descriptors.

Examples:
- jitter
- shimmer
- HNR
- CPP
- spectral tilt
- pitch statistics
- noise measures

---

# Candidate Feature Selection

The notebook searches for features containing keywords:

```python
jitter
shimmer
hnr
harmonic
noise
pitch
cepstral
spectral
voice
```

This isolates voice-quality-related biomarkers.

---

# Statistical Testing

For each feature:

## Welch’s t-test

```python
ttest_ind(equal_var=False)
```

is used because pathological and control variances differ.

---

## Effect Size

Cohen’s d is computed:

```python
(mean1 - mean2) / pooled_std
```

This identifies:
- not just statistical significance
- but clinically meaningful separation

---

## Multiple Hypothesis Correction

The notebook applies:

```python
FDR Benjamini-Hochberg
```

to reduce false discoveries.

---

# Fingerprint Output

The result is a pathology fingerprint table containing:

- feature name
- pathology mean
- control mean
- effect size
- corrected p-values

This becomes the statistical target for synthesis.

---

# Stage 6 — Visualization & Latent Structure

The notebook explores pathology geometry using:

## Violin Plots

Used to visualize:
- jitter
- shimmer
- CPP
- spectral tilt

between groups.

---

## Correlation Heatmaps

Used to inspect redundancy and relationships between biomarkers.

---

## PCA

Projects participants into low-dimensional “acoustic pathology space.”

Goal:
- see if pathology clusters separately
- determine if pathology occupies distinct acoustic regions

---

# Stage 7 — Temporal Feature Analysis

This section moves beyond static features.

---

# Goal

Model dynamic voice behavior over time.

Instead of:
- average pitch
- average loudness

the notebook examines:
- temporal instability
- fluctuations
- slopes
- variability

---

# Temporal Data Sources

The notebook loads:

```python
torchaudio_pitch.parquet
sparc_periodicity.parquet
sparc_loudness.parquet
torchaudio_mel_spectrogram.parquet
```

These contain frame-wise trajectories.

---

# Core Problem

Temporal arrays have:
- variable lengths
- inconsistent orientations
- nested structures
- NaNs

So the notebook builds normalization utilities.

---

# Important Helper Functions

## `parse_temporal_array(x)`

Safely converts:
- nested lists
- object arrays
- malformed arrays

into numeric NumPy arrays.

---

## `pad_or_crop_spec(x, target_len)`

Standardizes mel spectrogram lengths.

Ensures:
- fixed temporal dimensions
- consistent model comparisons

---

## `pad_or_crop_1d(x, target_len)`

Same idea for:
- pitch
- loudness
- periodicity

---

# Mean Temporal Trajectories

The notebook computes average trajectories for:
- pitch
- periodicity
- loudness

between:
- vocal fold paralysis
- controls

This reveals:
- instability
- flattening
- noisiness
- fatigue patterns

---

# Temporal Summary Statistics

The function:

```python
temporal_summary()
```

extracts:

| Metric | Meaning |
|---|---|
| mean | average level |
| std | variability |
| range | dynamic range |
| slope | trend over time |
| cv | coefficient of variation |

These become higher-level temporal biomarkers.

---

# Stage 8 — Mel Spectrogram Analysis

This section analyzes spectral energy structure.

---

# Goal

Determine how pathological voices distribute energy differently across frequencies.

---

# Mean Spectrograms

The notebook computes:

```python
VF mean spectrogram
Control mean spectrogram
Difference spectrogram
```

This visually highlights:
- weakened harmonics
- excess high-frequency noise
- spectral flattening

---

# Mel-Band Fingerprints

The spectrogram is divided into:
- low bands
- mid bands
- high bands

The notebook computes:
- average energy
- variability
- high/low energy ratios

These become synthesis targets later.

---

# Early Transformation Attempts

The notebook contains multiple transformation experiments.

Most earlier approaches were partially successful but flawed.

---

# Trial 1 — Direct Perturbation

## Approach

Manipulated:
- shimmer
- breath noise
- harmonic degradation
- spectral tilt

using:
- waveform modulation
- STFT manipulation

---

## Problems

Output often sounded:
- robotic
- artificial
- detached from natural speech production

The pathology sounded “added on top” rather than integrated into the voice source.

---

# Trial 2 — Rough Source Model

## Goal

Simulate diplophonia and irregular vocal fold vibration.

---

# Method

Added:
- delayed copies
- harmonic smearing
- stochastic amplitude modulation
- roughness perturbations

This improved realism.

---

## Remaining Problems

Still produced:
- synthetic artifacts
- unstable timbre
- unnatural breathing

The transformation still lacked physiological coherence.

---

# Trial 3 — Fingerprint Matching

This introduced the major conceptual breakthrough.

---

# Core Idea

Instead of hand-tuning effects:

> Optimize transformations so synthetic speech matches real pathology fingerprints.

---

# Audio Fingerprint Function

The notebook defines:

```python
audio_fingerprint()
```

which extracts:
- mel-band ratios
- high-frequency energy
- loudness CV
- pitch CV

from generated speech.

---

# Optimization

The notebook performs:
- parameter sweeps
- brute-force search

over:
- spectral blur
- spectral tilt
- roughness
- breathiness

Then computes:

```python
fingerprint_loss()
```

between generated audio and pathology targets.

---

# Why This Was Important

This changed the project from:
- “audio effects”

to:
- “data-driven pathology synthesis”

This was the key conceptual pivot.

---

# Trial 4 — Final Successful Method

This is the primary successful section.

---

# Major Breakthrough

The notebook switches from:
- direct waveform manipulation

to:

# WORLD Vocoder Decomposition

Using:

```python
pyworld
```

The voice is decomposed into:
1. F0 (pitch)
2. Spectral envelope
3. Aperiodicity

This creates physiologically meaningful control.

---

# Why WORLD Was Important

Earlier methods modified the waveform directly.

WORLD instead manipulates:
- source characteristics
- resonance structure
- noise structure

separately.

This dramatically improved realism.

---

# WORLD Pipeline

## Step 1 — Decomposition

Using:
```python
pw.dio()
pw.stonemask()
pw.cheaptrick()
pw.d4c()
```

Extracts:
- pitch trajectory
- spectral envelope
- aperiodicity structure

---

## Step 2 — Pitch Instability

The notebook creates:
- micro jitter
- slow drift

using:
- Gaussian perturbations
- smooth random walks

This simulates unstable vocal fold vibration.

---

## Step 3 — Spectral Envelope Blur

Gaussian smoothing is applied to:
- spectral envelopes

This weakens harmonic sharpness.

Result:
- breathier
- weaker
- pathological resonance structure

---

## Step 4 — Spectral Tilt

Boosts high-frequency energy.

Simulates:
- turbulence
- leakage
- incomplete glottal closure

---

## Step 5 — Aperiodicity Boost

Pathological voices often contain:
- turbulent airflow
- noise leakage
- reduced harmonic periodicity

The notebook boosts WORLD’s:
```python
aperiodicity
```

to simulate this.

---

## Step 6 — Resynthesis

The modified:
- F0
- spectral envelope
- aperiodicity

are resynthesized into final audio.

This creates much more realistic pathology.

---

# Fingerprint-Guided Optimization

This section introduces:
- beam search
- parameter refinement

---

# Optimization Parameters

The search tunes:
- pitch jitter
- pitch drift
- aperiodicity boost
- spectral blur
- spectral tilt

---

# Objective Function

The loss compares generated fingerprints against:
- real VF pathology fingerprints

using weighted error terms.

---

# Why This Matters

This transforms the pipeline into:

> A pathology-conditioned acoustic optimization system

rather than manual audio engineering.

---

# Final Breath Addition Section

This is the second major successful section.

---

# Goal

Simulate:
- fatigue
- breathlessness
- strained respiration
- incomplete phrase support

This makes the pathology feel biologically real.

---

# Key Insight

Earlier breathing attempts inserted:
- random breaths
- disconnected noise

which sounded fake.

The final approach instead:
- ties breathing to speech energy
- modifies phrase dynamics
- simulates respiratory fatigue

---

# `add_out_of_breath_effect()`

This function adds:

## 1. Speech Slowdown

```python
librosa.effects.time_stretch()
```

Slightly slows speech without changing pitch.

Effect:
- fatigue
- effortful speech

---

## 2. Energy-Conditioned Breath Noise

Breath noise amplitude follows:
```python
RMS speech envelope
```

This makes turbulence feel attached to the voice.

Instead of:
- background noise

it behaves like:
- airflow leakage during phonation

---

## 3. Phrase-Level Fading

Toward phrase endings:
- amplitude fades
- energy weakens

This simulates:
- running out of breath
- reduced respiratory support

---

## 4. Inhalation Pauses

Synthetic inhale segments are inserted:
- between phrases
- with smooth envelopes
- using turbulent noise

This creates:
- recovery breathing
- respiratory strain

---

# Final Result

The final system combines:

| Component | Purpose |
|---|---|
| WORLD vocoder | Physiological decomposition |
| Pitch drift | Instability |
| Aperiodicity | Turbulence |
| Spectral blur | Harmonic degradation |
| Spectral tilt | Breathiness |
| Fingerprint optimization | Match real pathology statistics |
| Breath module | Respiratory fatigue realism |

Together these produce the first convincing pathological transformations in the notebook.

---

# Conceptual Evolution of the Notebook

The notebook evolved through several important conceptual pivots.

---

# Pivot 1 — From Effects to Biomarkers

Early attempts:
- manually added artifacts

Later attempts:
- derived targets from real pathology data

This was the biggest scientific improvement.

---

# Pivot 2 — From Waveform Manipulation to WORLD

Direct waveform edits caused:
- robotic artifacts
- instability
- unnatural harmonics

WORLD allowed:
- physiologically meaningful control

---

# Pivot 3 — From Static Distortion to Temporal Physiology

Later sections realized pathology is not:
- just noisy audio

but:
- dynamic instability
- respiratory fatigue
- phrase-level breakdown
- temporal inconsistency

This led to the breath/fatigue system.

---

# Why Trial 4 Worked

Trial 4 succeeded because it finally aligned:
- acoustic statistics
- physiological modeling
- temporal instability
- source/filter decomposition

Earlier trials manipulated symptoms.

Trial 4 manipulated:
- the vocal production mechanism itself.

---

# Important Acoustic Biomarkers Used

The final pipeline relies heavily on:

| Biomarker | Meaning |
|---|---|
| Jitter | pitch instability |
| Shimmer | amplitude instability |
| HNR | harmonic/noise balance |
| CPP | harmonic organization |
| Spectral tilt | breathiness |
| Pitch CV | pitch variability |
| Loudness CV | instability |
| High/low mel ratio | spectral balance |
| Aperiodicity | turbulent airflow |

---

# Final Architecture Summary

## Input
- clean human speech
- AI-generated speech

---

## Analysis Pipeline
1. load pathology datasets
2. demographic matching
3. extract acoustic biomarkers
4. compute pathology fingerprints
5. compute temporal statistics
6. analyze mel structures

---

## Transformation Pipeline
1. WORLD decomposition
2. perturb source/filter parameters
3. optimize toward pathology fingerprints
4. resynthesize speech
5. add respiratory fatigue

---

## Output
- synthetic pathological speech
- statistically matched to real pathology cohorts

---

# Key Scientific Contributions

This notebook explores several advanced ideas:

## 1. Data-Driven Pathology Synthesis
Using real cohort statistics as synthesis targets.

---

## 2. Acoustic Fingerprinting
Treating pathology as a measurable point in acoustic feature space.

---

## 3. Temporal Pathology Modeling
Capturing instability and fatigue over time.

---

## 4. Physiological Voice Manipulation
Using source/filter decomposition instead of naive waveform effects.

---

# Current Limitations

The notebook still has several limitations:

## Limited Clinical Validation
No perceptual studies or clinician evaluation yet.

---

## Handcrafted Biomarkers
Most fingerprints are manually engineered.

---

## No Learned Latent Representation
No neural embedding model yet.

Potential future directions:
- SSAST
- wav2vec
- HuBERT
- autoencoders
- diffusion synthesis

---

## Single Pathology Focus
The final successful pipeline is mostly tuned for:
- unilateral vocal fold paralysis

Generalization remains untested.

---

# Potential Future Directions

## Neural Fingerprinting
Learn pathology embeddings directly from spectrograms.

---

## Differentiable Optimization
Gradient-based matching instead of brute-force search.

---

## Multi-Pathology Conditioning
Condition synthesis on:
- Parkinson’s
- ALS
- cognitive impairment
- laryngeal disorders

---

## Generative Models
Potential future models:
- diffusion vocoders
- neural codecs
- voice conversion architectures

---

# Final Takeaway

The notebook evolved from:
- manually adding audio artifacts

into:

> A statistically guided pathological voice synthesis framework grounded in real acoustic biomarkers.

The major breakthroughs were:
1. pathology fingerprint extraction
2. WORLD vocoder decomposition
3. optimization against real pathology statistics
4. temporally realistic breath/fatigue modeling

The successful sections are:
- Trial 4
- Final breath augmentation pipeline

These are the first sections where the transformed speech begins to sound:
- physiologically coherent
- statistically grounded
- perceptually realistic.
