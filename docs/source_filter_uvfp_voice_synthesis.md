# Voice synthesis: Source–Filter model (UVFP)

Human speech can be modeled as **two (approximately) independent systems**:

- **Source**: vocal fold vibration in the larynx (periodic pulses + noise)
- **Filter**: vocal tract resonances (throat/mouth/nasal cavity) that shape the spectrum

This is the classic **source–filter model** of speech production.

## Clinical intuition for UVFP

In **unilateral vocal fold paralysis (UVFP)** the pathology is primarily **laryngeal**, so it mainly disrupts the **source**:

- **F0 instability** → higher *jitter* (pitch wobble)
- **Amplitude instability** → higher *shimmer* (loudness wobble)
- **Air leakage / breathiness** → lower *HNR* (more noise / aperiodicity)

The **filter** (vocal tract shape) is typically less affected because UVFP is not a tongue/lips/jaw motor disorder.

**Key idea**: UVFP damages the **SOURCE**, not the **FILTER**.

## Path A — FILTER from mel-spectrogram

The mel-spectrogram encodes the **spectral envelope / resonances** produced by the vocal tract.

However, a magnitude spectrogram lacks **phase**, so reconstructing a waveform requires a phase estimate.

**Griffin–Lim** reconstructs phase by iterating between STFT and iSTFT to find a phase consistent with the magnitude.

Result: a waveform whose **spectral shape** matches the target envelope, but with the well-known “phasey/metallic” artifacts.

This gives a practical **FILTER carrier waveform**.

## Path B — SOURCE as excitation (vocoder-style)

We synthesize a glottal-like excitation using acoustic parameters:

- \(F_0\): pitch / pulse rate
- **Aperiodicity**: noise vs tonality (proxy from HNR: low HNR → high aperiodicity)
- **Spectral envelope**: set flat (we want *excitation*, not vocal tract shaping here)

Then explicitly add UVFP-like perturbations:

- **Jitter**: cycle-to-cycle perturbations in \(F_0\)
- **Shimmer**: cycle-to-cycle amplitude perturbations

Result: an excitation signal that encodes the **pathological laryngeal behavior** (the disordered “engine”).

## Combine — SOURCE × FILTER

In frequency domain, the source–filter combination is multiplication:

\[
SPEECH(f) = SOURCE(f) \times FILTER(f)
\]

One practical implementation (frame-wise):

1. STFT the excitation (Path B) → provides **phase**
2. Replace its **magnitude** with the (mel) spectral envelope (Path A)
3. iSTFT back to time domain

Interpretation:

- **Phase** comes from the pathological excitation (carries disorder)
- **Magnitude / envelope** comes from the mel-spectrogram (carries vocal tract shaping)

## Why this may preserve pathology better than a healthy-speech GAN

A neural vocoder trained only on healthy speech (e.g., HiFi-GAN) has a prior toward **regular periodic voicing** and can “denoise” pathology.

The explicit source–filter approach has **no learned prior** toward healthy voice; pathology is **directly parameterized** (jitter/shimmer/HNR-derived aperiodicity).

## Mapping to *this repo’s* feature files

This repo’s adult dataset uses `participant_id` / `session_id` / `task_name` keys for acoustic feature tables.

### FILTER features

- **Mel spectrogram**: `b2ai_adult_dataset/3.0.0/features/torchaudio_mel_spectrogram.parquet`
  - column: `mel_spectrogram` (shape \([60, time]\))
  - config: 16 kHz, \(n\_fft=400\), `hop_length=160`, then time-subsampled by 2 (effective hop 320 samples = 20 ms)
- **Linear spectrogram** (optional alternative): `.../features/torchaudio_spectrogram.parquet`
  - column: `spectrograms` (shape \([201, time]\))

### SOURCE features (UVFP-related correlates)

From `b2ai_adult_dataset/3.0.0/features/static_features.tsv` (OpenSMILE-like summary features):

- **Jitter (summary)**: `jitterLocal_sma3nz_amean`
- **Shimmer (summary)**: `shimmerLocaldB_sma3nz_amean`
- **HNR (summary)**: `HNRdBACF_sma3nz_amean`

Per-frame pitch (if you want an \(F_0(t)\) track rather than a single summary):

- `b2ai_adult_dataset/3.0.0/features/sparc_pitch.parquet`
  - column: `pitch` (Hz, 50 Hz frame rate)
- periodicity / voicing confidence:
  - `b2ai_adult_dataset/3.0.0/features/sparc_periodicity.parquet`
  - column: `periodicity` in \([0,1]\)

### Suggested “config keys” (if you implement synthesis code)

Instead of the placeholder CSV keys (`f0_mean`, `hnr_mean`, etc.), prefer:

- `participant_id` / `session_id` / `task_name`
- `jitterLocal_sma3nz_amean`
- `shimmerLocaldB_sma3nz_amean`
- `HNRdBACF_sma3nz_amean`
- `pitch` track from SPARC (or derive a mean F0 from the track)

