# Feature store inventory (canonical with `features/*.json`)

Primary key for tensor Parquet rows: **`participant_id`, `session_id`, `task_name`**. Common columns: **`n_frames`**.

## Frame rates and alignment (Bridge2AI + FEATURES.MD)

| Source | Tensor / column | Time axis | Notes |
|--------|-----------------|-----------|--------|
| Torchaudio | `spectrograms`, `mel_spectrogram`, `mfcc` | **~50 Hz** (effective hop **320** samples @ 16 kHz after `[:, ::2]`) | STFT hop 160 then half of frames kept |
| Torchaudio | `pitch` | **~100 Hz** if one frame per 160-sample hop (verify on your build); treat as **higher rate than 2D** | No `::2` in schema — **resample to mel time** before fusion |
| SPARC | `pitch`, `periodicity`, `loudness` | **50 Hz** | Use `periodicity > τ` for voicing (e.g. 0.5) |
| PPG | `ppg` | **100 Hz** | Shape `[40, T]` |

## Parquet column names by file (dataset `features/`)

| Parquet stem | Key columns |
|--------------|-------------|
| `torchaudio_pitch` | `pitch` (1D list) |
| `torchaudio_mel_spectrogram` | `mel_spectrogram` (nested list 60×T) |
| `torchaudio_mfcc` | `mfcc` (60×T) |
| `torchaudio_spectrogram` | `spectrograms` (201×T) |
| `sparc_periodicity` | `periodicity` |
| `sparc_pitch` | `pitch` |
| `ppgs` | `ppg` |

## Static tabular

- **`static_features.tsv`**: one wide row per clip; join keys `participant_id`, `session_id`, `task_name`; includes openSMILE-style jitter/HNR/phonation_ratio etc.

## Provenance for secondary features

Persist: Torchaudio/SPARC/PPG JSON checksums, voicing mask source (`sparc_periodicity` vs `pitch_bounds`), and target resampling grid length when writing derived tables.
