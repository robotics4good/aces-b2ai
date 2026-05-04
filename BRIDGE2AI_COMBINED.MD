# Bridge2AI Voice — Pediatric dataset: comprehensive reference

This document consolidates **working-session notes** (chat-derived summaries), **feature documentation**, **phenotype/task structure**, **Torchaudio vs static vs SPARC guidance**, and **analysis outputs** (breathing/airway EDA) into one place. Paths are relative to the dataset folder that contains `README.md`, `features/`, and `phenotype/`.

It also integrates a full **signal processing and acoustic feature engineering curriculum** developed across working sessions, grounding each concept in the source-filter model, respiratory physiology, and the specific constraints of pediatric voice data.

**Last assembled:** 2026 (from project working notes; re-run scripts to refresh numeric tables in `analysis_outputs/`).

---

## Table of contents

1. [Chat / session narrative (what was covered)](#1-chat--session-narrative-what-was-covered)
2. [Repository snapshot](#2-repository-snapshot)
3. [Phenotype: sessions, recordings, acoustic tasks](#3-phenotype-sessions-recordings-acoustic-tasks)
4. [Breathing and airway labels (phenotype)](#4-breathing-and-airway-labels-phenotype)
5. [Torchaudio: semantics, derived metrics, clinical hypotheses](#5-torchaudio-semantics-derived-metrics-clinical-hypotheses)
6. [Outputs: scripts and generated files](#6-outputs-scripts-and-generated-files)
7. [EDA: static + Torchaudio vs airway conditions (summary)](#7-eda-static--torchaudio-vs-airway-conditions-summary)
8. [Features: modalities, derivations, library parameters (full)](#8-features-modalities-derivations-library-parameters-full)
9. [Signal processing foundations: what each feature actually measures](#9-signal-processing-foundations-what-each-feature-actually-measures)
10. [The source-filter model: harmonics vs formants](#10-the-source-filter-model-harmonics-vs-formants)
11. [Formants: physical meaning, rates of change, clinical relevance](#11-formants-physical-meaning-rates-of-change-clinical-relevance)
12. [Jitter family: cycle-to-cycle perturbation measures](#12-jitter-family-cycle-to-cycle-perturbation-measures)
13. [Shimmer family: amplitude perturbation measures](#13-shimmer-family-amplitude-perturbation-measures)
14. [HNR and CPP: periodicity and breathiness](#14-hnr-and-cpp-periodicity-and-breathiness)
15. [Phonation ratio and voiced run statistics](#15-phonation-ratio-and-voiced-run-statistics)
16. [Research context: Kaur et al. (2023) RRVB model](#16-research-context-kaur-et-al-2023-rrvb-model)
17. [Pediatric-specific considerations](#17-pediatric-specific-considerations)
18. [Secondary feature derivation: full plan](#18-secondary-feature-derivation-full-plan)
19. [Dynamical systems features: derivatives, polynomials, phase portraits](#19-dynamical-systems-features-derivatives-polynomials-phase-portraits)
20. [Temporal modulation and spectral dynamics](#20-temporal-modulation-and-spectral-dynamics)
21. [Feature priority ranking for pediatric respiratory modeling](#21-feature-priority-ranking-for-pediatric-respiratory-modeling)
22. [Secondary feature library: architecture and implementation plan](#22-secondary-feature-library-architecture-and-implementation-plan)
23. [Critical technical considerations](#23-critical-technical-considerations)
24. [MelHuBERT TransformerBranch](#section-24--melhubert-transformerbranch)
    - [24.1 Vendored source](#241-vendored-source)
    - [24.2 Checkpoint + distribution shift](#242-checkpoint)
    - [24.3 Mel 60→80 bin zero-padding](#243-mel-spectrogram-input-60-bin--80-bin-zero-padding)
    - [24.4 SPARC augmentation two-mode design](#244-sparc-augmentation-two-mode-design)
    - [24.5 FiLM conditioning (why not MHA)](#245-static-feature-conditioning-film-not-mha)
    - [24.6 Default sparse keys](#246-default-sparse-conditioning-keys)
    - [24.7 Data splitting strategy](#247-data-splitting-strategy)
    - [24.8 Class imbalance pos_weight](#248-class-imbalance-pos_weight)
    - [24.9 Progressive encoder unfreezing](#249-progressive-encoder-unfreezing)
    - [24.10 load_pretrained device safety](#2410-load_pretrained-device-safety)
    - [24.11 TransformerBranchConfig reference](#2411-transformerbranchconfig-quick-reference)
    - [24.12 Training CLI reference](#2412-training-cli-reference)
    - [24.13 Known pitfalls](#2413-known-pitfalls-all-addressed)

---

## 1. Chat / session narrative (what was covered)

- **Repository summary:** Bridge2AI Voice Pediatric v1.0.0 — phenotype tables (demographics, medical conditions, questionnaires, task metadata) plus `features/` (static TSV + JSON sidecars; Parquet tensors for Torchaudio, SPARC, PPG when present). Primary keys for tensor rows: `participant_id`, `session_id`, `task_name`. `SHA256SUMS.txt` lists the full release manifest.
- **Task taxonomy:** Activities come from `phenotype/task/acoustic_task.tsv` field `acoustic_task_name` (e.g. `passage`, `picture`, `naming-animals`, `long-sounds`, `ready-for-school`, …). `acoustic_task` is one row per participant × session × activity type; `recording.tsv` is one row per audio file (many clips per activity). Join: `session_id`, `acoustic_task_id` = `recording_acoustic_task_id`.
- **Activity "categories" plot:** A bar chart grouped activities into six informal themes (conversation, word lists, voice play, picture tasks, read speech, time/counting). Those group labels were chosen for visualization, not taken from the Bridge2AI website or from a category column in the dataset JSONs.
- **SPARC vs Torchaudio:** SPARC outputs (pitch, periodicity, loudness, EMA) are ML / model-based; treat as approximate with possible domain shift. Torchaudio features are classical DSP (STFT, mel, MFCC, pitch) with parameters fixed in `features/*.json` — better for reproducibility and interpretation.
- **Torchaudio + static for airway work:** Derived participant-level metrics from Torchaudio Parquet were joined to breathing/airway flags parsed from `peds_mc_breathing_conditions` and free text in `peds_mc_conditions`. Mann–Whitney tests and a narrative report were produced; high-dimensional FDR often leaves no features significant at strict thresholds — use effect sizes and multivariate ranking (RF) as exploratory.
- **Extensive EDA plan:** Implemented in `scripts/eda_airway_comprehensive.py` — participant-level `agg_mean_*` and `agg_stdw_*` (mean and across-clip std of static features), merge age and Torchaudio aggregates, correlation blocks, PCA, violins for asthma/intubation, RF/LASSO for outcomes with sufficient sample size.
- **Signal processing curriculum:** Across working sessions we built up a complete conceptual understanding of: the source-filter model, harmonics vs formants, jitter/shimmer families, HNR/CPP, phonation ratio, F0 depletion dynamics, temporal modulation spectra, spectral flatness, phase portraits, and polynomial trajectory features. All of this is documented in Sections 9–21 below.
- **Research grounding:** The Kaur et al. (2023) JMIR paper on Respiratory-Responsive Vocal Biomarkers (RRVB) was reviewed in detail, including Appendix A methodology. Key finding: a logistic regression model trained on 6-second sustained "ahh" phonations from asthma patients generalizes zero-shot to COPD, ILD, cough, and COVID-19. Pediatric patients explicitly identified as untested future work.
- **Feature priority ranking:** A tiered ranking (S, A, B, C) of features by pediatric robustness and mechanistic grounding was developed (see Section 21).
- **Secondary feature library architecture:** A full Python library architecture was planned covering MFCC secondary features, mel spectrogram secondary features, spectrogram secondary features, pitch secondary features, and cross-modal features (see Section 22).

---

## 2. Repository snapshot

| Area | Contents |
|------|-----------|
| `README.md` | Dataset description, folder layout, Python loading examples |
| `SHA256SUMS.txt` | Cryptographic hashes for release files |
| `phenotype/pediatric/` | Demographics, medical conditions (incl. breathing checklist + free text) |
| `phenotype/pediatric_questionnaire/` | Voice-related surveys (JSON + TSV) |
| `phenotype/task/` | `acoustic_task`, `recording`, `session` (JSON schemas + TSV) |
| `features/` | `static_features.tsv` + `*.json`; paired `*.parquet` for tensors when distributed |
| `analysis_outputs/` | Torchaudio participant stats, `eda/` EDA tables and figures |

**Scale (typical):** on the order of ~300 participants in phenotype; ~23k clip rows in `static_features.tsv`; ~28k recording rows; Torchaudio Parquet row counts may differ slightly by modality (e.g. mel vs pitch).

---

## 3. Phenotype: sessions, recordings, acoustic tasks

### Session (`session.tsv` / `session.json`)

`session_id`, `session_duration` (seconds), `session_is_control_participant` (yes/no), `session_status` (e.g. completed).

### Recording (`recording.tsv` / `recording.json`)

Per file: `recording_id`, `recording_name`, `recording_session_id`, `recording_acoustic_task_id`, duration, size (KiB), microphone, input gain, `recording_profile_name` (e.g. `Speech`), profile version.

### Acoustic task (`acoustic_task.tsv` / `acoustic_task.json`)

Per logical task in a session: `acoustic_task_id`, `acoustic_task_name`, `acoustic_task_session_id`, duration, cohort (e.g. Pediatrics), status.

### Example activity names (`acoustic_task_name`)

Includes (non-exhaustive): `ready-for-school`, `favorite-show-movie-game`, `favorite-food`, `outside-of-school`, `naming-animals`, `naming-food`, `repeat-words`, `role-namings`, `long-sounds`, `silly-sounds`, `noisy-sounds`, `picture`, `picture-description`, `pictures-and-doors`, `sentence`, `passage`, `days`, `months`, `123s`.

---

## 4. Breathing and airway labels (phenotype)

Parsed for analysis from `peds_mc_breathing_conditions` and `peds_mc_conditions` (case-insensitive text), aligned with REDCap-style values:

| Flag | Typical text / code |
|------|---------------------|
| asthma | `asthma` |
| intubation | `intubat` substring (e.g. intubated) |
| airway_surgery | `airway surgery` |
| chronic_cough | `chronic cough` |
| rsv | word-boundary `rsv` |
| recurrent_croup | `recurrent croup` or croup + recurrent |
| bronchiolitis | `bronchiolitis` |
| vocal_cord_webs | `vocal cord` + `web` |
| tracheostomy | `tracheostomy` |

**Counts in merged EDA table (300 participants):** asthma=33, intubation=18, airway_surgery=8, chronic_cough=5, rsv=4, recurrent_croup=3, bronchiolitis=2, vocal_cord_webs=1, tracheostomy=1.

---

## 5. Torchaudio: semantics, derived metrics, clinical hypotheses

### What each modality represents

- **Linear spectrogram:** STFT log-magnitude (dB); 201 freq bins; privacy time subsampling `::2` on stored tensor.
- **Mel spectrogram:** 60 mels; same STFT window/hop and `::2` subsampling.
- **MFCC:** 60 coefficients; same front-end.
- **Pitch:** F0 Hz, `detect_pitch_frequency`, band 80–500 Hz; no `::2` documented for pitch — align timelines explicitly when fusing with mel/spec.

### Derived metrics (per clip, then often mean per participant)

| Family | Metrics | Interpretation |
|--------|---------|----------------|
| Pitch | voiced fraction, mean/std/IQR F0 | Voicing amount, central tendency, variability |
| Mel | mean, std, low/mid/high energy ratios, modulation | Level, tilt, HF noise proxies, frame-to-frame change |
| MFCC | delta mean, c0 mean | Spectral dynamics, coarse energy envelope |
| Spectrogram | low/mid/high ratios, centroid, mean dB | Linear-frequency analog of mel summaries |

### Exploratory hypotheses (not diagnostic)

| Condition | Acoustic angle (hypothesis) |
|-----------|------------------------------|
| Asthma | Broadband noise, less stable voicing; mel HF / pitch variability |
| Intubation history | Breathiness/hoarseness; HNR-like proxies, pitch breaks |
| Airway surgery | Turbulence / stridor-like HF energy |
| Chronic cough | Bursts, higher modulation / delta MFCC |
| RSV / bronchiolitis | Context-dependent; often acute |
| Recurrent croup | HF noise proxies (mel) |
| Vocal cord webs | Breathy, limited F0 range, lower voiced fraction |
| Tracheostomy | Very small n — descriptive only |

### Extraction settings (from JSON)

```
sample_rate_hz: 16000
n_fft: 400
win_length: 400
hop_length: 160
subsample_time: every 2nd frame on 2D tensors (::2)
mel_bins: 60
mfcc_coeffs: 60
spec_freq_bins: 201
pitch_band_hz: (80, 500)
```

**Parquet note:** linear spectrogram column may be `spectrogram` (singular) in Parquet while JSON field name can read `spectrograms`.

---

## 6. Outputs: scripts and generated files

| Script | Purpose |
|--------|---------|
| `scripts/torchaudio_breathing_analysis.py` | Stream Torchaudio Parquets → per-clip derived stats → participant means; merge flags; Mann–Whitney; `TORCHAUDIO_BREATHING_REPORT.md` |
| `scripts/eda_airway_comprehensive.py` | Static `agg_mean_*` / `agg_stdw_*`, merge Torchaudio + age + flags; EDA CSVs + figures + `EDA_AIRWAY_FINDINGS.md` |

**Key outputs**

- `analysis_outputs/participant_torchaudio_stats.csv` — Torchaudio participant aggregates + flags
- `analysis_outputs/eda/participant_master_table.csv` — full merged participant table for modeling
- `analysis_outputs/eda/participant_static_aggregates.csv` — static aggregates only
- `analysis_outputs/eda/group_tests_participant_level.csv` — univariate tests + Cliff's delta + FDR
- `analysis_outputs/eda/figures/` — co-occurrence heatmap, correlation heatmaps, PCA, violins
- `analysis_outputs/eda/EDA_AIRWAY_FINDINGS.md` — narrative + tables
- `analysis_outputs/TORCHAUDIO_BREATHING_REPORT.md` — Torchaudio-focused report

---

## 7. EDA: static + Torchaudio vs airway conditions (summary)

**Design:** 300 participants after inner join of labels, static aggregates, demographics, Torchaudio stats. Static features summarized as mean across clips (`agg_mean_*`) and std across clips (`agg_stdw_*`) per participant.

**Multiple testing:** Benjamini–Hochberg FDR per condition across all numeric tests. With ~280+ features, often no feature survives FDR < 0.1; interpret raw p-values, Cliff's delta, and Random Forest importance as exploratory.

**Asthma — lowest raw p-values (examples):** see full table in `analysis_outputs/eda/group_tests_participant_level.csv`. EDA report snapshot showed top hits often among `agg_stdw_*` (cross-clip variability), e.g. F2-related stddevNorm, alpha ratio UV, MFCC slopes.

**Random Forest (exploratory importance):**

- **Asthma:** strong weight on `agg_stdw_F2frequency_sma3nz_stddevNorm`, `agg_mean_logRelF0-H1-H2_sma3nz_stddevNorm`, `agg_stdw_mean_b1_loc`, spectral slopes, loudness peaks/sec, etc.
- **Intubation:** `agg_mean_logRelF0-H1-A3_sma3nz_amean`, level/unvoiced segment variability (`agg_stdw_equivalentSoundLevel_dBp`, `agg_stdw_MeanUnvoicedSegmentLength`), intensity-related features.

**Recommended shortlist (hypothesis + EDA — validate externally):**

- **Static:** `agg_mean_mean_hnr_db`, jitter/shimmer aggregates, `agg_mean_spectral_tilt`, `agg_mean_spectral_slope`, CPP, Hammarberg/alpha, `agg_mean_phonation_ratio`, pause/speaking rate; also consider selected `agg_stdw_*` when modeling inconsistency across clips.
- **Torchaudio:** `m_mel_high_ratio`, `m_mel_modulation`, `m_mel_centroid_bin`, `p_pitch_f0_std_hz`, `p_pitch_voiced_fraction`, `s_spec_high_ratio`, `s_spec_centroid_bin`.

**Confounding:** use `analysis_outputs/eda/age_spearman_top_features.csv` and `task_kruskal_features.csv`; control age and task in formal models.

---

## 8. Features: modalities, derivations, library parameters (full)

Data layout common to almost all feature stores: primary key `[participant_id, session_id, task_name]` plus `n_frames` and one or more Parquet columns (nested lists, zstd compression). Companion JSONs in `features/` are the authoritative schema + extraction config.

### 8.1 File-by-file breakdown

#### `features/static_features.json` + `features/static_features.tsv`

One wide row per clip (tabular). Content groups:

- **IDs / text:** `participant_id`, `session_id`, `task_name`, `transcription`
- **openSMILE-style LLDs (GeMAPS-like naming):** F0 semitone features, loudness, spectral flux, MFCCs 1–4, jitter/shimmer, HNR, harmonic ratios, F1–F3 frequency/bandwidth/relative amplitude, voiced vs unvoiced spectral summaries, segment timing stats, equivalent level
- **Extended scalars:** speaking/articulation rates, phonation ratio, pause metrics, Praat F0/intensity stats, HNR, spectral slope/tilt, CPP, formant track summaries, spectral moments (gravity, std, skewness, kurtosis), full jitter/shimmer family, STOI, PESQ, SI-SDR

#### `features/torchaudio_spectrogram.json`

Tensor `spectrograms` — log-magnitude spectrogram, float32, shape `[201, T]`, dims frequency × time, units dB. STFT config: n_fft 400, win_length 400, hop_length 160. Post-processing: `data[:, ::2]` — half time frames kept → effective hop 320 samples (~25 Hz frame rate after subsample). Provenance: `torchaudio.transforms.Spectrogram` 2.8.0.

#### `features/torchaudio_mel_spectrogram.json`

Tensor `mel_spectrogram`, shape `[60, T]`, mel_bin × time. Config: sample_rate 16000, same STFT, n_mels 60. Same `::2` subsampling. Provenance: `torchaudio.transforms.MelSpectrogram` 2.8.0.

#### `features/torchaudio_mfcc.json`

Tensor `mfcc`, shape `[60, T]`, coefficient × time; n_mfcc 60 with melkwargs matching mel front-end. Same `::2` subsampling. Provenance: `torchaudio.transforms.MFCC` 2.8.0.

#### `features/torchaudio_pitch.json`

Tensor `pitch`, 1D `[T]`, units Hz. Config: sample_rate 16000, freq_low 80, freq_high 500. Serialization: `list<float>` (no nested 2D). **No `::2` subsampling** — must resample/interpolate when fusing with mel/spec tensors.

#### `features/ppgs.json`

Tensor `ppg`, shape `[40, T]`, phoneme × time, values probability [0,1], row-normalized per frame. 40 ARPAbet-style labels ending in `silence`. Time axis: sampling_rate_hz 100. Provenance: `ppgs` package 0.0.9.

#### SPARC (`speech-articulatory-coding` 0.1.0, model_name `multi`)

All SPARC 1D streams share sampling_rate_hz 50.

| File | Series | Shape / notes |
|------|--------|----------------|
| `sparc_pitch.json` | `pitch` | 1D Hz, value_range [50, 550]; CREPE-derived |
| `sparc_periodicity.json` | `periodicity` | 1D [0, 1] voicing confidence |
| `sparc_loudness.json` | `loudness` | 1D; z-scored waveform amplitude per 20ms windows |
| `sparc_ema.json` | `ema` | 2D `[T, 12]` — TD/TB/TT/LI/UL/LL each X/Y |

### 8.2 Parameters to carry in a secondary features library

| Source | Parameters to store |
|--------|---------------------|
| Torchaudio STFT family | `sample_rate`, `n_fft`, `win_length`, `hop_length`, `subsample_time_axis` (factor 2), `n_mels`, `n_mfcc` |
| Torchaudio pitch | `freq_low`, `freq_high`, `sample_rate` |
| PPG | `sample_rate`, frame rate 100 Hz, 40 labels order, `ppgs` version |
| SPARC shared | `speech-articulatory-coding` version, `model_name`, 50 Hz grid, CREPE fmin/fmax |
| SPARC loudness | 20 ms window, utterance-wise z-score behavior |
| SPARC EMA | 12 channel order (TDX, TDY, …) |
| Static | Pipeline name / config hash, column schema version |

---

## 9. Signal processing foundations: what each feature actually measures

### What a periodic signal is

A periodic signal repeats at a fundamental frequency F0. The time between repetitions is the period T = 1/F0. For voiced speech, the vocal folds open and close at F0, producing one complete glottal cycle per period. At 300 Hz (a typical child's F0), there are 300 cycles per second, each period lasting ~3.3 ms.

### Why sustained phonation is the ideal task

The 6-second sustained "ahh" task maximally loads the respiratory-phonatory coupling. The subglottal pressure — driven by lung recoil and respiratory muscles — directly drives vocal fold vibration. Any respiratory impairment (airflow obstruction, reduced lung capacity, chest tightness) leaves acoustic fingerprints in the sustained sound. The Kaur et al. (2023) RRVB model achieves cross-condition generalization specifically because it detects a shared physiological substrate: impaired respiratory drive, regardless of whether that impairment is from asthma, COPD, COVID-19, or other conditions.

### The temporal subsampling constraint

The Torchaudio tensors have every other time frame removed for privacy (`::2`). This means:
- Original hop: 160 samples at 16 kHz = 10 ms
- Effective hop after subsampling: 320 samples = 20 ms
- Effective frame rate: ~50 Hz native → ~25 Hz after subsampling

Glottal cycles in children (200–400 Hz F0) have periods of 2.5–5 ms. These are completely invisible in the subsampled tensors. **The only source of cycle-level perturbation information is the static jitter and shimmer features, which were computed from the raw waveform before any subsampling.** Delta features from MFCC/mel/spectrogram capture 20ms-scale dynamics, not cycle-level dynamics.

---

## 10. The source-filter model: harmonics vs formants

This is the foundational conceptual distinction for interpreting all speech acoustic features. Harmonics and formants are completely different physical phenomena.

### Harmonics — from the source (vocal folds)

When the vocal folds vibrate at F0, they produce a buzzing pulse shape that mathematically decomposes into F0, 2×F0, 3×F0, 4×F0... These are harmonics, labeled H1, H2, H3, etc. They are always integer multiples of F0, perfectly evenly spaced by F0 on the frequency axis. H1 = F0 = fundamental frequency. H2 = 2F0. And so on up the spectrum. Adding more harmonics builds up the buzzing quality of the glottal pulse — a pure sine wave is just H1; a natural voice has many harmonics.

### Formants — from the filter (vocal tract)

The vocal tract (mouth, throat, nasal cavity) is a tube roughly 17 cm long in adults (shorter in children). Like any tube, it has natural resonant frequencies — specific frequencies where sound bounces back and forth and reinforces. These resonances are formants. They are labeled F1, F2, F3 (note: F here means formant, completely different from F0 which is fundamental frequency). Formants are NOT multiples of F0. They are NOT related to F0 at all. They are determined purely by the geometry of the vocal tract.

For a neutral "ahh" vowel, approximate values: F1 ≈ 700–800 Hz, F2 ≈ 1200 Hz, F3 ≈ 2500–2700 Hz. None of these have any mathematical relationship to F0.

### The source-filter equation

```
What you hear = SOURCE (harmonics) × FILTER (formants)
```

The vocal folds produce harmonics at F0, 2F0, 3F0... The vocal tract then amplifies any harmonics that happen to fall near a formant frequency and attenuates the rest. The formant frequencies don't change the harmonic spacing — they change which harmonics are loud or quiet. This is how different vowels are produced: same harmonic structure (same F0), different formant positions (different mouth/tongue shape), different spectral envelope shape.

### Why both matter for respiratory analysis

Harmonics and their perturbations (jitter, shimmer, HNR) tell you about the stability of vocal fold vibration driven by breath pressure. Formant bandwidths tell you about energy absorption in the system, which changes when respiratory mechanics alter the coupling between lungs and vocal tract. These are genuinely independent signals about different parts of the same system.

---

## 11. Formants: physical meaning, rates of change, clinical relevance

### F1 — jaw opening and tongue height (vertical)

F1 is controlled by jaw opening and tongue vertical position. Low F1 means high tongue, closed jaw (as in "ee"). High F1 means low tongue, open jaw (as in "ahh", ~700–800 Hz for adults, higher for children). For a sustained "ahh" F1 should be stable throughout. Any drift reflects the speaker losing articulatory control.

**F1 bandwidth is the most diagnostically relevant bandwidth measure.** When the glottis is more open (as in breathy phonation from impaired breath support), the trachea below acts as a coupled resonator that bleeds energy specifically out of F1. This subglottal coupling widens F1 bandwidth. The mechanism: respiratory impairment → less stable subglottal pressure → more open or irregular glottis → more subglottal energy coupling → wider F1 bandwidth. This is a direct pathway from respiratory mechanics to formant bandwidth.

**Rate of change dF1/dt:** How fast F1 is changing. Should be near zero for a stable sustained vowel. Non-zero means articulatory drift from fatigue or effort changes.

### F2 — tongue front-back position and lip rounding

F2 is controlled by tongue horizontal position and lip rounding. High F2 means tongue pushed forward (as in "ee", ~2200 Hz). Low F2 means tongue pulled back or lips rounded (as in "oo", ~850 Hz). For "ahh", F2 is intermediate (~1200 Hz). F2 stability reflects the speaker's ability to maintain a fixed articulatory configuration under effort.

**Rate of change dF2/dt:** Tongue horizontal movement speed. In cross-formant analysis, correlated drift of dF1/dt and dF2/dt suggests a common underlying cause (e.g., whole-body vocal tract tension shifting with respiratory effort) rather than independent noise.

### F3 — tongue tip position and larynx height

F3 sits at 2500–3000 Hz and is sensitive to tongue tip position (retroflex movements lower F3 substantially) and larynx height. It is harder to measure reliably due to lower SNR at high frequencies. F3 bandwidth is sensitive to larynx height changes that occur under respiratory effort.

### Feature relevance in the dataset

In the EDA results, `agg_stdw_F2frequency_sma3nz_stddevNorm` was among the top features for asthma — this is the cross-clip variability in F2 variation within clips. This makes sense: children with asthma may show more inconsistent articulatory control across recording sessions, and/or more within-clip F2 drift, than healthy controls.

`agg_stdw_mean_b1_loc` (cross-clip variability in F1 bandwidth) also showed RF importance for asthma — consistent with the subglottal coupling hypothesis: impaired respiratory support leads to variable F1 bandwidth.

### Derived features to compute from formant data

```python
# Rate of change (per-frame derivative)
dF1_dt = np.gradient(f1_track)   # Hz per frame
dF2_dt = np.gradient(f2_track)
dF3_dt = np.gradient(f3_track)

# Cross-formant coupling
f1_f2_correlation = np.corrcoef(dF1_dt, dF2_dt)[0,1]

# Segment drift
f1_drift = f1_track[-len(f1_track)//3:].mean() - f1_track[:len(f1_track)//3].mean()

# Trajectory curvature (polynomial fit)
t = np.linspace(0, 1, len(f1_track))
a0, a1, a2 = np.polyfit(t, f1_track, 2)
# a1 = linear drift rate, a2 = curvature (does drift accelerate?)
```

---

## 12. Jitter family: cycle-to-cycle perturbation measures

### Physical mechanism

When vocal folds vibrate at F0, each open-close cycle should ideally be the same length. Jitter measures how much the period length varies from cycle to cycle. The mechanism connecting respiratory disease to jitter: respiratory impairment → unstable subglottal pressure → irregular driving force for glottal vibration → uneven cycle lengths → measurable jitter.

All jitter variants are normalized as ratios (divided by mean period), making them relatively scale-invariant across vocal tract sizes and F0 values. This is critical for pediatric work.

### The four variants

```
local_jitter = mean(|T_{i+1} - T_i|) / mean(T_i)
```
Compares adjacent pairs of periods. Captures the fastest, most random perturbations. Most sensitive but also most affected by pitch tracker noise.

```
rap_jitter (Relative Average Perturbation) = mean(|T_i - mean(T_{i-1}, T_i, T_{i+1})|) / mean(T_i)
```
Compares each cycle to its 3-cycle local average. Cancels truly random frame-to-frame noise while preserving real physiological perturbation. Generally considered more robust than local_jitter as a clinical measure.

```
ppq5_jitter (5-point Period Perturbation Quotient) = mean(|T_i - mean(T_{i-2}...T_{i+2})|) / mean(T_i)
```
Smooths over a 5-cycle window. Captures perturbations correlated over slightly longer timescales — more like slow drift in respiratory support than purely random jitter. Relevant for airway disease where instability may have temporal structure related to breathing cycles.

```
ddp_jitter = mean(|(T_{i+2} - T_{i+1}) - (T_{i+1} - T_i)|) / mean(T_i)
```
Second derivative of the period sequence. Measures whether the change between consecutive periods is itself changing. High DDP means the perturbation pattern is irregular and unpredictable. DDP ≈ 3 × RAP for Gaussian noise; deviations from this ratio indicate structured perturbation patterns.

### Timescale sensitivity summary

| Measure | Timescale | Best captures |
|---------|-----------|---------------|
| local_jitter | Adjacent cycles | Fast random perturbation, noise-sensitive |
| RAP | 3-cycle window | Physiological perturbation, robust to tracker noise |
| PPQ5 | 5-cycle window | Slower drift from respiratory drive instability |
| DDP | 2nd-order difference | Irregularity of the irregularity, chaotic instability |

### Pediatric considerations

Children have baseline jitter values naturally slightly higher than adults due to still-developing laryngeal motor control. Adult-derived absolute thresholds for "abnormal" jitter will over-flag healthy children. Use jitter values relative to a healthy pediatric reference group of the same age, or use normalized/ranked versions of these features.

Since jitter is computed at the waveform level (before subsampling), it has full cycle-level temporal resolution even in this dataset.

---

## 13. Shimmer family: amplitude perturbation measures

Shimmer is the amplitude analog of jitter. Instead of measuring variation in period length, it measures variation in cycle-to-cycle amplitude. The mechanism: respiratory impairment → variable subglottal pressure → variable amplitude of each glottal pulse → shimmer.

### Variants

```
local_shimmer = mean(|A_{i+1} - A_i|) / mean(A_i)
localDB_shimmer = mean(20 * log10(A_{i+1} / A_i))  [in dB]
apq3_shimmer = mean(|A_i - mean(A_{i-1}, A_i, A_{i+1})|) / mean(A_i)
apq5_shimmer = 5-point amplitude perturbation quotient
apq11_shimmer = 11-point amplitude perturbation quotient
dda_shimmer = mean(|(A_{i+1} - A_i) - (A_i - A_{i-1})|) [second difference]
```

APQ11 is particularly useful for respiratory work because the 11-cycle window captures longer-timescale amplitude modulation that can reflect slower respiratory fluctuations (breathing cycle effects on subglottal pressure).

---

## 14. HNR and CPP: periodicity and breathiness

### HNR (Harmonics-to-Noise Ratio)

HNR measures the ratio of periodic (harmonic) energy to aperiodic (noise/turbulence) energy. Respiratory impairment → turbulent airflow through obstructed or narrowed airways → more aperiodic noise mixed into phonation → lower HNR.

HNR is measured in dB. Higher is better (more periodic, cleaner voice). Typical healthy adult values: 20–25 dB. Values below 10 dB indicate substantially breathy or hoarse voice.

HNR is computed from the autocorrelation of the signal — it compares the autocorrelation peak (periodicity strength) to the residual after subtracting the periodic component. This requires accurate F0 estimation, which is a limitation for high-F0 pediatric voices.

### CPP (Cepstral Peak Prominence)

CPP measures how prominent the fundamental period peak is in the cepstrum (the "spectrum of the spectrum"). It is computed by finding the peak in the cepstrum at the quefrency corresponding to F0, and measuring how much it rises above a regression line fit to surrounding cepstral values.

**CPP is preferred over HNR for pediatric work.** Both measure roughly the same thing (periodic vs aperiodic energy ratio) but CPP is substantially more robust to F0 estimation failures. Since children's high F0 causes pitch trackers to fail more often, CPP's robustness to those failures is a real practical advantage. Clinical research on pediatric voice quality has consistently found CPP more reliable than HNR for children under approximately 12.

`cepstral_peak_prominence_std` (temporal variability of CPP) captures how much voice quality fluctuates during the recording — a measure the mean alone misses. For a sustained vowel, the speaker is asked to hold steady; CPP variability directly quantifies deviation from that ideal.

---

## 15. Phonation ratio and voiced run statistics

### Physical meaning

For a sustained "ahh" task, the ideal is one continuous voiced segment spanning the full recording. Every break is a moment the speaker failed to maintain breath support. In healthy children, breaks essentially don't occur (they can easily sustain breath for 6 seconds). In children with respiratory impairment, breaks occur because either: (a) they run out of adequate breath support and must inhale, or (b) decreasing subglottal pressure drops below the threshold needed to maintain vocal fold vibration.

### Feature definitions

```python
phonation_ratio = phonation_time / total_duration    # fraction in [0,1]

# From voiced run sequence:
voiced_run_count            # number of separate voiced segments
MeanVoicedSegmentLengthSec  # average duration of voiced segments
StddevVoicedSegmentLengthSec # consistency of segment lengths
voiced_run_longest_fraction = longest_run / total_voiced_frames  # [0,1]
```

### Why these are pediatric-robust

These features are computed from whether phonation is occurring or not — a decision that does not depend on absolute F0, vocal tract size, age, or sex. A voiced segment at 250 Hz and one at 400 Hz are both just voiced segments. This makes them essentially anatomy-agnostic, unlike formant frequencies or even jitter which have age-dependent baselines.

### Interpreting patterns

- `voiced_run_count` = 1, high `phonation_ratio`: healthy, single continuous phonation
- High `voiced_run_count`, similar total `phonation_ratio`: fragmented phonation, impaired breath support
- `voiced_run_longest_fraction` high: one long segment then fragments — initial capacity but rapid depletion
- `voiced_run_longest_fraction` low: consistently short bursts — baseline airflow restriction throughout

---

## 16. Research context: Kaur et al. (2023) RRVB model

### Paper overview

Kaur et al. (2023), *Journal of Medical Internet Research* — "Development and Validation of a Respiratory-Responsive Vocal Biomarker-Based Tool for Generalizable Detection of Respiratory Impairment."

A logistic regression model trained on 6-second sustained "ahh" phonations from ~1700 asthma patients (data from 20+ hospitals in India) was tested zero-shot on COVID-19 patients across 4 clinical sites in the US and India (497 participants). The same model also generalized to COPD, ILD, and persistent cough without retraining.

### Performance

| Condition | OR | Sensitivity | Specificity |
|-----------|-----|-------------|-------------|
| Asthma (training) | ~4 | 55–75% | 55–75% |
| COPD | ~9 | — | — |
| ILD | ~3.1 | — | — |
| Persistent cough | ~3.9 | — | — |
| COVID-19 (new test) | 4.64 | 73.2% | 62.9% |

### Why it generalizes: the shared physiological substrate

The model detects changes in vocal acoustic features that are common across all these conditions: impaired respiratory drive, airflow obstruction, and chest symptoms (shortness of breath, cough, chest tightness). These symptoms alter subglottal pressure in ways that leave consistent acoustic fingerprints regardless of the underlying disease.

### Appendix A: feature engineering methodology

Features were extracted from the sustained "ahh" using three families:
- Filter banks (time-frequency energy distribution)
- Cepstral analysis (MFCCs and variants)
- Local discriminant bases (LDB) — a wavelet-based data-driven method that finds the time-frequency representation maximally discriminating between classes

Feature selection: scored by lower bound of 90% CI of correlation with asthma label across 19 cross-validation folds, stratified independently by phone manufacturer, research coordinator, language, and random splits. Only features robust across all confounders were retained.

Final model: at most 5 features per gender, logistic regression. Gender-specific models trained separately. The deliberate parsimony (≤5 features) is credited as a key reason for generalization — small models avoid overfitting to training-data-specific artifacts.

### Key limitation: pediatric patients not tested

The paper explicitly states pediatric patients were not tested and require additional studies. This is the primary gap the Bridge2AI pediatric dataset addresses.

### Symptom-stratification finding

Sensitivity for COVID-positive patients with active respiratory symptoms: 78.4% vs 67.4% for those without respiratory symptoms at recording time. Even fully asymptomatic COVID-positive patients were detected at 68% sensitivity — the vocal signal captures subclinical physiological changes before self-report.

---

## 17. Pediatric-specific considerations

### Anatomical differences from adults

| Factor | Adults | Children |
|--------|--------|----------|
| F0 range | 80–200 Hz | 200–500 Hz |
| Vocal tract length | ~17 cm | ~10–14 cm |
| Formant frequencies | Lower | Higher (shorter tract) |
| Jitter/shimmer baseline | Lower | Higher even healthy |
| Breath support | Fully developed | Still developing |

### Implications for feature transfer

Features with adult-derived absolute frequency thresholds will not transfer to children. Specifically problematic:
- Alpha ratio (50–1000 Hz vs 1–5 kHz bands): designed for adult vocal tract resonances
- Hammarberg index (0–2 kHz vs 2–5 kHz): same issue
- Absolute formant frequencies (F1, F2, F3 center frequencies): strongly age/sex dependent

Features that are ratios or normalized within-recording are more likely to transfer:
- Jitter family (all ratios, normalized by mean period)
- Shimmer family (ratios normalized by mean amplitude)
- HNR and CPP (ratios/dB measures)
- Phonation ratio and voiced run statistics (dimensionless fractions)
- F0 depletion slope (rate of change within recording, not absolute level)

### Pitch tracker range

The Torchaudio pitch tracker uses band 80–500 Hz. For young children (F0 up to 400–500 Hz), this range is barely adequate. Validate carefully — check what fraction of voiced frames have pitch estimates near the 500 Hz ceiling, as clipping may occur.

SPARC pitch covers 50–550 Hz (better for children). Consider using SPARC periodicity as the voicing mask and SPARC pitch as the F0 source for pediatric analyses.

### F0 octave errors

Pitch trackers can return F0/2 for high-F0 voices (octave error: mistakes second harmonic for fundamental). Detection heuristic:

```python
def detect_octave_errors(pitch, voiced_mask):
    voiced_f0 = pitch[voiced_mask]
    median_f0 = np.median(voiced_f0)
    # Frames at roughly half the median are likely octave errors
    octave_low_mask = voiced_mask & (pitch < median_f0 * 0.6) & (pitch > 0)
    corrected = pitch.copy()
    corrected[octave_low_mask] *= 2.0
    return corrected
```

### Normalization strategy for pediatric data

Three approaches, in increasing sophistication:

1. **Z-score within recording** — subtract mean, divide by std, computed from that recording. Removes absolute scale effects. Use for: formant center frequencies, F0 mean.

2. **Percentile normalization** — rank-transform features within the dataset. More robust to outliers. Use for: jitter, shimmer (already ratios but outlier-sensitive).

3. **Age/sex residualization** — fit a reference model on healthy controls predicting feature value from age and sex, use the residual as the actual feature. Most principled for pediatric data but requires sufficient healthy control sample. Use for: F0 mean, formant frequencies, any feature with strong developmental trajectory.

---

## 18. Secondary feature derivation: full plan

### From MFCC tensor (60 coefficients × T frames, 20ms hop)

| Feature | Computation | Respiratory interpretation |
|---------|-------------|---------------------------|
| `mfcc_delta_coeff{i}_abs_mean` | `mean(|gradient(mfcc[i])|)` | Cepstral instability magnitude at 20ms scale |
| `mfcc_delta2_coeff{i}_abs_mean` | Second gradient abs mean | Cepstral acceleration |
| `mfcc_coeff{i}_skewness` | `scipy.stats.skew(mfcc[i])` | Distributional asymmetry |
| `mfcc_coeff{i}_kurtosis` | `scipy.stats.kurtosis(mfcc[i])` | Heavy-tail behavior, impulsive events |
| `cms_residual_energy` | `mean((mfcc - mfcc.mean(axis=1, keepdims=True))**2)` | Within-recording variability after mean removal |
| `mfcc_segment_drift_l2` | `||mean(last_third) - mean(first_third)||` | Breath depletion across full task |
| `mfcc_cov_eigval_rank{k}` | Eigenvalues of 60×60 covariance | Variance structure of cepstral space |
| `mfcc_cov_top3_explained_var` | `eigvals[:3].sum() / total_var` | Compactness of cepstral variance |

**Note on CMS (Cepstral Mean Subtraction):** Subtracting the per-recording mean MFCC vector before computing statistics removes the speaker-level baseline (which varies enormously across pediatric ages due to vocal tract size). This is critical for cross-age generalization.

### From mel spectrogram (60 mel bins × T frames)

| Feature | Computation | Respiratory interpretation |
|---------|-------------|---------------------------|
| `vmap_mean_abs` | `mean(|diff(mel, axis=1)|)` | Mean frame-to-frame spectral change |
| `vmap_{band}_energy` | Band-specific vmap energy | Respiratory effects in specific frequency bands |
| `mel_bin_entropy_mean` | Mean Shannon entropy per bin over time | Temporal stability of spectral energy |
| `mel_centroid_slope` | Linear fit slope of centroid trajectory | Spectral drift as breath depletes |
| `mel_energy_depletion` | `(early_third - late_third) / mean` | Breath support decrease, normalized within-recording |
| `mel_phonation_stability` | `frame_energy.std() / frame_energy.mean()` | CV of frame energy |
| `mel_{band}_energy_slope` | Linear fit over time per band | Per-band energy trajectory |

**Mel band groups (relative to 60 mel bins):**
```python
MEL_BAND_GROUPS = {
    "low":  (0,  14),   # fundamental + low harmonics
    "mid":  (15, 39),   # main formant region
    "high": (40, 59),   # higher formants + breathiness/turbulence
}
```

### From spectrogram (201 frequency bins × T frames)

| Feature | Computation | Respiratory interpretation |
|---------|-------------|---------------------------|
| `spec_flux_skewness` | Skewness of frame-to-frame diff distribution | Impulsive events in phonation |
| `spec_flux_kurtosis` | Kurtosis of spectral flux | Heavy-tail: irregular glottal events |
| `spec_band_instability_{band}` | Variance over time per frequency band | Per-band energy stability |
| `spec_turbulence_band_var` | Variance in 4000–8000 Hz over time | Airway turbulence proxy |
| `spec_flatness_mean` | `mean(geo_mean(frame) / arith_mean(frame))` | Mean tonality vs noisiness |
| `spec_flatness_slope` | Linear trend in flatness | Is signal becoming noisier over time? |

**Frequency bands for spectrogram (201 bins, 16 kHz → 40 Hz/bin):**
```python
SPEC_BAND_GROUPS = {
    "fundamental": (0,   12),    # 0–500 Hz
    "formant1_2":  (13,  49),    # 500–2000 Hz
    "formant3":    (50,  99),    # 2000–4000 Hz
    "turbulence":  (100, 200),   # 4000–8000 Hz
}
```

### From pitch track (T frames, 20ms hop for Torchaudio; must resample from SPARC 50 Hz if using SPARC)

| Feature | Computation | Respiratory interpretation |
|---------|-------------|---------------------------|
| `voiced_ratio_from_pitch` | `voiced_frames / total_frames` | Phonation continuity |
| `f0_depletion_slope` | `polyfit(voiced_times, voiced_f0, 1)[0]` | Dropping subglottal pressure during task |
| `f0_depletion_r2` | R² of linear fit | How linear (vs chaotic) the depletion is |
| `f0_trajectory_entropy` | Entropy of F0 histogram | Pitch trajectory complexity |
| `voiced_run_longest_fraction` | `max_run / total_voiced` | Proportion of time in best sustained segment |
| `f0_local_instability_mean` | Mean rolling std of F0 | Local F0 variability |
| `f0_track_jitter_mean` | `mean(|diff(voiced_f0)|) / mean(voiced_f0)` | Track-level jitter, robust for high-F0 voices |

### Cross-modal features

| Feature | Sources | Computation | Interpretation |
|---------|---------|-------------|----------------|
| `mfcc0_f0_coupling` | MFCC + pitch | Pearson corr(MFCC[0], F0) over voiced frames | Source-filter coupling: high = destabilized phonation |
| `flux_per_phonation_unit` | Static | spectralFlux / phonation_ratio | Flux normalized by phonation amount |
| `perturbation_composite` | Static | `(jitter + shimmer) / abs(HNR)` | Combined voice quality impairment score |
| `cpp_hnr_ratio` | Static | `CPP_mean / abs(HNR_mean)` | Consistency check between two periodicity measures |
| `mel_energy_depletion` | Mel | `(early_energy - late_energy) / mean` | Breath depletion, fully within-recording normalized |

---

## 19. Dynamical systems features: derivatives, polynomials, phase portraits

This section connects to the dynamical systems framing that the research group specializes in. The respiratory acoustic system can be treated as a dynamical system: F0 and related trajectories trace paths in state space, and the geometry of those paths encodes information about the system's stability.

### Polynomial features of F0 trajectory

Model F0(t) as a polynomial:

```
F0(t) = a0 + a1*t + a2*t^2 + a3*t^3 + ε(t)
```

The coefficients are features:
- **a1** = linear depletion slope. The most basic dynamical feature. Negative = F0 dropping as breath depletes. Directly measures the rate of subglottal pressure decrease.
- **a2** = curvature. Does the depletion accelerate (a2 < 0) or decelerate (a2 > 0)? A child who starts depleting slowly but accelerates has worse respiratory reserve than one who depletes linearly.
- **a3** = inflection. Is there a change in the depletion pattern? Could indicate recovery behavior or sudden collapse.
- **ε(t)** = residual after polynomial fit. Its statistics (std, entropy, autocorrelation) are themselves features capturing the high-frequency irregular component.

The same polynomial feature decomposition can be applied to any trajectory: F1(t), F2(t), mel band energies, MFCC[0](t), spectral flatness(t). This generalizes the depletion slope concept to the entire feature space.

### Derivatives as features

```python
def compute_derivatives(track):
    d1 = np.gradient(track)       # velocity: how fast is it changing?
    d2 = np.gradient(d1)          # acceleration: how fast is the velocity changing?
    d3 = np.gradient(d2)          # jerk: how fast is the acceleration changing?
    return d1, d2, d3
```

Feature extraction from derivatives:
```python
# Mean absolute values at each derivative order
mean_abs_d1 = np.mean(np.abs(d1))   # overall instability magnitude
mean_abs_d2 = np.mean(np.abs(d2))   # acceleration magnitude
max_abs_d1 = np.max(np.abs(d1))     # worst-case velocity spike
```

Higher derivatives amplify instability (more sensitive to subtle impairment) but also amplify noise. Smooth before differentiating using a running mean window of 3–5 frames before computing second and third derivatives.

### Phase portrait

The phase portrait plots F0(t) on the x-axis and F0'(t) (first derivative) on the y-axis. Each point is one moment in time, connected by lines. Color by time (early = light, late = dark).

Healthy phonation: a tight, nearly circular orbit — the system is in a stable attractor, returning to the same state after small perturbations.

Impaired phonation: a diffuse cloud with directional drift — the attractor is broader (more instability) and the cloud shows drift (depletion).

Features derivable from the phase portrait:
```python
# Area of the orbit (proxy for instability)
from scipy.spatial import ConvexHull
points = np.column_stack([f0_track, d1_track])
hull = ConvexHull(points)
orbit_area = hull.volume  # area in 2D

# Centroid drift (early vs late cloud center)
early_centroid = points[:len(points)//3].mean(axis=0)
late_centroid = points[-len(points)//3:].mean(axis=0)
centroid_drift = np.linalg.norm(late_centroid - early_centroid)
```

### Takens delay embedding

A more sophisticated attractor reconstruction: plot (F0(t), F0(t-τ)) for a time lag τ. This reconstructs the system's attractor from a single time series without needing to compute derivatives explicitly. For healthy phonation, this produces a tight oval. For impaired phonation, a diffuse irregular cloud. Choosing τ: use the first minimum of the mutual information of the F0 time series.

### Sample entropy

Sample entropy quantifies the unpredictability of the F0 time series without requiring assumptions about linearity:

```python
def sample_entropy(x, m=2, r=0.2):
    """r is tolerance (fraction of std): typically 0.1-0.25"""
    r_abs = r * np.std(x)
    # Count template matches of length m and m+1
    # SampEn = -log(A/B) where A=matches of length m+1, B=matches of length m
    # (use antropy or nolds library for efficient implementation)
    pass
```

Higher sample entropy = more irregular, more unpredictable F0 = more impaired. This is a nonlinear measure that captures what jitter and shimmer measure at cycle level but applied to the longer F0 trajectory.

---

## 20. Temporal modulation and spectral dynamics

### Temporal modulation spectrum

The most novel secondary feature family. For any mel bin's time series across T frames, compute its Fourier transform. The result is the modulation spectrum of that frequency band — at what temporal rates is that band's energy fluctuating?

```python
def temporal_modulation_spectrum(mel, sr_frames=50.0):
    """
    mel: (n_mels, T) array
    sr_frames: frame rate in Hz (25 Hz after ::2 subsampling)
    Returns: (n_mels, T//2) modulation spectrum
    """
    # Window each mel bin's time series and FFT
    n_mels, T = mel.shape
    tms = np.zeros((n_mels, T//2))
    window = np.hanning(T)
    for b in range(n_mels):
        fft_result = np.abs(np.fft.rfft(mel[b] * window))
        tms[b] = fft_result[:T//2]
    return tms

# Summarize into modulation bands
mod_freqs = np.fft.rfftfreq(T, d=1/sr_frames)
def modulation_band_energy(tms, mod_freqs):
    return {
        "dc_to_1hz":    tms[:, mod_freqs < 1].mean(),
        "1hz_to_4hz":   tms[:, (mod_freqs >= 1) & (mod_freqs < 4)].mean(),
        "4hz_to_8hz":   tms[:, (mod_freqs >= 4) & (mod_freqs < 8)].mean(),
        "above_8hz":    tms[:, mod_freqs >= 8].mean(),
    }
```

Physiologically meaningful modulation bands:
- 0–1 Hz: very slow drift (depletion, fatigue effects)
- 1–4 Hz: respiratory rhythm (~0.2–0.5 Hz) and slow instability
- 4–8 Hz: tremor range (pathological oscillation in vocal tract)
- 8+ Hz: fast flutter from airway instability

### Spectral flatness

Spectral flatness per frame = geometric_mean(spectrum) / arithmetic_mean(spectrum). Values near 0 mean the energy is concentrated in a few bins (tonal, harmonic structure dominant). Values near 1 mean energy is spread uniformly (noise-like).

```python
def spectral_flatness_trajectory(spec):
    """spec: (n_bins, T) array"""
    # Add small epsilon to avoid log(0)
    spec_pos = np.abs(spec) + 1e-10
    geo_mean = np.exp(np.log(spec_pos).mean(axis=0))
    arith_mean = spec_pos.mean(axis=0)
    flatness = geo_mean / arith_mean
    return flatness  # shape (T,)
```

Flatness trajectory statistics:
- `spec_flatness_mean`: average tonality vs noise across the recording
- `spec_flatness_std`: variability in tonality
- `spec_flatness_slope`: is the signal becoming progressively noisier (depleting breath support allows more turbulent airflow into phonation)?

### MFCC coefficient autocorrelation

For each MFCC coefficient, the autocorrelation at lag k tells you: if the coefficient is at value X now, how correlated is it with its value k frames ago (k×20ms ago)?

High autocorrelation at lag 5 (100ms): the cepstral state persists for at least 100ms — stable, predictable phonation. Low autocorrelation at all lags: rapidly fluctuating cepstral state — unstable.

```python
def mfcc_autocorr_features(mfcc, lags=[1, 2, 5, 10]):
    features = {}
    for coeff in range(min(13, mfcc.shape[0])):
        x = mfcc[coeff] - mfcc[coeff].mean()
        acf = np.correlate(x, x, mode='full')
        acf = acf[len(acf)//2:] / (acf[len(acf)//2] + 1e-10)
        for lag in lags:
            if lag < len(acf):
                features[f'mfcc{coeff}_autocorr_lag{lag}'] = float(acf[lag])
    return features
```

---

## 21. Feature priority ranking for pediatric respiratory modeling

Based on: mechanistic grounding in respiratory-phonatory coupling, pediatric robustness (normalized, anatomy-agnostic), support from Kaur et al. findings, and EDA results in this dataset.

### Tier S — do not skip

These are the most directly connected to the respiratory-phonatory mechanism, normalized within-recording, and supported by both the paper and the pediatric literature.

1. **Full jitter family** (`local_jitter`, `rap_jitter`, `ppq5_jitter`, `ddp_jitter`): Direct readout of unstable subglottal pressure. Computed at waveform level (full temporal resolution). All normalized as ratios.

2. **CPP mean + std** (`cepstral_peak_prominence_mean`, `cepstral_peak_prominence_std`): Most important single feature for pediatric work. Measures periodic vs aperiodic energy ratio, more robust than HNR to F0 estimation failures at high pediatric frequencies. The std captures temporal variability in voice quality during the task.

3. **Phonation ratio + voiced run statistics** (`phonation_ratio`, `MeanVoicedSegmentLengthSec`, `StddevVoicedSegmentLengthSec`, derived `voiced_run_longest_fraction`, `voiced_run_count`): Most interpretable features in the set. Completely anatomy-agnostic. For a sustained phonation task, any break is a direct behavioral signal of impaired breath support.

4. **F0 depletion slope** (derived from pitch track): Captures temporal trajectory — the rate of F0 decline as lung volume decreases. Novel feature not captured by summary statistics. Directly measures respiratory support capacity. Robust across pediatric F0 range because it is a rate of change, not an absolute value.

### Tier A — high value

5. **HNR mean + std** (`HNRdBACF_sma3nz_amean`, `HNRdBACF_sma3nz_stddevNorm`, `mean_hnr_db`, `std_hnr_db`): Fundamental voice quality. Two sources (openSMILE and Praat) provide cross-validation. Lower priority than CPP only because of pediatric F0 estimation sensitivity.

6. **Full shimmer family** (`local_shimmer`, `apq3_shimmer`, `apq5_shimmer`, `apq11_shimmer`, `dda_shimmer`): Amplitude-perturbation analog of jitter. APQ11 particularly relevant for capturing respiratory cycle effects on amplitude.

7. **MFCC segment drift** (`mfcc_segment_drift_l2`, `mfcc{0-3}_segment_drift`): Captures breath depletion at the cepstral level over the full task duration. Normalized within-recording.

8. **Mel energy depletion** (`mel_energy_depletion`): (Early energy − late energy) / mean energy. Fully within-recording normalized. Simple, robust, directly measures breath support decrease.

### Tier B — solid supporting features

9. **Delta abs_mean for first 4 MFCC coefficients** (`mfcc_delta_coeff{0-3}_abs_mean`): Mean absolute rate of cepstral change at 20ms scale. Captures instability at phrase-level timescales.

10. **Perturbation composite** = (jitter + shimmer) / |HNR|: Combined impairment score. Useful for exploratory analysis and as a single summary.

11. **F0 trajectory entropy** (`f0_trajectory_entropy`): Nonlinear complexity measure of pitch trajectory. Captures what mean/std don't: the overall irregularity pattern.

12. **Mel phonation stability** (`mel_phonation_stability`): CV of frame-level mel energy. Simple, robust, normalized.

### Tier C — worth including but secondary

13. **Spectral flux statistics** (`spectralFlux_sma3_amean`, `spec_flux_kurtosis`): Flux mean already in static features; kurtosis captures impulsive events.

14. **MFCC-F0 coupling** (`mfcc0_f0_coupling`): Novel cross-modal feature. Clear physiological rationale but sensitive to pitch tracking quality.

15. **F1 bandwidth** (`F1bandwidth_sma3nz_amean`): Subglottal coupling proxy. Moderately age-dependent but less so than F1 center frequency.

16. **Spectral flatness trajectory** (`spec_flatness_mean`, `spec_flatness_slope`): From spectrogram. Captures progressive increase in noisiness.

### Features requiring caution or normalization for pediatric use

- `mean_f0_hertz`, `F0semitoneFrom27.5Hz_sma3nz_amean`: Strong developmental confound. Normalize by age/sex before use.
- `F1frequency_sma3nz_amean`, `mean_f1_loc`, `mean_f2_loc`: Vocal tract length dependent. Use rates of change or within-recording normalized versions.
- `alphaRatioV_sma3nz_amean`, `hammarbergIndexV_sma3nz_amean`: Adult frequency bands. May not transfer.
- `stoi`, `pesq`: Not validated for pediatric clinical use.

---

## 22. Secondary feature library: architecture and implementation plan

### Directory structure

```
respiratory_features/
├── __init__.py
├── core/
│   ├── base.py              # Abstract BaseExtractor, ExtractionResult dataclass
│   ├── registry.py          # Feature registry / dispatcher
│   └── validation.py        # Input validation, shape checks, voiced mask
├── extractors/
│   ├── mfcc_secondary.py    # Delta, delta-delta, CMS, statistics, covariance
│   ├── mel_secondary.py     # V-maps, A-maps, entropy, centroid trajectory
│   ├── spectrogram_secondary.py  # Flux stats, harmonic tracking, band instability
│   ├── pitch_secondary.py   # F0 depletion, entropy, voiced runs, track jitter
│   └── crossmodal.py        # MFCC-F0 coupling, interaction features
├── utils/
│   ├── framing.py           # Segment splitting, voiced/unvoiced masking
│   ├── normalization.py     # CMS, z-score, percentile, age/sex residualization
│   ├── pitch_utils.py       # F0 range validation, octave error detection
│   └── pediatric.py         # Developmental normalization utilities
├── pipeline.py              # End-to-end orchestration
├── schema.py                # Output feature namespace definitions
└── config.py                # Band definitions, F0 range, default parameters
```

### Core data class

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ExtractionResult:
    features: Dict[str, float]
    metadata: Dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    # NaN used for hard failures (too few frames, all unvoiced)
    # warnings carries soft failures: computed but flagged
```

### Naming convention

Strict namespace: `{source}_{feature_group}_{statistic}`

Examples:
- `mfcc_delta_coeff0_mean` (NOT `mfcc0_delta_mean`)
- `mel_entropy_low_band_mean`
- `pitch_depletion_slope_hz_per_frame`
- `cross_mfcc0_f0_coupling_pearson`
- `spec_turbulence_band_variance`

This enables programmatic selection: `[f for f in features if f.startswith("pitch_")]`

### Pipeline interface

```python
class FeatureExtractionPipeline:
    def extract(
        self,
        static_features: Dict[str, float],
        mfcc: np.ndarray,          # (60, T) — 20ms hop after ::2
        mel_spec: np.ndarray,       # (60, T) — 20ms hop after ::2
        spectrogram: np.ndarray,    # (201, T) — 20ms hop after ::2
        pitch: np.ndarray,          # (T,) — Torchaudio: 20ms, SPARC: 20ms (50Hz)
        participant_id: str = None,
        session_id: str = None,
        age_years: float = None,    # for pediatric normalization
        sex: str = None,            # 'M'/'F' for gender-specific thresholds
    ) -> Dict: ...
```

### Synthetic test cases

```python
# For unit testing extractors with known expected values
import numpy as np

def make_pure_sine(f0=300, duration=6.0, sr=16000):
    """Should give: near-zero jitter, high HNR, stable MFCC trajectory."""
    t = np.linspace(0, duration, int(duration * sr))
    return np.sin(2 * np.pi * f0 * t)

def make_noisy_sine(f0=300, snr_db=10, duration=6.0, sr=16000):
    """Should give: degraded HNR, increased spectral flatness, higher entropy."""
    signal = make_pure_sine(f0, duration, sr)
    noise_power = np.var(signal) / (10 ** (snr_db / 10))
    return signal + np.random.randn(len(signal)) * np.sqrt(noise_power)

def make_depleting_sine(f0=300, amplitude_decay=0.5, duration=6.0, sr=16000):
    """Should give: negative mel_energy_depletion, negative f0_depletion_slope."""
    t = np.linspace(0, duration, int(duration * sr))
    amplitude = 1.0 - amplitude_decay * (t / duration)
    return amplitude * np.sin(2 * np.pi * f0 * t)

def make_interrupted_phonation(f0=300, n_breaks=5, duration=6.0, sr=16000):
    """Should give: high voiced_run_count, short MeanVoicedSegmentLength."""
    signal = make_pure_sine(f0, duration, sr)
    n_samples = len(signal)
    break_len = int(0.1 * sr)
    for pos in np.random.choice(n_samples // 2, n_breaks, replace=False):
        signal[pos*2:pos*2 + break_len] = 0
    return signal
```

---

## 23. Critical technical considerations

### 1. Temporal subsampling and what it means

The `::2` subsampling on Torchaudio 2D tensors (mel, MFCC, spectrogram) halves temporal resolution. Effective hop = 20ms. Glottal cycles in children = 2.5–5ms. Therefore: individual glottal cycles are invisible in all tensors. Delta features from tensors capture 20ms-scale dynamics (phrase-level). Waveform-level cycle perturbation is only available from static jitter/shimmer (computed before subsampling).

Document this in feature names: `mfcc_delta_*` = 20ms-scale, `local_jitter` = cycle-scale. These are complementary, not redundant.

The pitch tensor has no documented `::2` subsampling — verify alignment explicitly before fusing with mel/MFCC features.

### 2. Voiced frame detection

Use a single consistent voiced mask across all extractors, derived from one source. Recommended hierarchy:
- First choice: SPARC periodicity > threshold (0.5 recommended)
- Second choice: Torchaudio pitch > 0 and within expected range
- Third choice: openSMILE VoicedSegmentsPerSec / MeanVoicedSegmentLengthSec for static analysis

```python
def get_voiced_mask(pitch, f0_min=80.0, f0_max=500.0):
    return (pitch > f0_min) & (pitch < f0_max) & np.isfinite(pitch)
```

Pass `voiced_mask` as an argument to all extractors. Do not let each extractor define voicing independently.

### 3. Pediatric F0 range handling

Torchaudio pitch band: 80–500 Hz. For young children (F0 up to 400–500 Hz), check for clipping near the ceiling. SPARC pitch band: 50–550 Hz — better for pediatric voices.

Validation check:

```python
PEDIATRIC_F0_RANGE = (80.0, 500.0)  # Hz
def validate_pitch_range(pitch, expected_range=PEDIATRIC_F0_RANGE, warn_threshold=0.20):
    voiced = pitch[(pitch > 0) & np.isfinite(pitch)]
    if len(voiced) == 0:
        return ["no voiced frames detected"]
    out_of_range = ((voiced < expected_range[0]) | (voiced > expected_range[1])).sum()
    fraction = out_of_range / len(voiced)
    warnings = []
    if fraction > warn_threshold:
        warnings.append(f"{fraction:.1%} of voiced frames outside expected range {expected_range} Hz")
    return warnings
```

### 4. Normalization strategy

Three strategies:

```python
# Z-score within recording (for absolute-scale features: formant frequencies, F0 mean)
def zscore_within_recording(features):
    return (features - features.mean()) / (features.std() + 1e-10)

# Percentile normalization (for perturbation measures: jitter, shimmer)
from scipy.stats import rankdata
def percentile_normalize(features):
    return rankdata(features) / len(features)

# Age/sex residualization (most principled for pediatric generalization)
def age_sex_residualize(feature_value, age_years, sex, reference_model):
    expected = reference_model.predict([[age_years, 1 if sex == 'F' else 0]])[0]
    return feature_value - expected
```

### 5. Cross-modal time alignment

| Source | Frame rate | After ::2 |
|--------|------------|-----------|
| Torchaudio mel/MFCC/spec | ~50 Hz (10ms hop) | ~25 Hz (20ms hop) |
| Torchaudio pitch | ~50 Hz (no ::2 documented) | ~50 Hz |
| SPARC all streams | 50 Hz | 50 Hz |
| PPG | 100 Hz | 100 Hz |

Use `np.interp` with normalized time axis [0,1] for resampling between rates. Document which pitch track is source of truth for voicing mask in each experiment.

### 6. Missing data handling

Clinical recordings will have sessions where modalities fail — pitch returns all zeros, recording too short for thirds computation, etc. Use NaN for hard failures (allows downstream models to handle explicitly). Implement `fill_value` config per feature group. The `ExtractionResult.warnings` list carries soft failures with explanatory strings.

### 7. EDA-driven confound control

From analysis outputs, known confounds to control in formal models:
- **Age**: use `analysis_outputs/eda/age_spearman_top_features.csv` to identify features with strong age correlation
- **Task type**: use `task_kruskal_features.csv`; different tasks (sustained phonation vs free speech) have very different acoustic statistics
- **Microphone and recording gain**: from `recording.tsv` — include as covariates or stratify

---

## Document maintenance

- **Refresh EDA numbers:** run `python3 scripts/eda_airway_comprehensive.py` and replace Section 7 with the new `EDA_AIRWAY_FINDINGS.md` if you need publication-fresh tables.
- **Refresh Torchaudio participant stats:** run `python3 scripts/torchaudio_breathing_analysis.py` first if Parquets change.
- **Secondary feature library:** implement from Section 22 architecture; unit-test with synthetic signals from Section 22 test cases before running on real data.
- **Pediatric normalization:** build healthy-control reference models (Section 23 point 4) once sufficient healthy control data is confirmed in the dataset.
- **Single source for feature definitions:** treat this file as master; update Section 18 when secondary feature library is finalized.

---

## Section 24 — MelHuBERT TransformerBranch

This section documents the full architecture and all design decisions for the
`TransformerBranch` pipeline used in `scripts/train_binary_classifier.py`.

---

### 24.1 Vendored source

The MelHuBERT model source is vendored at:

    src/aces_b2ai/models/melhubert_vendor/

Contains `model.py`, `module.py`, `fairseq_code/`, and `pytorch_code/` copied from
[nervjack2/MelHuBERT](https://github.com/nervjack2/MelHuBERT) (ASRU 2023, MIT License).
The only modifications are import-path fixes (bare module names → relative imports).

```python
from aces_b2ai.models.melhubert_vendor import MelHuBERTModel, MelHuBERTConfig
from aces_b2ai.models.branches import TransformerBranch
from aces_b2ai.models.model_config import TransformerBranchConfig
```

---

### 24.2 Checkpoint

Stored at `weights/melhubert-20ms-stg2.ckpt` (1.08 GB, gitignored via `weights/*.ckpt`).

| Property | Value |
|---|---|
| Training step | 582,000 (stage 2, fully converged) |
| `feat_emb_dim` | 80 (two stacked 40-bin fbank frames, 20 ms period) |
| `encoder_layers` | 12 |
| `encoder_embed_dim` | 768 |
| `encoder_attention_heads` | 12 |
| State dict tensors | 201 |
| Pretraining corpus | LibriSpeech 360h — **adult read speech** |

**Distribution shift warning:** The checkpoint was trained on clean adult English read speech.
Our data is pediatric clinical speech with voice and hearing disorders. Key mismatches:

| Axis | LibriSpeech (pretraining) | Our data |
|---|---|---|
| Age | Adults | Children |
| F0 range | ~80–200 Hz | ~200–500 Hz (kids ~2× higher) |
| Vocal tract | Adult-length | Shorter → shifted formants |
| Task | Read speech | Sustained vowels, clinical tasks |
| Condition | Healthy | Cochlear implant, voice disorder, etc. |

Despite this, frozen-encoder probing achieves val AUC ~0.85 on hearing_cochlear — the general
spectral shape representations transfer. Fine-tuning top encoder layers pushes further.

Additional available checkpoints:

| Name | `feat_emb_dim` | Link |
|---|---|---|
| MelHuBERT-20ms 360h stage 2 *(this file)* | 80 | https://drive.google.com/file/d/11wzYf8u9pXPvQyQU2Wodx79W31Ka2e0Z/view |
| MelHuBERT-10ms 360h stage 1 | 40 | https://drive.google.com/file/d/1YZP9nBSRaQ_Z2cFaFLmLkGilEYsEHb2b/view |
| MelHuBERT-20ms 960h stage 2 | 80 | https://drive.google.com/file/d/1Fn_C5VoH5iV3LdvBEjvfAsbMPhWFFPdd/view |
| MelHuBERT-10ms 960h stage 2 | 40 | https://drive.google.com/file/d/18u2u-528uDh5T7R1bp1wvWJ2ygcrNlzx/view |

---

### 24.3 Mel spectrogram input: 60-bin → 80-bin zero-padding

The dataset's `torchaudio_mel_spectrogram.parquet` stores 60-bin spectrograms.
MelHuBERT's `pre_extract_proj` expects exactly 80 bins (two stacked 40-bin fbank frames).

**Fix:** zero-pad the frequency axis:

```python
mel_80 = np.zeros((80, T), dtype=np.float32)
mel_80[:60, :] = mel_60           # bins 60–79 stay zero
```

This is done in `load_mel_features()` in `train_binary_classifier.py`. The upper 20 bins
are always zero — they represent frequencies above the dataset's fbank ceiling. The pretrained
`pre_extract_proj` Linear layer was trained on 80-bin input, so it handles the zero padding
gracefully (those weights simply receive zero input).

---

### 24.4 SPARC augmentation: two-mode design

The checkpoint's `pre_extract_proj` maps exactly `80 → 768`. Concatenating EMA position
channels changes the input to `80+S`, breaking `load_state_dict(strict=True)`.

**Two-mode design** (controlled by `sparc_channels` in `TransformerBranchConfig`):

| Mode | `sparc_channels` | Backbone `feat_emb_dim` | `pre_extract_proj` | Load mode |
|---|---|---|---|---|
| Clean mel only | 0 | 80 (matches ckpt) | `Linear(80→768)` loaded from ckpt | `strict=True` |
| SPARC-augmented | >0 | 768 (= `embed_dim`) | `None` (backbone skips it) | `strict=False` |

In SPARC mode, `TransformerBranch.input_proj = Linear(80+S → 768)` is trained from scratch.
On load, `pre_extract_proj.*` (2 tensors) are silently skipped; all 12 transformer encoder
layers load correctly (199/201 tensors).

**Use EMA positions, not jerk time-series.** Jerk is the 3rd derivative of position — each
differentiation amplifies high-frequency noise. At 50 Hz the SNR is too low for frame-level
features. Jerk scalars belong in `sparse_keys` for utterance-level conditioning instead.

---

### 24.5 Static feature conditioning: FiLM (not MHA)

Utterance-level scalar features (HNR, F0, speaking rate, etc.) are injected into the encoder
hidden states via a `SparseConditionedLayer`. We use **FiLM (Feature-wise Linear Modulation)**
rather than `MultiheadAttention`.

**Why not MHA?**

With a single conditioning token (`kv_len=1`), MHA reduces to a weighted projection:

```
softmax(q · k^T / sqrt(d)) · v   where kv_len=1
→ just v weighted by a scalar attention score
→ equivalent to a learned linear combination of v
```

But `nn.MultiheadAttention(embed_dim=768, num_heads=12)` costs `4 × 768² ≈ 2.36M` parameters
regardless of how many tokens you attend over. With 91–200 training clips, this causes
immediate memorisation.

**FiLM uses `~10K` parameters for the same representational power:**

```
scale = Linear(sparse_dim → 768)    # ~5K params
shift = Linear(sparse_dim → 768)    # ~5K params
out   = LayerNorm(hidden × (1 + scale) + shift)
```

The `(1 + scale)` formulation initialises close to identity (weights zeroed at init) so the
residual stream is undisturbed at the start of training. The network learns to selectively
amplify or suppress encoder dimensions based on utterance-level acoustic statistics.

**Empirical result:** switching from MHA (2.37M trainable params) to FiLM (14,593 trainable
params) eliminated NaN loss and raised best val AUC from 0.43 (collapsed) to 0.85.

---

### 24.6 Default sparse conditioning keys

When `--feature-mode melhubert` is used without `--sparse-keys`, these 7 OpenSMILE
eGeMAPSv02 features are used automatically (from `static_features.tsv`):

| Key | What it captures | Clinical relevance |
|---|---|---|
| `HNRdBACF_sma3nz_stddevNorm` | HNR variability | Irregular voicing quality |
| `speaking_rate` | Syllables/voiced segs per sec | Fluency, motor control |
| `jitterLocal_sma3nz_stddevNorm` | F0 perturbation variability | Laryngeal stability |
| `F0semitoneFrom27.5Hz_sma3nz_amean` | Mean fundamental frequency | Pitch (child range) |
| `loudness_sma3_amean` | Mean perceived loudness | Respiratory support |
| `alphaRatioV_sma3nz_amean` | Voiced spectral tilt | Breathiness indicator |
| `VoicedSegmentsPerSec` | Voicing density | Connected speech timing |

These were selected by first-layer weight magnitude from a trained `BinaryMLP` on the same
task. To disable conditioning entirely: `--sparse-keys ''`.

**NaN guard:** `speaking_rate` has 4 missing rows in the dataset. The loading code uses
`pd.isna()` (not `v or 0.0` which passes NaN through since `nan` is truthy in Python).
`SparseConditionedLayer.forward` also applies `torch.nan_to_num` as a last-resort guard.
All values are z-scored on train-set statistics then clamped to `[-5, 5]` before the model.

---

### 24.7 Data splitting strategy

Splits are **participant-stratified**, not clip-stratified. A participant's clips must not
appear in both train and validation — otherwise the model can memorize speaker identity
rather than condition.

```
pos_pids = all participants with label=1  (shuffled)
neg_pids = all participants with label=0  (shuffled)

val_pids  = first val_frac of each       (default 20%)
train_pids = remainder
```

**Illness-at-recording confound:** participants who were acutely ill at the time of recording
have contaminated acoustic features (nasal congestion, changed respiratory patterns). These
are excluded with `--exclude-ill`, which filters on `peds_mc_l2w` in the phenotype data.
Failing to exclude them leaks condition-unrelated illness signal into training.

**Negative class options:**

| `--negative-class` | Who is in the negative class |
|---|---|
| `all` | All participants without the target condition |
| `healthy_only` | Only participants with no clinical flags in any category |

`healthy_only` produces a cleaner binary contrast but reduces N. For rare conditions with
few positives, `all` is preferred (larger negative pool, better-calibrated pos_weight).

---

### 24.8 Class imbalance: `pos_weight`

Binary cross-entropy loss is reweighted by:

```
pos_weight = n_neg_train / n_pos_train
```

This is passed to `F.binary_cross_entropy_with_logits(pos_weight=...)` so the loss gradient
from each positive sample is up-weighted to compensate for imbalance. For `hearing_cochlear`
with `healthy_only`, this is ~1.94 (31 pos, 60 neg in train).

---

### 24.9 Progressive encoder unfreezing

The MelHuBERT encoder has ~85M parameters. With small datasets (<500 clips), unfreezing
all layers catastrophically overrides the adult-speech representations.

**Strategy: freeze-then-partially-thaw**

1. Train with frozen encoder for `--unfreeze-after` epochs (head + FiLM only).
2. At that epoch, unfreeze only the top `--unfreeze-top-n` transformer layers.
3. The unfrozen encoder parameters receive `0.1×` base LR (handled via `get_param_groups()`).
4. The optimizer is **rebuilt** at unfreeze time so the new parameters are properly registered.

**Why rebuild the optimizer?** AdamW maintains per-parameter momentum state. If you simply
`.requires_grad_(True)` on frozen params, they exist in the model graph but not in the
optimizer's param groups — their gradients accumulate but no update is applied. Rebuilding
the optimizer at unfreeze time ensures a clean start for the thawed parameters.

**Recommended settings by dataset size:**

| Clips | `--unfreeze-top-n` | `--lr` (encoder) | Notes |
|---|---|---|---|
| < 200 | 2 | 1e-5 | Only last 2 layers + LayerNorm |
| 200–500 | 3 | 1e-5 | Safe for cluster runs |
| 500–2000 | 6 | 5e-6 | Half the network |
| > 2000 | `None` (all) | 1e-6 | Full fine-tune, very low LR |

The encoder LR is always `args.lr × 0.1`. So `--lr 1e-4` gives encoder params `1e-5`.

---

### 24.10 `load_pretrained` device safety

`load_pretrained()` calls `MelHuBERTModel(model_cfg)` internally, which always allocates on
CPU. If the outer model was already moved to MPS or CUDA via `.to(device)`, the replacement
backbone would be on CPU while the rest of the model is on the accelerator.

**Fix:** capture the device from the existing backbone before replacing it:

```python
target_device = next(self.backbone.parameters()).device
new_backbone = MelHuBERTModel(model_cfg)
# ... load state dict ...
self.backbone = new_backbone.to(target_device)
```

This must be done even when the model has not been moved yet (target_device = cpu in that
case, which is safe).

---

### 24.11 `TransformerBranchConfig` quick reference

| Field | Default | Description |
|---|---|---|
| `ckpt_path` | `None` | Path to `.ckpt`; `None` = from-scratch |
| `freeze_encoder` | `True` | Freeze backbone at init |
| `input_mel_bins` | `80` | Bins in input mel tensor |
| `melhubert_feat_dim` | `80` | Must match ckpt `feat_emb_dim` (80 for 20ms) |
| `use_frame_stacking` | `False` | Stack pairs of 40-bin frames → 80 dim |
| `sparc_channels` | `0` | EMA position channels to concat onto mel |
| `sparse_keys` | `[]` | Scalar feature keys for FiLM conditioning |
| `sparse_dim` | `64` | Dimension of FiLM conditioning vector |
| `embed_dim` | `768` | Encoder hidden dimension (768 = Base) |
| `n_encoder_layers` | `2` | Used only for from-scratch mode |
| `pool` | `"mean"` | `"mean"` or `"weighted_sum"` |

---

### 24.12 Training CLI reference

Script: `scripts/train_binary_classifier.py`

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--target` | required | Condition name, e.g. `hearing_cochlear` |
| `--feature-mode` | `mlp` | `mlp`, `tcn`, or `melhubert` |
| `--negative-class` | `all` | `all` or `healthy_only` |
| `--exclude-ill` | off | Exclude participants ill at recording |
| `--tasks` | all | Filter to specific task names, e.g. `long-sounds` |
| `--ckpt-path` | `weights/melhubert-20ms-stg2.ckpt` | Checkpoint path |
| `--sparse-keys` | 7 defaults | Keys for FiLM conditioning; `''` to disable |
| `--unfreeze-after` | `None` | Epoch to begin partial unfreezing |
| `--unfreeze-top-n` | `None` | Number of top encoder layers to unfreeze |
| `--patience` | `None` | Early stopping patience (epochs without val AUC gain) |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`, `mps` |
| `--lr` | `1e-3` | Base LR (encoder gets `0.1×` after unfreeze) |
| `--batch-size` | `16` | Batch size |
| `--epochs` | `10` | Max epochs |

**Recommended cluster command (hearing_cochlear):**

```bash
python scripts/train_binary_classifier.py \
    --target hearing_cochlear \
    --negative-class healthy_only \
    --feature-mode melhubert \
    --ckpt-path weights/melhubert-20ms-stg2.ckpt \
    --exclude-ill \
    --tasks long-sounds \
    --epochs 40 \
    --batch-size 16 \
    --lr 1e-4 \
    --patience 8 \
    --unfreeze-after 5 \
    --unfreeze-top-n 3 \
    --log-every 5 \
    --out-dir runs/cochlear_melhubert \
    --device cuda
```

---

### 24.13 Known pitfalls (all addressed)

1. **PyTorch ≥ 2.6 `weights_only` default** — checkpoint stores `argparse.Namespace`;
   fixed with `torch.serialization.add_safe_globals([argparse.Namespace])` + `weights_only=False`.

2. **`pre_extract_proj` shape mismatch** — two-mode design (§24.4).

3. **`n_encoder_layers` override** — `load_pretrained()` always reconstructs 12 layers from
   the checkpoint's own stored config; `cfg.n_encoder_layers` is ignored.

4. **EMA frame rate** — `MelSPARCAugmentation` uses `F.interpolate(mode="nearest")` to
   resample sparc to match stacked mel length automatically.

5. **Backbone on CPU after `.to(device)`** — `load_pretrained()` detects current device
   and moves the rebuilt backbone there (§24.10).

6. **`speaking_rate` NaN rows** — 4 rows in `static_features.tsv` have missing speaking_rate.
   `pd.isna()` guard in `_make_mel_rows()`; `torch.nan_to_num` in `SparseConditionedLayer`.

7. **HNR outliers** — `HNRdBACF_sma3nz_stddevNorm` ranges from −1812 to +624 raw.
   z-scored and clamped to `[-5, 5]` before the model.

8. **MHA → FiLM switch** — `SparseConditionedLayer` originally used `nn.MultiheadAttention`
   (2.36M params). With 91 clips this caused immediate memorisation and NaN-level gradient
   spikes (GNorm ~35). Replaced with FiLM (§24.5); 14,593 total trainable params.

9. **`final_proj` unused** — in checkpoint and backbone but never called when `no_pred=True`.
   Grad is `None`; smoke tests check encoder layer params specifically.

10. **1.0 GB checkpoint** — added `weights/*.ckpt` to `.gitignore`; stored in `weights/` subdir.
