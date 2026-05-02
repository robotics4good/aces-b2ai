"""Complete registry of every feature family in the Bridge2AI pediatric dataset.

Each ``FeatureEntry`` maps directly to a field described in the dataset's
``features/*.json`` schema files (read verbatim during the codebase audit).
Time-series entries list the aggregation statistics that will be computed when
building the scalar matrix for matrix factorisation.

Scalar groups (used directly, no aggregation needed):
    - STATIC_SCALARS  — 135-col openSMILE-derived ``static_features.tsv``

Time-series groups (aggregated → scalars on the fly):
    - SPARC_LOUDNESS  — 1×T @ 50 Hz
    - SPARC_PERIODICITY — 1×T @ 50 Hz
    - SPARC_PITCH     — 1×T @ 50 Hz (CREPE-based, 50–550 Hz)
    - SPARC_EMA       — 12×T @ 50 Hz (TD, TB, TT, LI, UL, LL × X/Y)
    - PPG             — 40×T @ 100 Hz (phoneme posteriors)
    - MFCC            — 60×T @ 50 Hz
    - MEL             — 60×T @ 50 Hz
    - SPECTROGRAM     — 201×T @ 50 Hz (log-magnitude dB)
    - TORCHAUDIO_PITCH — 1×T (Hz, variable rate)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureEntry:
    """Describes one feature family in the dataset.

    Attributes
    ----------
    name:
        Unique identifier used as prefix when aggregation produces multiple
        scalar columns (e.g. ``"ema"`` → ``"ema_ch0_mean"``, etc.).
    family:
        Broad group label used for filtering and colouring in analysis plots.
    parquet_file:
        Filename (without path) of the Parquet file that contains this feature,
        relative to the dataset ``features/`` directory.  ``None`` for the TSV.
    column:
        Column name inside the Parquet / TSV that holds the tensor or scalar.
    n_channels:
        Fixed channel count for 2-D tensors; 1 for 1-D time series; 0 for
        pre-aggregated scalars.
    frame_rate_hz:
        Temporal sampling rate in Hz.  ``None`` when not applicable or unknown.
    feature_type:
        ``"scalar"``           — already a number per row (static_features.tsv).
        ``"timeseries_1d"``    — 1-D float array per row (T,).
        ``"timeseries_2d"``    — 2-D float array per row (C, T) or (T, C).
        ``"timeseries_2d_tc"`` — 2-D array stored time-first (T, C); transposed
                                  before aggregation.
    channel_labels:
        Optional human-readable labels for each channel (e.g. EMA articulator
        names).  Empty list when not meaningful (MFCC indices, etc.).
    agg_stats:
        Statistics computed over the time axis when converting a time-series
        feature to scalars.  The default set covers the mean, standard
        deviation, and 10th/50th/90th percentiles, giving 5 scalars per
        channel.
    is_nonneg:
        True when the values are guaranteed ≥ 0 (loudness, periodicity,
        PPG posterior probabilities, mel power, etc.).  Enables NMF.
    description:
        Taken verbatim (or paraphrased) from the dataset JSON ``description``
        field.
    """

    name: str
    family: str
    parquet_file: str | None
    column: str
    n_channels: int
    frame_rate_hz: float | None
    feature_type: str
    channel_labels: list[str] = field(default_factory=list)
    agg_stats: list[str] = field(default_factory=lambda: ["mean", "std", "p10", "p50", "p90"])
    is_nonneg: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# EMA channel labels (from sparc_ema.json articulator_channel axis)
# ---------------------------------------------------------------------------
_EMA_CHANNELS = [
    "td_x", "td_y",   # tongue dorsum
    "tb_x", "tb_y",   # tongue body
    "tt_x", "tt_y",   # tongue tip
    "li_x", "li_y",   # lower incisor
    "ul_x", "ul_y",   # upper lip
    "ll_x", "ll_y",   # lower lip
]

# ---------------------------------------------------------------------------
# PPG phoneme labels (from ppgs.json, 40 classes + silence)
# ---------------------------------------------------------------------------
_PPG_PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B",  "CH", "D",  "DH", "EH", "ER",
    "EY", "F",  "G",  "HH", "IH", "IY",
    "JH", "K",  "L",  "M",  "N",  "NG",
    "OW", "OY", "P",  "R",  "S",  "SH",
    "T",  "TH", "UH", "UW", "V",  "W",
    "Y",  "Z",  "ZH", "SIL",
]

# ---------------------------------------------------------------------------
# Static scalar feature groups (from static_features.json)
# Each sub-list groups columns that appear in static_features.tsv.
# ---------------------------------------------------------------------------

STATIC_F0_COLS = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
]

STATIC_LOUDNESS_COLS = [
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
    "loudness_sma3_percentile20.0",
    "loudness_sma3_percentile50.0",
    "loudness_sma3_percentile80.0",
    "loudness_sma3_pctlrange0-2",
    "loudness_sma3_meanRisingSlope",
    "loudness_sma3_stddevRisingSlope",
]

STATIC_SPECTRAL_FLUX_COLS = [
    "spectralFlux_sma3_amean",
    "spectralFlux_sma3_stddevNorm",
]

STATIC_MFCC_COLS = [
    f"mfcc{i}_sma3_{stat}"
    for i in range(1, 5)
    for stat in ("amean", "stddevNorm")
]

STATIC_VOICE_QUALITY_COLS = [
    "jitterLocal_sma3nz_amean",
    "jitterLocal_sma3nz_stddevNorm",
    "shimmerLocaldB_sma3nz_amean",
    "shimmerLocaldB_sma3nz_stddevNorm",
    "HNRdBACF_sma3nz_amean",
    "HNRdBACF_sma3nz_stddevNorm",
    "logRelF0-H1-H2_sma3nz_amean",
    "logRelF0-H1-H2_sma3nz_stddevNorm",
    "logRelF0-H1-A3_sma3nz_amean",
    "logRelF0-H1-A3_sma3nz_stddevNorm",
]

STATIC_FORMANT_COLS = [
    f"{formant}_{stat}"
    for formant in (
        "F1frequency_sma3nz", "F1bandwidth_sma3nz", "F1amplitudeLogRelF0_sma3nz",
        "F2frequency_sma3nz", "F2bandwidth_sma3nz", "F2amplitudeLogRelF0_sma3nz",
        "F3frequency_sma3nz", "F3bandwidth_sma3nz", "F3amplitudeLogRelF0_sma3nz",
    )
    for stat in ("amean", "stddevNorm")
]

STATIC_VOICED_SPECTRAL_COLS = [
    "alphaRatioV_sma3nz_amean", "alphaRatioV_sma3nz_stddevNorm",
    "hammarbergIndexV_sma3nz_amean", "hammarbergIndexV_sma3nz_stddevNorm",
    "slopeV0-500_sma3nz_amean", "slopeV0-500_sma3nz_stddevNorm",
    "slopeV500-1500_sma3nz_amean", "slopeV500-1500_sma3nz_stddevNorm",
    "spectralFluxV_sma3nz_amean", "spectralFluxV_sma3nz_stddevNorm",
    "mfcc1V_sma3nz_amean", "mfcc1V_sma3nz_stddevNorm",
    "mfcc2V_sma3nz_amean", "mfcc2V_sma3nz_stddevNorm",
    "mfcc3V_sma3nz_amean", "mfcc3V_sma3nz_stddevNorm",
    "mfcc4V_sma3nz_amean", "mfcc4V_sma3nz_stddevNorm",
]

STATIC_UNVOICED_SPECTRAL_COLS = [
    "alphaRatioUV_sma3nz_amean",
    "hammarbergIndexUV_sma3nz_amean",
    "slopeUV0-500_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
    "spectralFluxUV_sma3nz_amean",
]

STATIC_RHYTHM_COLS = [
    "loudnessPeaksPerSec",
    "VoicedSegmentsPerSec",
    "MeanVoicedSegmentLengthSec",
    "StddevVoicedSegmentLengthSec",
    "MeanUnvoicedSegmentLength",
    "StddevUnvoicedSegmentLength",
]

STATIC_TIMING_COLS = [
    "equivalentSoundLevel_dBp",
    "duration",
    "speaking_rate",
    "articulation_rate",
    "phonation_ratio",
    "pause_rate",
    "mean_pause_duration",
]

STATIC_EXTENDED_PROSODY_COLS = [
    "mean_f0_hertz", "std_f0_hertz",
    "mean_intensity_db", "std_intensity_db", "range_ratio_intensity_db",
    "mean_hnr_db", "std_hnr_db",
    "spectral_slope", "spectral_tilt",
    "cepstral_peak_prominence_mean", "cepstral_peak_prominence_std",
    "mean_f1_frequency_loc", "std_f1_frequency_loc",
    "mean_f1_bandwidth_loc", "std_f1_bandwidth_loc",
    "mean_f2_frequency_loc", "std_f2_frequency_loc",
    "mean_f2_bandwidth_loc", "std_f2_bandwidth_loc",
    "spectral_gravity", "spectral_std_dev", "spectral_skewness", "spectral_kurtosis",
    "local_jitter", "rap_jitter", "ppq5_jitter", "ddp_jitter", "mean_jitter",
    "local_shimmer", "localDB_shimmer", "apq3_shimmer", "apq5_shimmer",
    "apq11_shimmer", "mean_shimmer",
]

STATIC_QUALITY_COLS = ["stoi", "pesq", "si_sdr"]

# Ordered collection of ALL static column groups for iteration.
STATIC_FEATURE_GROUPS: dict[str, list[str]] = {
    "f0": STATIC_F0_COLS,
    "loudness": STATIC_LOUDNESS_COLS,
    "spectral_flux": STATIC_SPECTRAL_FLUX_COLS,
    "mfcc_global": STATIC_MFCC_COLS,
    "voice_quality": STATIC_VOICE_QUALITY_COLS,
    "formants": STATIC_FORMANT_COLS,
    "voiced_spectral": STATIC_VOICED_SPECTRAL_COLS,
    "unvoiced_spectral": STATIC_UNVOICED_SPECTRAL_COLS,
    "rhythm": STATIC_RHYTHM_COLS,
    "timing": STATIC_TIMING_COLS,
    "extended_prosody": STATIC_EXTENDED_PROSODY_COLS,
    "quality_metrics": STATIC_QUALITY_COLS,
}

# Flat list of all 135 static scalar column names (non-text columns only).
ALL_STATIC_COLUMNS: list[str] = [
    col for cols in STATIC_FEATURE_GROUPS.values() for col in cols
]

# ---------------------------------------------------------------------------
# Time-series feature entries (one entry per parquet feature family)
# ---------------------------------------------------------------------------

TIMESERIES_ENTRIES: list[FeatureEntry] = [
    FeatureEntry(
        name="sparc_loudness",
        family="sparc",
        parquet_file="sparc_loudness.parquet",
        column="loudness",
        n_channels=1,
        frame_rate_hz=50.0,
        feature_type="timeseries_1d",
        is_nonneg=True,
        description=(
            "Estimated loudness (average absolute amplitude of z-scored waveform), "
            "50 Hz, z-scored per utterance."
        ),
    ),
    FeatureEntry(
        name="sparc_periodicity",
        family="sparc",
        parquet_file="sparc_periodicity.parquet",
        column="periodicity",
        n_channels=1,
        frame_rate_hz=50.0,
        feature_type="timeseries_1d",
        is_nonneg=True,
        description=(
            "CREPE-derived voicing confidence in [0, 1]. "
            "50 Hz, fmin=50 Hz, fmax=550 Hz, hop=160 samples."
        ),
    ),
    FeatureEntry(
        name="sparc_pitch",
        family="sparc",
        parquet_file="sparc_pitch.parquet",
        column="pitch",
        n_channels=1,
        frame_rate_hz=50.0,
        feature_type="timeseries_1d",
        is_nonneg=True,
        description="F0 in Hz (50–550 Hz), 0/NaN where unvoiced. CREPE via SPARC.",
    ),
    FeatureEntry(
        name="ema",
        family="sparc",
        parquet_file="sparc_ema.parquet",
        column="ema",
        n_channels=12,
        frame_rate_hz=50.0,
        # Schema: (T, 12) — time-major; transposed before aggregation.
        feature_type="timeseries_2d_tc",
        channel_labels=_EMA_CHANNELS,
        description=(
            "Articulatory kinematics from SPARC inversion model. "
            "12 channels: TD, TB, TT, LI, UL, LL × X/Y. 50 Hz, z-scored."
        ),
    ),
    FeatureEntry(
        name="ppg",
        family="ppg",
        parquet_file="ppgs.parquet",
        column="ppgs",   # actual column name in the parquet is "ppgs" not "ppg"
        n_channels=40,
        frame_rate_hz=100.0,
        feature_type="timeseries_2d",
        channel_labels=_PPG_PHONEMES,
        is_nonneg=True,
        description=(
            "Phonetic posteriorgram: 40 phoneme class probabilities ∈ [0, 1], "
            "normalised per frame. 100 Hz."
        ),
    ),
    FeatureEntry(
        name="mfcc",
        family="torchaudio",
        parquet_file="torchaudio_mfcc.parquet",
        column="mfcc",
        n_channels=60,
        frame_rate_hz=50.0,
        feature_type="timeseries_2d",
        description="60 MFCC coefficients. Torchaudio 2.8.0, ::2 time subsampled → 50 Hz.",
    ),
    FeatureEntry(
        name="mel",
        family="torchaudio",
        parquet_file="torchaudio_mel_spectrogram.parquet",
        column="mel_spectrogram",
        n_channels=60,
        frame_rate_hz=50.0,
        feature_type="timeseries_2d",
        is_nonneg=True,
        description="60-bin mel spectrogram (power). Torchaudio 2.8.0, ::2 → 50 Hz.",
    ),
    FeatureEntry(
        name="spectrogram",
        family="torchaudio",
        parquet_file="torchaudio_spectrogram.parquet",
        column="spectrograms",
        n_channels=201,
        frame_rate_hz=50.0,
        feature_type="timeseries_2d",
        is_nonneg=True,
        description="201-bin log-magnitude spectrogram (dB). Torchaudio 2.8.0, ::2 → 50 Hz.",
    ),
    FeatureEntry(
        name="torchaudio_pitch",
        family="torchaudio",
        parquet_file="torchaudio_pitch.parquet",
        column="pitch",
        n_channels=1,
        frame_rate_hz=None,
        feature_type="timeseries_1d",
        is_nonneg=True,
        description=(
            "F0 in Hz (80–500 Hz) via torchaudio detect_pitch_frequency. "
            "Frame rate not specified in schema."
        ),
    ),
]

# Lookup by name
TIMESERIES_BY_NAME: dict[str, FeatureEntry] = {e.name: e for e in TIMESERIES_ENTRIES}


def timeseries_agg_columns(entry: FeatureEntry) -> list[str]:
    """Return the flat scalar column names produced when aggregating ``entry``.

    Example: ``ema`` with default stats → ``["ema_td_x_mean", "ema_td_x_std", ...]``
    """
    if entry.feature_type == "timeseries_1d":
        return [f"{entry.name}_{stat}" for stat in entry.agg_stats]

    channels = (
        entry.channel_labels
        if entry.channel_labels
        else [f"ch{i:03d}" for i in range(entry.n_channels)]
    )
    return [
        f"{entry.name}_{ch}_{stat}"
        for ch in channels
        for stat in entry.agg_stats
    ]
