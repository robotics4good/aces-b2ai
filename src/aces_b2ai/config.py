from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TorchaudioFeatureConfig:
    """Mirrors dataset `features/torchaudio_*.json` (16 kHz pipeline)."""

    sample_rate_hz: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    subsample_time_factor: int = 2  # stored tensors use [::2] on time for 2D
    mel_bins: int = 60
    mfcc_coeffs: int = 60
    spec_freq_bins: int = 201
    pitch_band_hz: tuple[float, float] = (80.0, 500.0)


@dataclass(frozen=True)
class VoicingConfig:
    periodicity_threshold: float = 0.5
    f0_min_hz: float = 80.0
    f0_max_hz: float = 500.0


@dataclass(frozen=True)
class ModulationBandsHz:
    dc_to_1: tuple[float, float] = (0.0, 1.0)
    hz_1_to_4: tuple[float, float] = (1.0, 4.0)
    hz_4_to_8: tuple[float, float] = (4.0, 8.0)
    above_8: float = 8.0


@dataclass(frozen=True)
class ArticulatorySmoothnessConfig:
    """Config for the articulatory smoothness (normalized jerk) extractor."""

    fs: float = 50.0
    # Savitzky-Golay parameters — window must be odd and > polyorder
    sg_window_frames: int = 11  # 220 ms at 50 Hz
    sg_polyorder: int = 3
    # Active-frame masking: "velocity" uses SG deriv=1 threshold (self-contained),
    # "loudness" requires loudness_sparc on context, "none" skips masking
    active_mask_source: str = "velocity"
    # Threshold in z-scored normalized units/s (EMA is z-scored, not raw mm)
    velocity_threshold: float = 0.5
    min_active_frames: int = 20
    # Strip window_length//2 frames from each end before integrating (edge artifact guard)
    trim_edge_frames: bool = True
    # Channel indices from sparc_ema.json — TDX=0,TDY=1,TBX=2,TBY=3,...,ULY=9,LLX=10,LLY=11
    idx_tbx: int = 2
    idx_tby: int = 3
    idx_uly: int = 9
    idx_lly: int = 11


@dataclass(frozen=True)
class PipelineConfig:
    torchaudio: TorchaudioFeatureConfig = TorchaudioFeatureConfig()
    voicing: VoicingConfig = VoicingConfig()
    modulation_bands: ModulationBandsHz = ModulationBandsHz()
    """Effective frame rate (Hz) for Torchaudio 2D tensors after ::2."""
    torchaudio_2d_frame_rate_hz: float = 50.0
    enable_dynamical: bool = False
    """Minimum voiced frames required for phase-portrait / sampen extractors."""
    dynamical_min_voiced_frames: int = 20
    enable_articulatory: bool = False
    articulatory: ArticulatorySmoothnessConfig = ArticulatorySmoothnessConfig()
