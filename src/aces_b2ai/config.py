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
class PipelineConfig:
    torchaudio: TorchaudioFeatureConfig = TorchaudioFeatureConfig()
    voicing: VoicingConfig = VoicingConfig()
    modulation_bands: ModulationBandsHz = ModulationBandsHz()
    """Effective frame rate (Hz) for Torchaudio 2D tensors after ::2."""
    torchaudio_2d_frame_rate_hz: float = 50.0
    enable_dynamical: bool = False
    """Minimum voiced frames required for phase-portrait / sampen extractors."""
    dynamical_min_voiced_frames: int = 20
