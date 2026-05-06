from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pandas as pd
import pyworld
import soundfile as sf


@dataclass(frozen=True)
class SourceFilterConfig:
    # Dataset paths (relative to repo root)
    dataset_root: Path = Path("b2ai_adult_dataset/3.0.0")
    mel_parquet: Path = Path("features/torchaudio_mel_spectrogram.parquet")
    spec_parquet: Path = Path("features/torchaudio_spectrogram.parquet")
    pitch_parquet: Path = Path("features/sparc_pitch.parquet")
    periodicity_parquet: Path = Path("features/sparc_periodicity.parquet")
    static_tsv: Path = Path("features/static_features.tsv")

    # Signal config (matches JSON metadata shipped with dataset)
    sample_rate: int = 16000
    n_fft: int = 400
    win_length: int = 400
    hop_length_effective: int = 320  # time-subsampled by 2 in parquet
    n_mels: int = 60

    # Griffin–Lim
    griffin_lim_iters: int = 60

    # WORLD
    world_frame_period_ms: float = 20.0  # matches 50 Hz pitch/periodicity tracks

    # UVFP perturbation scaling (heuristics; tune later)
    jitter_scale: float = 1.0
    shimmer_scale: float = 1.0

    # Convert HNR (dB) → aperiodicity in [0,1]
    hnr_db_low: float = 0.0
    hnr_db_high: float = 20.0


Keyed = tuple[str, str, str]  # participant_id, session_id, task_name


def _ensure_2d_time_major(x: np.ndarray) -> np.ndarray:
    """
    Stored tensors are (freq, time) or (mel, time).
    We keep them as (freq, time) for spectral ops.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    return x


def _hnr_to_aperiodicity(hnr_db: float, *, low: float, high: float) -> float:
    # low HNR => noisy => high aperiodicity
    # clamp to [0,1]
    if np.isnan(hnr_db):
        return 0.5
    x = (hnr_db - low) / (high - low) if high != low else 0.5
    x = float(np.clip(x, 0.0, 1.0))
    return 1.0 - x


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_static_features(
    cfg: SourceFilterConfig, *, participant_id: str, session_id: str, task_name: str
) -> dict[str, float]:
    path = cfg.dataset_root / cfg.static_tsv
    df = pd.read_csv(path, sep="\t", dtype=str)
    hit = df[
        (df["participant_id"] == participant_id)
        & (df["session_id"] == session_id)
        & (df["task_name"] == task_name)
    ]
    if len(hit) != 1:
        raise ValueError(f"Expected 1 row in {path} for key, got {len(hit)}")
    row = hit.iloc[0].to_dict()
    return {
        "jitterLocal_sma3nz_amean": _as_float(row.get("jitterLocal_sma3nz_amean")),
        "shimmerLocaldB_sma3nz_amean": _as_float(row.get("shimmerLocaldB_sma3nz_amean")),
        "HNRdBACF_sma3nz_amean": _as_float(row.get("HNRdBACF_sma3nz_amean")),
    }


def _read_nested_tensor_cell(value) -> np.ndarray:
    # parquet_nested_list comes in as list[list[float]] (already deserialized by pandas/pyarrow)
    arr = np.asarray(value, dtype=np.float32)
    return arr


def load_filter_representation(
    cfg: SourceFilterConfig,
    *,
    participant_id: str,
    session_id: str,
    task_name: str,
    mode: Literal["mel", "linear"] = "mel",
) -> np.ndarray:
    if mode == "mel":
        path = cfg.dataset_root / cfg.mel_parquet
        df = pd.read_parquet(path, columns=["participant_id", "session_id", "task_name", "mel_spectrogram"])
        hit = df[
            (df["participant_id"] == participant_id)
            & (df["session_id"] == session_id)
            & (df["task_name"] == task_name)
        ]
        if len(hit) != 1:
            raise ValueError(f"Expected 1 row in {path} for key, got {len(hit)}")
        mel = _read_nested_tensor_cell(hit.iloc[0]["mel_spectrogram"])
        return _ensure_2d_time_major(mel)

    if mode == "linear":
        path = cfg.dataset_root / cfg.spec_parquet
        df = pd.read_parquet(path, columns=["participant_id", "session_id", "task_name", "spectrograms"])
        hit = df[
            (df["participant_id"] == participant_id)
            & (df["session_id"] == session_id)
            & (df["task_name"] == task_name)
        ]
        if len(hit) != 1:
            raise ValueError(f"Expected 1 row in {path} for key, got {len(hit)}")
        spec_db = _read_nested_tensor_cell(hit.iloc[0]["spectrograms"])
        return _ensure_2d_time_major(spec_db)

    raise ValueError(f"Unknown mode={mode}")


def load_pitch_track(
    cfg: SourceFilterConfig, *, participant_id: str, session_id: str, task_name: str
) -> np.ndarray:
    path = cfg.dataset_root / cfg.pitch_parquet
    df = pd.read_parquet(path, columns=["participant_id", "session_id", "task_name", "pitch"])
    hit = df[
        (df["participant_id"] == participant_id)
        & (df["session_id"] == session_id)
        & (df["task_name"] == task_name)
    ]
    if len(hit) != 1:
        raise ValueError(f"Expected 1 row in {path} for key, got {len(hit)}")
    f0 = np.asarray(hit.iloc[0]["pitch"], dtype=np.float32)
    # WORLD expects unvoiced as 0.0
    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
    return f0


def load_periodicity_track(
    cfg: SourceFilterConfig, *, participant_id: str, session_id: str, task_name: str
) -> np.ndarray | None:
    path = cfg.dataset_root / cfg.periodicity_parquet
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["participant_id", "session_id", "task_name", "periodicity"])
    hit = df[
        (df["participant_id"] == participant_id)
        & (df["session_id"] == session_id)
        & (df["task_name"] == task_name)
    ]
    if len(hit) != 1:
        return None
    p = np.asarray(hit.iloc[0]["periodicity"], dtype=np.float32)
    p = np.clip(np.where(np.isfinite(p), p, 0.0), 0.0, 1.0)
    return p


def synthesize_excitation_world(
    cfg: SourceFilterConfig,
    *,
    f0_hz: np.ndarray,
    hnr_db: float,
    jitter_local: float,
    shimmer_db: float,
    periodicity: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Creates a WORLD-style excitation proxy:
    - Use f0 track (with UVFP jitter applied)
    - Use global aperiodicity derived from HNR
    - Use flat spectral envelope so we don't bake in vocal-tract filtering
    """
    rng = np.random.default_rng(seed)

    f0 = np.asarray(f0_hz, dtype=np.float64)
    f0 = np.where(np.isfinite(f0), f0, 0.0)
    f0 = np.clip(f0, 0.0, 550.0)

    # Jitter: apply proportional random perturbation on voiced frames.
    # jitterLocal is typically a small unitless ratio (or percent); we treat it as a fraction.
    jitter = float(jitter_local) if np.isfinite(jitter_local) else 0.0
    voiced = f0 > 0
    if np.any(voiced) and jitter > 0:
        eps = rng.normal(loc=0.0, scale=jitter * cfg.jitter_scale, size=f0.shape)
        f0 = np.where(voiced, np.maximum(0.0, f0 * (1.0 + eps)), 0.0)

    # Duration based on frame rate
    frame_period = cfg.world_frame_period_ms  # ms
    n_frames = len(f0)
    n_samples = int(round(n_frames * (frame_period / 1000.0) * cfg.sample_rate))

    # Flat spectral envelope (sp) expected shape: (n_frames, fft_size/2+1)
    fft_size = pyworld.get_cheaptrick_fft_size(cfg.sample_rate)
    n_bins = fft_size // 2 + 1
    sp = np.ones((n_frames, n_bins), dtype=np.float64)

    # Aperiodicity (ap) expected shape: (n_frames, n_bins) in [0,1]
    ap_level = _hnr_to_aperiodicity(hnr_db, low=cfg.hnr_db_low, high=cfg.hnr_db_high)
    ap = np.full((n_frames, n_bins), ap_level, dtype=np.float64)
    if periodicity is not None and len(periodicity) == n_frames:
        # Higher periodicity => lower aperiodicity
        ap = np.clip(ap * (1.0 - periodicity[:, None]), 0.0, 1.0)

    y = pyworld.synthesize(
        f0.astype(np.float64),
        sp,
        ap,
        cfg.sample_rate,
        frame_period,
    ).astype(np.float32)

    # Shimmer: amplitude modulation. shimmerLocaldB is in dB; treat as stddev in dB.
    shimmer = float(shimmer_db) if np.isfinite(shimmer_db) else 0.0
    if shimmer > 0 and len(y) > 0:
        # Create a per-frame gain (in dB), then upsample to samples.
        g_db = rng.normal(loc=0.0, scale=shimmer * cfg.shimmer_scale, size=n_frames)
        g_lin = (10.0 ** (g_db / 20.0)).astype(np.float32)
        # Upsample via repeat to sample resolution (simple, stable).
        reps = int(math.ceil(len(y) / n_frames))
        g = np.repeat(g_lin, reps)[: len(y)]
        y = y * g

    # Trim/pad exactly to computed length (WORLD can differ by a few samples)
    if len(y) > n_samples:
        y = y[:n_samples]
    elif len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))

    # Normalize to safe range
    peak = float(np.max(np.abs(y))) if len(y) else 1.0
    if peak > 0:
        y = 0.8 * (y / peak)
    return y


def reconstruct_filter_waveform_from_mel(cfg: SourceFilterConfig, mel: np.ndarray) -> np.ndarray:
    """
    Uses librosa's mel inversion + Griffin–Lim to get a waveform whose envelope matches mel.
    """
    mel = _ensure_2d_time_major(mel)
    y = librosa.feature.inverse.mel_to_audio(
        M=mel,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length_effective,
        win_length=cfg.win_length,
        n_iter=cfg.griffin_lim_iters,
        power=2.0,
    ).astype(np.float32)
    return y


def combine_source_with_filter_magnitude(
    cfg: SourceFilterConfig,
    *,
    excitation: np.ndarray,
    target_linear_spec_db: np.ndarray,
) -> np.ndarray:
    """
    Frame-wise: keep excitation phase; replace magnitude with target linear spectrogram magnitude.
    Dataset spectrogram is in dB (per JSON), so convert to amplitude.
    """
    spec_db = _ensure_2d_time_major(target_linear_spec_db)
    target_mag = librosa.db_to_amplitude(spec_db, ref=1.0).astype(np.float32)

    stft = librosa.stft(
        excitation,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length_effective,
        win_length=cfg.win_length,
        center=True,
    )
    phase = np.exp(1j * np.angle(stft))

    # Align time frames
    T = min(stft.shape[1], target_mag.shape[1])
    F = min(stft.shape[0], target_mag.shape[0])
    y = librosa.istft(
        target_mag[:F, :T] * phase[:F, :T],
        hop_length=cfg.hop_length_effective,
        win_length=cfg.win_length,
        length=len(excitation),
    ).astype(np.float32)

    peak = float(np.max(np.abs(y))) if len(y) else 1.0
    if peak > 0:
        y = 0.8 * (y / peak)
    return y


def synthesize_subject(
    cfg: SourceFilterConfig,
    *,
    participant_id: str,
    session_id: str,
    task_name: str,
    output_wav: Path,
    combine_mode: Literal["mel_only", "phase_from_excitation_linear_mag"] = "mel_only",
    seed: int = 0,
) -> Path:
    static = load_static_features(
        cfg, participant_id=participant_id, session_id=session_id, task_name=task_name
    )
    f0 = load_pitch_track(cfg, participant_id=participant_id, session_id=session_id, task_name=task_name)
    periodicity = load_periodicity_track(
        cfg, participant_id=participant_id, session_id=session_id, task_name=task_name
    )

    excitation = synthesize_excitation_world(
        cfg,
        f0_hz=f0,
        hnr_db=static["HNRdBACF_sma3nz_amean"],
        jitter_local=static["jitterLocal_sma3nz_amean"],
        shimmer_db=static["shimmerLocaldB_sma3nz_amean"],
        periodicity=periodicity,
        seed=seed,
    )

    if combine_mode == "mel_only":
        mel = load_filter_representation(
            cfg,
            participant_id=participant_id,
            session_id=session_id,
            task_name=task_name,
            mode="mel",
        )
        y = reconstruct_filter_waveform_from_mel(cfg, mel)
    else:
        spec_db = load_filter_representation(
            cfg,
            participant_id=participant_id,
            session_id=session_id,
            task_name=task_name,
            mode="linear",
        )
        y = combine_source_with_filter_magnitude(cfg, excitation=excitation, target_linear_spec_db=spec_db)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, y, cfg.sample_rate)
    return output_wav

