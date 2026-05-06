from __future__ import annotations

import argparse
import math
import warnings
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
    """
    Configuration aligned to the dataset JSON metadata.

    Confirmed from `b2ai_adult_dataset/3.0.0/features/torchaudio_mel_spectrogram.json`:
    - Parquet mel values are **float32**, **linear power scale** (torchaudio `MelSpectrogram` default),
      with **no log/dB compression**.
    - Parquet time axis is **subsampled by 2** for privacy:
      original hop was 160 samples (10 ms @ 16 kHz), stored hop is 320 samples (20 ms @ 16 kHz).
    - Therefore `librosa.feature.inverse.mel_to_audio(..., power=2.0)` is correct for inversion.
    """

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
    # hop_length_effective=320 reflects 2x temporal subsampling in parquet for privacy:
    # original hop was 160 (10 ms), stored hop is 320 (20 ms).
    hop_length_effective: int = 320
    n_mels: int = 60

    # Griffin–Lim
    griffin_lim_iters: int = 60

    # Mel scale handling
    # The dataset *should* store linear-power mel (torchaudio MelSpectrogram default),
    # but if it were log/dB-compressed, inversion would be silently wrong.
    mel_scale_autoconvert_db_to_power: bool = True

    # WORLD
    world_frame_period_ms: float = 20.0  # matches 50 Hz pitch/periodicity tracks

    # UVFP perturbation scaling (heuristics; tune later)
    jitter_scale: float = 1.0
    shimmer_scale: float = 1.0

    # Convert HNR (dB) → aperiodicity in [0,1]
    hnr_db_low: float = 0.0
    hnr_db_high: float = 20.0

    # ==========================
    # UVFP dynamics (tunable)
    # ==========================
    # Enable/disable the dynamic UVFP parameter transforms (A/B comparisons).
    apply_dynamics: bool = True

    # (1) Lower mean F0
    f0_shift_hz: float = -25.0

    # (2) Prosody compression (1.0=no compression, 0.0=monotone)
    prosody_compression: float = 0.4

    # (3) Spectral tilt (higher = darker/more low-pass)
    spectral_tilt_alpha: float = 1.5

    # (4) HF noise boost to aperiodicity
    hf_noise_boost: float = 0.15

    # (5) Amplitude decay within breath groups
    amplitude_decay_rate: float = 0.8

    # (6) Pitch drop near phrase ends (Hz)
    phrase_end_f0_drop_hz: float = 15.0

    # (7) Aphonic moments near phrase ends
    aphonia_prob: float = 0.3

    # (8) Breath breaks for overly long breath groups
    max_breath_group_frames: int = 75
    breath_break_frames: int = 10

    # (9) Diplophonia (post-synthesis mix with pitch-shifted copy)
    diplophonia_enabled: bool = False
    diplophonia_ratio: float = 0.95


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


def _detect_mel_scale(mel: np.ndarray) -> str:
    """Heuristic: log/dB mels often have negative values; linear power mels are >= 0."""
    mel = np.asarray(mel)
    if mel.size == 0:
        return "unknown"
    mmin = float(np.nanmin(mel))
    mmax = float(np.nanmax(mel))
    if mmin < -1.0:
        return "log_or_dB"
    if mmin >= 0.0 and mmax > 100.0:
        return "linear_power"
    if mmin >= 0.0 and mmax <= 100.0:
        # could be linear-power with a small dynamic range, or amplitude-scale
        return "unknown"
    return "unknown"


def _hnr_to_aperiodicity(hnr_db: float, *, low: float, high: float) -> float:
    # low HNR => noisy => high aperiodicity
    # clamp to [0,1]
    if np.isnan(hnr_db):
        return 0.5
    x = (hnr_db - low) / (high - low) if high != low else 0.5
    x = float(np.clip(x, 0.0, 1.0))
    return 1.0 - x


def apply_uvfp_dynamics(
    f0: np.ndarray,  # shape (n_frames,) float64
    sp: np.ndarray,  # shape (n_frames, n_bins) float64
    ap: np.ndarray,  # shape (n_frames, n_bins) float64
    cfg: SourceFilterConfig,
    static: dict,  # from load_static_features()
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply time-varying UVFP transformations to WORLD parameters before synthesis.

    Transformations are applied in the order specified in the task request.
    """
    rng = np.random.default_rng(seed)

    f0 = np.asarray(f0, dtype=np.float64).copy()
    sp = np.asarray(sp, dtype=np.float64).copy()
    ap = np.asarray(ap, dtype=np.float64).copy()

    if f0.ndim != 1:
        raise ValueError(f"Expected f0 shape (n_frames,), got {f0.shape}")
    if sp.ndim != 2 or ap.ndim != 2:
        raise ValueError(f"Expected sp/ap shape (n_frames, n_bins), got sp={sp.shape}, ap={ap.shape}")
    if sp.shape != ap.shape or sp.shape[0] != f0.shape[0]:
        raise ValueError(f"Frame mismatch: f0={f0.shape}, sp={sp.shape}, ap={ap.shape}")

    n_frames, n_bins = sp.shape
    voiced = f0 > 0

    # === STATIC TRANSFORMATIONS (always on, frame-uniform) ===
    # 1) LOWER MEAN F0
    if np.any(voiced):
        f0[voiced] = np.maximum(60.0, f0[voiced] + float(cfg.f0_shift_hz))

    # 2) FLATTEN PROSODY (F0 RANGE COMPRESSION)
    if np.any(voiced):
        f0_voiced_mean = float(np.mean(f0[voiced]))
        comp = float(cfg.prosody_compression)
        f0[voiced] = f0_voiced_mean + (f0[voiced] - f0_voiced_mean) * comp

    # 3) INCREASED SPECTRAL TILT
    alpha = float(cfg.spectral_tilt_alpha)
    if alpha != 0.0 and n_bins > 0:
        k = np.arange(n_bins, dtype=np.float64)
        tilt = np.exp(-alpha * k / float(n_bins))
        sp *= tilt[None, :]

    # 4) ELEVATED BROADBAND NOISE FLOOR (HIGH-FREQ)
    upper_bins = int(0.6 * n_bins)
    if 0 <= upper_bins < n_bins:
        ap[:, upper_bins:] = np.clip(ap[:, upper_bins:] + float(cfg.hf_noise_boost), 0.0, 1.0)

    # === DYNAMIC TRANSFORMATIONS (time-varying, phrase-level) ===
    # Identify voiced segments as breath groups.
    breath_groups: list[tuple[int, int]] = []
    i = 0
    while i < n_frames:
        if f0[i] > 0:
            j = i + 1
            while j < n_frames and f0[j] > 0:
                j += 1
            breath_groups.append((i, j))  # [i, j)
            i = j
        else:
            i += 1

    # 5) AMPLITUDE DECAY WITHIN BREATH GROUPS
    rate = float(cfg.amplitude_decay_rate)
    if rate > 0 and breath_groups:
        for s, e in breath_groups:
            L = e - s
            if L <= 1:
                continue
            idx = np.arange(L, dtype=np.float64)
            decay = np.exp(-rate * idx / float(L))
            sp[s:e] *= decay[:, None]

    # 6) PITCH DROP AT PHRASE ENDS
    drop_hz = float(cfg.phrase_end_f0_drop_hz)
    if drop_hz > 0 and breath_groups:
        for s, e in breath_groups:
            L = e - s
            n_tail = int(0.2 * L)
            if n_tail <= 0:
                continue
            tail_start = e - n_tail
            for j in range(n_tail):
                drop = drop_hz * (j / float(max(1, n_tail)))
                f0[tail_start + j] = max(60.0, f0[tail_start + j] - drop)

    # 7) APHONIC MOMENTS AT PHRASE ENDS (VOICING FAILURES)
    p_aph = float(cfg.aphonia_prob)
    if p_aph > 0 and breath_groups:
        for s, e in breath_groups:
            L = e - s
            n_tail = int(0.05 * L)
            if n_tail <= 0:
                continue
            tail_start = e - n_tail
            mask = rng.random(n_tail) < p_aph
            f0[tail_start:e][mask] = 0.0

    # 8) SHORTENED BREATH GROUPS (EARLY BREATH BREAKS)
    max_len = int(cfg.max_breath_group_frames)
    break_len = int(cfg.breath_break_frames)
    if max_len > 0 and break_len > 0 and breath_groups:
        for s, e in breath_groups:
            L = e - s
            if L <= max_len:
                continue
            split_point = s + max_len
            gap_end = min(e, split_point + break_len)
            f0[split_point:gap_end] = 0.0
            ap[split_point:gap_end] = 0.9

    f0 = np.where(f0 > 0, np.maximum(60.0, f0), 0.0)
    ap = np.clip(ap, 0.0, 1.0)
    sp = np.maximum(sp, 1e-12)

    return f0, sp, ap


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
    if value is None:
        return np.asarray([], dtype=np.float32)

    # Common cases:
    # - list[list[float]]
    # - list[np.ndarray] with consistent lengths
    # - np.ndarray(dtype=object) of sequences
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        rows = [np.asarray(r, dtype=np.float32) for r in value]
        if not rows:
            return np.asarray([], dtype=np.float32)
        # stack is stricter than asarray and will surface shape mismatches clearly
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

    return np.asarray(value, dtype=np.float32)


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
        mel = _ensure_2d_time_major(mel)

        scale = _detect_mel_scale(mel)
        if scale != "linear_power":
            warnings.warn(
                f"Mel scale detected as '{scale}' for {participant_id}/{session_id}/{task_name}. "
                "If mel values are log/dB, invert (e.g., db_to_power) before mel_to_audio(power=2.0).",
                RuntimeWarning,
            )
            if scale == "log_or_dB" and cfg.mel_scale_autoconvert_db_to_power:
                # If these are dB values, this is the correct inverse for power.
                # If they are ln(power) instead, this will still be wrong; the warning remains.
                mel = librosa.db_to_power(mel, ref=1.0).astype(np.float32)

        return mel

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
    print(f"SPARC pitch frames: len(f0)={len(f0)}")
    # If SPARC pitch was NOT subsampled, this will be 2x the mel frame count.
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
    n_mel_frames: int | None = None,
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

    if n_mel_frames is not None and n_mel_frames > 0 and len(f0) > 0:
        ratio = len(f0) / float(n_mel_frames)
        if 0.9 <= ratio <= 1.1:
            # Aligned within tolerance (off-by-one is common); no subsampling needed.
            pass
        elif 1.9 <= ratio <= 2.1:
            print(
                "WARNING: pitch/mel frame mismatch detected "
                f"(len(f0)={len(f0)} ~= 2 * n_mel_frames={n_mel_frames}). "
                "Subsampling pitch: f0 = f0[::2]"
            )
            f0 = f0[::2]
            if periodicity is not None and len(periodicity) >= 2 * n_mel_frames:
                periodicity = periodicity[::2]
        else:
            print(f"WARNING: unexpected pitch/mel frame ratio={ratio:.3f} — proceeding without subsampling")

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

    if cfg.apply_dynamics:
        f0, sp, ap = apply_uvfp_dynamics(
            f0=f0,
            sp=sp,
            ap=ap,
            cfg=cfg,
            static={
                "jitterLocal_sma3nz_amean": jitter_local,
                "shimmerLocaldB_sma3nz_amean": shimmer_db,
                "HNRdBACF_sma3nz_amean": hnr_db,
            },
            seed=seed,
        )

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

    if cfg.diplophonia_enabled:
        # WORLD cannot directly synthesize dual-F0; approximate by mixing with
        # a slightly pitch-shifted copy post-hoc.
        # Use a fixed, small detune by default. cfg.diplophonia_ratio is retained as a
        # tunable knob, but pitch-shifting is applied in semitone space.
        y_shifted = librosa.effects.pitch_shift(y, sr=cfg.sample_rate, n_steps=-0.5).astype(np.float32)
        y = (0.7 * y + 0.3 * y_shifted).astype(np.float32)
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
        # power=2.0 is correct because the dataset stores **power-scale** mel values.
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


def check_frame_alignment(
    cfg: SourceFilterConfig,
    *,
    participant_id: str,
    session_id: str,
    task_name: str,
) -> None:
    mel = load_filter_representation(
        cfg,
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        mode="mel",
    )
    f0 = load_pitch_track(cfg, participant_id=participant_id, session_id=session_id, task_name=task_name)

    print(f"mel frames: mel.shape[1]={mel.shape[1]}")
    print(f"pitch frames: len(f0)={len(f0)}")

    ratio = len(f0) / float(mel.shape[1]) if mel.shape[1] else float("inf")
    if 0.9 <= ratio <= 1.1:
        print("ALIGNED: pitch and mel at same frame rate (within 10% tolerance)")
        print("  1-frame discrepancy is normal — trim/pad will handle it")
    elif 1.9 <= ratio <= 2.1:
        print("MISMATCH: pitch at 10ms, mel at 20ms — subsampling pitch by 2")
        # (printing the suggestion here; the synthesis path will do the subsampling if needed)
    else:
        print(f"UNKNOWN: ratio={ratio:.3f} — inspect manually")


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
    mel_for_frames = load_filter_representation(
        cfg,
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        mode="mel",
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
        n_mel_frames=int(mel_for_frames.shape[1]),
        periodicity=periodicity,
        seed=seed,
    )

    if combine_mode == "mel_only":
        y = reconstruct_filter_waveform_from_mel(cfg, mel_for_frames)
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


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check-alignment", action="store_true")
    p.add_argument("--participant-id")
    p.add_argument("--session-id")
    p.add_argument("--task-name")
    args = p.parse_args()

    if args.check_alignment:
        missing = [
            name
            for name in ("participant_id", "session_id", "task_name")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"Missing required args for --check-alignment: {', '.join(missing)}")

        cfg = SourceFilterConfig()
        check_frame_alignment(
            cfg,
            participant_id=args.participant_id,
            session_id=args.session_id,
            task_name=args.task_name,
        )


if __name__ == "__main__":
    _main()

