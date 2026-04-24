from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from aces_b2ai.align import prepare_clip_context
from aces_b2ai.config import PipelineConfig
from aces_b2ai.context import ClipContext
from aces_b2ai.core.base import ExtractionResult
from aces_b2ai.core.registry import ExtractorRegistry
from aces_b2ai.extractors.crossmodal import CrossmodalExtractor
from aces_b2ai.extractors.dynamical import DynamicalExtractor
from aces_b2ai.extractors.mel_secondary import MelSecondaryExtractor
from aces_b2ai.extractors.mfcc_secondary import MfccSecondaryExtractor
from aces_b2ai.extractors.pitch_secondary import PitchSecondaryExtractor
from aces_b2ai.loaders import parquet_row_to_numpy_1d, parquet_row_to_numpy_2d


def default_registry(cfg: PipelineConfig) -> ExtractorRegistry:
    reg = ExtractorRegistry(
        [
            PitchSecondaryExtractor(),
            MfccSecondaryExtractor(),
            MelSecondaryExtractor(),
            CrossmodalExtractor(),
        ]
    )
    if cfg.enable_dynamical:
        reg.register(DynamicalExtractor(min_voiced_frames=cfg.dynamical_min_voiced_frames))
    return reg


class FeatureExtractionPipeline:
    """End-to-end secondary feature extraction for one clip (combined doc §22)."""

    def __init__(
        self,
        cfg: PipelineConfig | None = None,
        registry: ExtractorRegistry | None = None,
    ) -> None:
        self.cfg = cfg or PipelineConfig()
        self.registry = registry or default_registry(self.cfg)

    def extract(
        self,
        *,
        participant_id: str,
        session_id: str,
        task_name: str,
        mel: np.ndarray | list | None = None,
        mfcc: np.ndarray | list | None = None,
        spectrogram: np.ndarray | list | None = None,
        pitch: np.ndarray | list | None = None,
        periodicity_sparc: np.ndarray | list | None = None,
        pitch_sparc: np.ndarray | list | None = None,
        static_features: dict[str, float] | None = None,
        age_years: float | None = None,
        sex: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        def _arr2d(x):
            if x is None:
                return None
            return np.asarray(x, dtype=np.float64)

        def _arr1d(x):
            if x is None:
                return None
            return np.asarray(x, dtype=np.float64).ravel()

        ctx = ClipContext(
            participant_id=str(participant_id),
            session_id=str(session_id),
            task_name=str(task_name),
            mel=_arr2d(mel),
            mfcc=_arr2d(mfcc),
            spectrogram=_arr2d(spectrogram),
            pitch_torch=_arr1d(pitch),
            periodicity_sparc=_arr1d(periodicity_sparc),
            pitch_sparc=_arr1d(pitch_sparc),
            static_features=dict(static_features or {}),
            age_years=age_years,
            sex=sex,
        )
        ctx = prepare_clip_context(ctx, self.cfg)
        out = self.registry.run_all(ctx)
        meta = dict(out.metadata)
        meta["pipeline"] = {"config": asdict(self.cfg)}
        if provenance:
            meta["provenance"] = provenance
        return ExtractionResult(features=out.features, metadata=meta, warnings=out.warnings)

    @staticmethod
    def from_torchaudio_parquet_row(
        row: dict[str, Any],
        *,
        cfg: PipelineConfig | None = None,
        mel_key: str = "mel_spectrogram",
        mfcc_key: str = "mfcc",
        spec_key: str = "spectrograms",
        pitch_key: str = "pitch",
    ) -> ExtractionResult:
        mel = row.get(mel_key)
        mfcc = row.get(mfcc_key)
        spec = row.get(spec_key)
        pitch = row.get(pitch_key)
        mel_a = parquet_row_to_numpy_2d(mel) if mel is not None else None
        mfcc_a = parquet_row_to_numpy_2d(mfcc) if mfcc is not None else None
        spec_a = parquet_row_to_numpy_2d(spec) if spec is not None else None
        pitch_a = parquet_row_to_numpy_1d(pitch) if pitch is not None else None
        pipe = FeatureExtractionPipeline(cfg=cfg)
        return pipe.extract(
            participant_id=str(row["participant_id"]),
            session_id=str(row["session_id"]),
            task_name=str(row["task_name"]),
            mel=mel_a,
            mfcc=mfcc_a,
            spectrogram=spec_a,
            pitch=pitch_a,
            provenance={"source": "torchaudio_parquet_row"},
        )
