from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from aces_b2ai.models.feature_descriptor import (
    DEFAULT_CLIP_DESCRIPTORS,
    FeatureDescriptor,
    FeatureType,
)

if TYPE_CHECKING:
    from aces_b2ai.context import ClipContext
    from aces_b2ai.core.base import ExtractionResult


@dataclass
class FeatureBundle:
    """Typed container of all features for one clip.

    Attributes
    ----------
    scalars:
        Dict of scalar features from all ``ExtractionResult.features`` dicts.
    spectrograms:
        Dict of (freq_bins, T) arrays, keyed by ``FeatureDescriptor.name``.
    series:
        Dict of (n_channels, T) arrays, keyed by ``FeatureDescriptor.name``.
    label_binary:
        0 = healthy, 1 = pathological.  ``None`` if labels are not available.
    label_multiclass:
        Integer class index for the specific disorder.  ``None`` if not set.
    participant_id / task_name:
        Pass-through identifiers for bookkeeping.
    """

    participant_id: str = ""
    task_name: str = ""
    scalars: dict[str, float] = field(default_factory=dict)
    spectrograms: dict[str, np.ndarray] = field(default_factory=dict)
    series: dict[str, np.ndarray] = field(default_factory=dict)
    label_binary: int | None = None
    label_multiclass: int | None = None


class FeatureBundler:
    """Assembles a ``FeatureBundle`` from a ``ClipContext`` + extractor outputs.

    Parameters
    ----------
    extra_descriptors:
        ``FeatureDescriptor`` objects for any tensor features written to
        ``ctx.extras`` by custom extractors.  Built-in ClipContext fields are
        handled automatically via ``DEFAULT_CLIP_DESCRIPTORS``.
    """

    def __init__(
        self,
        extra_descriptors: list[FeatureDescriptor] | None = None,
    ) -> None:
        self._descriptors: list[FeatureDescriptor] = list(DEFAULT_CLIP_DESCRIPTORS)
        if extra_descriptors:
            self._descriptors.extend(extra_descriptors)

    def register(self, descriptor: FeatureDescriptor) -> None:
        """Add a descriptor at runtime (e.g. from a newly registered extractor)."""
        self._descriptors.append(descriptor)

    def build(
        self,
        ctx: ClipContext,
        results: list[ExtractionResult] | None = None,
        *,
        label_binary: int | None = None,
        label_multiclass: int | None = None,
    ) -> FeatureBundle:
        """Build a ``FeatureBundle`` from a prepared ``ClipContext``.

        Parameters
        ----------
        ctx:
            Populated ``ClipContext`` after the extraction pipeline ran.
        results:
            ``ExtractionResult`` objects from all extractors.  Scalars from
            their ``.features`` dicts are merged into ``bundle.scalars``.
        label_binary / label_multiclass:
            Optional diagnosis labels.
        """
        bundle = FeatureBundle(
            participant_id=ctx.participant_id,
            task_name=ctx.task_name,
            label_binary=label_binary,
            label_multiclass=label_multiclass,
        )

        # --- Scalars: merge all ExtractionResult feature dicts ---------------
        for res in (results or []):
            bundle.scalars.update(res.features)

        # --- Tensor features: route by FeatureType ---------------------------
        for desc in self._descriptors:
            if desc.feature_type == FeatureType.SCALAR:
                continue  # scalars already collected from ExtractionResult

            arr = self._read_tensor(ctx, desc)
            if arr is None:
                continue

            if desc.needs_transpose:
                arr = arr.T  # e.g. ema_sparc (T, C) → (C, T)

            if desc.feature_type == FeatureType.SPECTROGRAM:
                bundle.spectrograms[desc.name] = arr
            elif desc.feature_type == FeatureType.TIMESERIES:
                bundle.series[desc.name] = arr

        return bundle

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_tensor(ctx: ClipContext, desc: FeatureDescriptor) -> np.ndarray | None:
        """Return the raw numpy array for a descriptor, or None if absent."""
        if desc.source_clip_field is not None:
            arr = getattr(ctx, desc.source_clip_field, None)
        else:
            arr = ctx.extras.get(desc.name)

        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        # Guard: only return 2-D arrays.
        return arr if arr.ndim == 2 else (arr[np.newaxis, :] if arr.ndim == 1 else None)
