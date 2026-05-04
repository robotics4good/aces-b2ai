from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class FeatureType(Enum):
    """Routing class for a feature: determines which model branch consumes it."""

    SCALAR = auto()       # single float per clip  → tabular branch (sklearn)
    SPECTROGRAM = auto()  # (freq_bins, T) ndarray  → 2D CNN branch
    TIMESERIES = auto()   # (n_channels, T) ndarray → 1D TCN branch


@dataclass(frozen=True)
class FeatureDescriptor:
    """Describes one feature produced by an extractor.

    Parameters
    ----------
    name:
        Unique key; must match the dict key in ``ExtractionResult.features``
        (for SCALAR) or the key written to ``ClipContext.extras`` (for
        SPECTROGRAM / TIMESERIES).
    feature_type:
        Routes the feature to the correct model branch.
    shape_hint:
        Expected array shape with ``None`` for variable axes, e.g.
        ``(60, None)`` for a 60-bin mel of variable length.  Empty tuple
        ``()`` for scalars.
    frame_rate_hz:
        Temporal sampling rate of the feature (``None`` for scalars).
    extractor_name:
        Name of the extractor that produces this feature (informational only).
    source_clip_field:
        If set, the bundler reads ``getattr(ctx, source_clip_field)`` instead
        of ``ctx.extras[name]``.  Set for built-in ClipContext fields.
    needs_transpose:
        If ``True``, the bundler transposes the array before placing it in the
        bundle.  Needed for ``ema_sparc`` which is stored as ``(T, C)`` on
        ClipContext but should be ``(C, T)`` for model branches.
    """

    name: str
    feature_type: FeatureType
    shape_hint: tuple = field(default=())
    frame_rate_hz: float | None = None
    extractor_name: str = ""
    source_clip_field: str | None = None
    needs_transpose: bool = False


# ---------------------------------------------------------------------------
# Built-in descriptors for the standard ClipContext fields.
# These are always available without any extractor registration.
# ---------------------------------------------------------------------------

DEFAULT_CLIP_DESCRIPTORS: list[FeatureDescriptor] = [
    FeatureDescriptor(
        name="mel",
        feature_type=FeatureType.SPECTROGRAM,
        shape_hint=(60, None),
        frame_rate_hz=50.0,
        source_clip_field="mel",
    ),
    FeatureDescriptor(
        name="mfcc",
        feature_type=FeatureType.SPECTROGRAM,
        shape_hint=(60, None),
        frame_rate_hz=50.0,
        source_clip_field="mfcc",
    ),
    FeatureDescriptor(
        name="spectrogram",
        feature_type=FeatureType.SPECTROGRAM,
        shape_hint=(201, None),
        frame_rate_hz=50.0,
        source_clip_field="spectrogram",
    ),
    # EMA is stored as (T, 12) on ClipContext → transpose to (12, T) for TCN.
    FeatureDescriptor(
        name="ema_sparc",
        feature_type=FeatureType.TIMESERIES,
        shape_hint=(12, None),
        frame_rate_hz=50.0,
        source_clip_field="ema_sparc",
        needs_transpose=True,
    ),
]
