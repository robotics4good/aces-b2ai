from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aces_b2ai.context import ClipContext
    from aces_b2ai.models.feature_descriptor import FeatureDescriptor


@dataclass
class ExtractionResult:
    """Scalar secondary features + metadata for one clip."""

    features: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class BaseExtractor(ABC):
    """One logical family of secondary features (e.g. pitch trajectory).

    Subclasses that also produce tensor outputs (2D spectrograms or ND time
    series) should override ``tensor_outputs`` with a list of
    ``FeatureDescriptor`` objects and write the corresponding arrays to
    ``ctx.extras[descriptor.name]`` inside ``extract()``.  The
    ``FeatureBundler`` will pick them up automatically.
    """

    name: str = "base"

    #: Declare tensor outputs produced by this extractor.  Each descriptor
    #: must have ``source_clip_field=None`` (i.e. written to ctx.extras).
    tensor_outputs: list[FeatureDescriptor] = []

    @abstractmethod
    def extract(self, ctx: ClipContext) -> ExtractionResult:
        raise NotImplementedError
