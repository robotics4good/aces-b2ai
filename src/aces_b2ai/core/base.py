from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aces_b2ai.context import ClipContext


@dataclass
class ExtractionResult:
    """Scalar secondary features + metadata for one clip."""

    features: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class BaseExtractor(ABC):
    """One logical family of secondary features (e.g. pitch trajectory)."""

    name: str = "base"

    @abstractmethod
    def extract(self, ctx: ClipContext) -> ExtractionResult:
        raise NotImplementedError
