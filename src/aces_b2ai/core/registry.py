from __future__ import annotations

from aces_b2ai.context import ClipContext
from aces_b2ai.core.base import BaseExtractor, ExtractionResult


class ExtractorRegistry:
    """Runs ordered secondary-feature extractors on a prepared ClipContext."""

    def __init__(self, extractors: list[BaseExtractor] | None = None) -> None:
        self._extractors: list[BaseExtractor] = list(extractors or [])

    def register(self, extractor: BaseExtractor) -> None:
        self._extractors.append(extractor)

    def run_all(self, ctx: ClipContext) -> ExtractionResult:
        features: dict[str, float] = {}
        metadata: dict[str, dict] = {}
        warnings: list[str] = []
        for ex in self._extractors:
            res = ex.extract(ctx)
            for k, v in res.features.items():
                if k in features:
                    warnings.append(f"duplicate feature key overwritten: {k}")
                features[k] = v
            if res.metadata:
                metadata[ex.name] = res.metadata
            warnings.extend(res.warnings)
        return ExtractionResult(features=features, metadata=metadata, warnings=warnings)
