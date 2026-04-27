"""ACES B2AI — secondary feature extraction from Bridge2AI tensor + static stores."""

from aces_b2ai.core.base import ExtractionResult
from aces_b2ai.pipeline import FeatureExtractionPipeline
from aces_b2ai.stress_labels import load_stress_labels_csv, merge_stress_labels
from aces_b2ai.tasks import (
    aggregate_secondary_by_task,
    filter_rows_by_tasks,
    is_sustained_phonation_task,
    task_theme,
)

__all__ = [
    "ExtractionResult",
    "FeatureExtractionPipeline",
    "aggregate_secondary_by_task",
    "filter_rows_by_tasks",
    "is_sustained_phonation_task",
    "load_stress_labels_csv",
    "merge_stress_labels",
    "task_theme",
    "__version__",
]

__version__ = "0.1.0"
