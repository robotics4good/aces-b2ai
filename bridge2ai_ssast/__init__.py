"""
bridge2ai_ssast - SSAST Transfer Learning for Bridge2AI Voice Datasets

A library for training and fine-tuning SSAST models on Bridge2AI adult and pediatric
voice datasets to study acoustic biomarker generalization across age groups.

Research Question:
    How well do spectrogram-based representations learned from adult voice data
    generalize to pediatric populations for medical condition detection, and which
    acoustic biomarkers are shared versus age-specific?

Modules:
    datasets - Adult and pediatric dataset loaders with condition label parsing
    preprocessing - Linear spectrogram → mel conversion, normalization
    models - SSAST model wrappers and configuration
    training - Training loops, evaluation, and checkpointing
    evaluation - Metrics, visualization, and interpretability tools
"""

__version__ = "0.1.0"

from bridge2ai_ssast.datasets.adult_dataset import AdultBridge2AIDataset
from bridge2ai_ssast.datasets.pediatric_dataset import PediatricBridge2AIDataset

__all__ = [
    "AdultBridge2AIDataset",
    "PediatricBridge2AIDataset",
]
