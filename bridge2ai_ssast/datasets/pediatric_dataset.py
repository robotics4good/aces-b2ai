"""Pediatric Bridge2AI dataset loader with multi-select condition parsing"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from bridge2ai_ssast.datasets.utils import pad_or_truncate, normalize_mel


class PediatricBridge2AIDataset(Dataset):
    """
    Pediatric Bridge2AI Voice Dataset (v1.0.0) for multi-label condition classification.

    Expects preprocessed 128-bin mel spectrograms @ 50Hz.

    IMPORTANT: Pediatric conditions must be pre-parsed from multi-select fields into
    binary columns via scripts/parse_pediatric_conditions.py

    Args:
        mel_parquet_path: Path to pediatric mel_128bin_50hz.parquet
        conditions_expanded_path: Path to pediatric_conditions_expanded.tsv
            (Output of parse_pediatric_conditions.py with binary condition columns)
        condition_mapping: Dict mapping condition names to expanded TSV column names
            Example: {'asthma': 'has_asthma', 'hearing_loss': 'has_hearing_loss', ...}
        target_length: Number of time frames to pad/truncate to (default 200 ≈ 4s @ 50Hz)
        normalize: If True, apply SSAST normalization using mean/std
        mean: Global mean for normalization
        std: Global std for normalization
        tasks: Optional list of task names to filter (e.g., ['long-sounds', 'passage'])
        participant_ids: Optional list of participant IDs to filter (for train/val/test splits)
    """

    def __init__(
        self,
        mel_parquet_path,
        conditions_expanded_path,
        condition_mapping,
        target_length=200,
        normalize=True,
        mean=None,
        std=None,
        tasks=None,
        participant_ids=None,
    ):
        # Load mel spectrograms
        self.mels = pq.read_table(mel_parquet_path).to_pandas()

        # Filter by tasks if specified
        if tasks is not None:
            self.mels = self.mels[self.mels['task_name'].isin(tasks)]

        # Filter by participant IDs (for stratified splitting)
        if participant_ids is not None:
            self.mels = self.mels[self.mels['participant_id'].isin(participant_ids)]

        # Load expanded condition labels
        self.conditions_df = pd.read_csv(conditions_expanded_path, sep='\t')

        # Merge on participant_id
        self.data = self.mels.merge(
            self.conditions_df,
            on='participant_id',
            how='inner'
        )

        print(f"Pediatric dataset: {len(self.data)} recordings after merging conditions")

        # Store condition mapping
        self.condition_mapping = condition_mapping
        self.conditions = list(condition_mapping.keys())
        self.condition_cols = {
            cond: condition_mapping[cond] for cond in self.conditions
        }

        # Verify all condition columns exist
        for cond_name, col_name in self.condition_cols.items():
            if col_name not in self.data.columns:
                raise ValueError(
                    f"Condition column '{col_name}' for '{cond_name}' not found in expanded conditions TSV"
                )

        # Normalization params
        self.target_length = target_length
        self.normalize = normalize
        self.mean = mean if mean is not None else 0.0
        self.std = std if std is not None else 1.0

        if normalize and (mean is None or std is None):
            print("⚠️  Warning: normalize=True but mean/std not provided. Using 0/1.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            mel: torch.Tensor of shape [target_length, 128] (time, freq)
            labels: torch.Tensor of shape [num_conditions] with binary 0/1 values
        """
        row = self.data.iloc[idx]

        # Load mel spectrogram [128, T]
        mel_data = row['mel_spectrogram']
        if isinstance(mel_data, np.ndarray) and mel_data.dtype == object:
            # Array of arrays from parquet → stack into 2D array
            mel = np.stack(mel_data).astype(np.float32)
        elif isinstance(mel_data, np.ndarray):
            mel = mel_data.astype(np.float32)
        else:
            mel = np.array(mel_data, dtype=np.float32)

        # Pad/truncate to [target_length, 128]
        mel = pad_or_truncate(mel, self.target_length)

        # Normalize
        if self.normalize:
            mel = normalize_mel(mel, self.mean, self.std)

        # Extract multi-label targets (already binary 0/1 from expanded TSV)
        labels = np.zeros(len(self.conditions), dtype=np.float32)
        for i, cond_name in enumerate(self.conditions):
            col_name = self.condition_cols[cond_name]
            labels[i] = float(row[col_name])

        return torch.tensor(mel, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def get_condition_counts(self):
        """Returns dict of condition prevalence counts"""
        counts = {}
        for cond_name, col_name in self.condition_cols.items():
            counts[cond_name] = self.data[col_name].sum()
        return counts

    def get_label_distribution(self):
        """Returns label distribution statistics"""
        labels_array = np.zeros((len(self.data), len(self.conditions)))
        for i in range(len(self.data)):
            _, labels = self[i]
            labels_array[i] = labels.numpy()

        return {
            'condition_names': self.conditions,
            'positive_counts': labels_array.sum(axis=0),
            'prevalence': labels_array.mean(axis=0),
        }

    def get_participant_ids(self):
        """Returns unique participant IDs in this dataset"""
        return self.data['participant_id'].unique().tolist()
