"""Adult Bridge2AI dataset loader with medical condition labels"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from bridge2ai_ssast.datasets.utils import pad_or_truncate, normalize_mel


class AdultBridge2AIDataset(Dataset):
    """
    Adult Bridge2AI Voice Dataset (v2.0.1) for multi-label condition classification.

    Expects preprocessed 128-bin mel spectrograms @ 50Hz (subsampled from 100Hz).

    Args:
        mel_parquet_path: Path to adult mel_128bin_50hz.parquet
        phenotype_path: Path to adult phenotype.tsv
        condition_mapping: Dict mapping condition names to phenotype column names
            Example: {'asthma': 'asthma', 'allergies': 'seasonal_allergies',
                     'hearing_loss': 'hearing_loss', ...}
        target_length: Number of time frames to pad/truncate to (default 200 ≈ 4s @ 50Hz)
        normalize: If True, apply SSAST normalization using mean/std
        mean: Global mean for normalization (compute via scripts/compute_mel128_normalization.py)
        std: Global std for normalization
        tasks: Optional list of task names to filter (e.g., ['passage', 'sentence'])
        age_range: Optional tuple (min_age, max_age) to filter by age (e.g., (30, 50) for middle-aged adults)
    """

    def __init__(
        self,
        mel_parquet_path,
        phenotype_path,
        condition_mapping,
        target_length=200,
        normalize=True,
        mean=None,
        std=None,
        tasks=None,
        age_range=None,
    ):
        # Load mel spectrograms
        self.mels = pq.read_table(mel_parquet_path).to_pandas()

        # Load phenotype data (need it for age filtering)
        self.phenotype = pd.read_csv(phenotype_path, sep='\t', low_memory=False)

        # Filter by age range if specified
        if age_range is not None:
            min_age, max_age = age_range
            age_filtered = self.phenotype[
                (self.phenotype['age'] >= min_age) &
                (self.phenotype['age'] <= max_age)
            ]
            valid_participant_sessions = list(
                zip(age_filtered['participant_id'], age_filtered['session_id'])
            )
            # Create a merged key for filtering
            self.mels['_key'] = list(zip(self.mels['participant_id'], self.mels['session_id']))
            self.mels = self.mels[self.mels['_key'].isin(valid_participant_sessions)]
            self.mels = self.mels.drop(columns=['_key'])
            print(f"Age filter ({min_age}-{max_age}): {len(age_filtered)} participants/sessions")

        # Filter by tasks if specified
        if tasks is not None:
            self.mels = self.mels[self.mels['task_name'].isin(tasks)]

        # Merge on participant_id and session_id
        self.data = self.mels.merge(
            self.phenotype,
            on=['participant_id', 'session_id'],
            how='inner'
        )

        print(f"Adult dataset: {len(self.data)} recordings after merging phenotype")

        # Store condition mapping and extract condition names
        self.condition_mapping = condition_mapping
        self.conditions = list(condition_mapping.keys())
        self.condition_cols = {
            cond: condition_mapping[cond] for cond in self.conditions
        }

        # Verify all condition columns exist
        for cond_name, col_name in self.condition_cols.items():
            if col_name not in self.data.columns:
                raise ValueError(
                    f"Condition column '{col_name}' for '{cond_name}' not found in phenotype data"
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

        # Extract multi-label targets
        labels = np.zeros(len(self.conditions), dtype=np.float32)
        for i, cond_name in enumerate(self.conditions):
            col_name = self.condition_cols[cond_name]
            # Handle different encodings: 1/0, yes/no, True/False
            value = row[col_name]
            if pd.isna(value):
                labels[i] = 0.0
            elif isinstance(value, (bool, np.bool_)):
                labels[i] = float(value)
            elif isinstance(value, str):
                labels[i] = 1.0 if value.lower() in ['yes', 'true', '1'] else 0.0
            else:
                labels[i] = float(value)

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
