"""
Convert 201-bin linear spectrograms to 128-bin mel spectrograms for SSAST input.

Adult: 201-bin linear @ 100Hz → subsample to 50Hz → convert to 128 mel bins
Pediatric: 201-bin linear @ 50Hz → convert to 128 mel bins (no subsampling)

Usage:
    python scripts/convert_linear_to_128mel.py --dataset adult
    python scripts/convert_linear_to_128mel.py --dataset pediatric
"""

import argparse
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def convert_linear_to_mel(linear_spec, n_mels=128, sample_rate=16000, n_fft=400):
    """
    Convert linear-scale spectrogram to mel-scale using torchaudio.

    Args:
        linear_spec: np.ndarray of shape [201, T] (frequency bins, time)
        n_mels: Number of mel bins (default 128 for SSAST)
        sample_rate: Original audio sample rate (16000 Hz)
        n_fft: FFT size used to generate linear spectrogram (400)

    Returns:
        mel_spec: np.ndarray of shape [128, T]
    """
    # Convert to torch tensor
    linear_tensor = torch.from_numpy(linear_spec.astype(np.float32))

    # Create mel filterbank transform
    mel_transform = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=0.0,
        f_max=sample_rate / 2,
        n_stft=n_fft // 2 + 1,  # 201 frequency bins
    )

    # Apply mel filterbank to linear spectrogram
    mel_spec = mel_transform(linear_tensor)

    return mel_spec.numpy()


def process_adult_dataset(input_path, output_path):
    """Process adult dataset: subsample 100Hz → 50Hz, convert to 128 mel bins"""
    print("Processing adult dataset...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Load adult spectrograms
    adult_table = pq.read_table(input_path)
    adult_df = adult_table.to_pandas()

    print(f"Loaded {len(adult_df)} adult spectrograms")

    # Process each recording
    adult_mel_rows = []
    for idx, row in tqdm(adult_df.iterrows(), total=len(adult_df), desc="Converting adult"):
        # Handle parquet array of arrays
        spec_data = row['spectrogram']
        if isinstance(spec_data, np.ndarray) and spec_data.dtype == object:
            linear_spec = np.stack(spec_data).astype(np.float32)  # [201, T]
        else:
            linear_spec = np.array(spec_data, dtype=np.float32)  # [201, T]

        # Subsample time axis: 100Hz → 50Hz to match pediatric
        linear_spec = linear_spec[:, ::2]

        # Convert to 128-bin mel
        mel_spec = convert_linear_to_mel(linear_spec, n_mels=128)

        adult_mel_rows.append({
            'participant_id': row['participant_id'],
            'session_id': row['session_id'],
            'task_name': row['task_name'],
            'mel_spectrogram': mel_spec.tolist(),
            'n_frames': mel_spec.shape[1],
        })

    # Save to parquet
    adult_mel_df = pd.DataFrame(adult_mel_rows)
    adult_mel_df.to_parquet(output_path, compression='zstd')

    print(f"✅ Saved {len(adult_mel_df)} mel spectrograms to {output_path}")
    print(f"   Shape: [128, T] where T varies (median: {adult_mel_df['n_frames'].median():.0f} frames)")


def process_pediatric_dataset(input_path, output_path):
    """Process pediatric dataset: convert to 128 mel bins (already 50Hz)"""
    print("Processing pediatric dataset...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Load pediatric spectrograms
    ped_table = pq.read_table(input_path)
    ped_df = ped_table.to_pandas()

    print(f"Loaded {len(ped_df)} pediatric spectrograms")

    # Process each recording
    ped_mel_rows = []
    for idx, row in tqdm(ped_df.iterrows(), total=len(ped_df), desc="Converting pediatric"):
        # Handle parquet array of arrays
        spec_data = row['spectrogram']
        if isinstance(spec_data, np.ndarray) and spec_data.dtype == object:
            linear_spec = np.stack(spec_data).astype(np.float32)  # [201, T]
        else:
            linear_spec = np.array(spec_data, dtype=np.float32)  # [201, T]

        # No time subsampling needed (already 50Hz)
        mel_spec = convert_linear_to_mel(linear_spec, n_mels=128)

        ped_mel_rows.append({
            'participant_id': row['participant_id'],
            'session_id': row['session_id'],
            'task_name': row['task_name'],
            'mel_spectrogram': mel_spec.tolist(),
            'n_frames': mel_spec.shape[1],
        })

    # Save to parquet
    ped_mel_df = pd.DataFrame(ped_mel_rows)
    ped_mel_df.to_parquet(output_path, compression='zstd')

    print(f"✅ Saved {len(ped_mel_df)} mel spectrograms to {output_path}")
    print(f"   Shape: [128, T] where T varies (median: {ped_mel_df['n_frames'].median():.0f} frames)")


def main():
    parser = argparse.ArgumentParser(description="Convert linear spectrograms to 128-bin mel")
    parser.add_argument(
        "--dataset",
        choices=["adult", "pediatric"],
        required=True,
        help="Which dataset to process"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input parquet path (overrides defaults)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output parquet path (overrides defaults)"
    )

    args = parser.parse_args()

    # Default paths
    if args.dataset == "adult":
        default_input = "adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/spectrogram.parquet"
        default_output = "adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/mel_128bin_50hz.parquet"
    else:  # pediatric
        default_input = "features/torchaudio_spectrogram.parquet"
        default_output = "features/mel_128bin_50hz.parquet"

    input_path = args.input if args.input else default_input
    output_path = args.output if args.output else default_output

    # Process
    if args.dataset == "adult":
        process_adult_dataset(input_path, output_path)
    else:
        process_pediatric_dataset(input_path, output_path)


if __name__ == "__main__":
    main()
