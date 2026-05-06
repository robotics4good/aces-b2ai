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

    # Stream from parquet file using ParquetFile API
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows to process: {total_rows}")

    # Process in row group batches (stream, don't load all at once)
    batch_size = 500  # Smaller batches for memory safety
    processed_count = 0
    batch_files = []

    import tempfile
    import os
    temp_dir = tempfile.mkdtemp(prefix="mel_conversion_")

    try:
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            batch_df = batch.to_pandas()

            adult_mel_rows = []
            for idx, row in tqdm(batch_df.iterrows(),
                                total=len(batch_df),
                                desc=f"Converting adult [{processed_count}-{processed_count+len(batch_df)}]",
                                leave=False):
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
                    'mel_spectrogram': [mel_spec[i, :].tolist() for i in range(mel_spec.shape[0])],  # List of lists for PyArrow
                    'n_frames': mel_spec.shape[1],
                })

            # Save batch to temporary parquet file
            batch_mel_df = pd.DataFrame(adult_mel_rows)
            batch_file = os.path.join(temp_dir, f"batch_{batch_idx:04d}.parquet")
            batch_mel_df.to_parquet(batch_file, compression='zstd', index=False)
            batch_files.append(batch_file)

            processed_count += len(batch_mel_df)
            print(f"  ✅ Saved batch {batch_idx} ({processed_count}/{total_rows} total) - {len(batch_mel_df)} recordings")

        # Concatenate all batch files into final output
        print(f"📦 Concatenating {len(batch_files)} batch files into {output_path}...")
        all_tables = [pq.read_table(f) for f in batch_files]
        import pyarrow as pa
        final_table = pa.concat_tables(all_tables)
        pq.write_table(final_table, output_path, compression='zstd')

        print(f"✅ Saved {processed_count} mel spectrograms to {output_path}")

    finally:
        # Clean up temporary files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🗑️  Cleaned up temporary directory")


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
            'mel_spectrogram': mel_spec,  # Keep as numpy array (not .tolist())
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
