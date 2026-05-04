"""
Compute mean and std statistics for 128-bin mel spectrograms.

These statistics are used for SSAST normalization: (x - mean) / (std * 2)

Usage:
    python scripts/compute_mel128_normalization.py --dataset adult
    python scripts/compute_mel128_normalization.py --dataset pediatric
"""

import argparse
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm


def compute_normalization_stats(parquet_path, sample_limit=None):
    """
    Compute mean/std for 128-bin mel spectrograms.

    Args:
        parquet_path: Path to mel_128bin_50hz.parquet
        sample_limit: Optional limit for faster computation (e.g., 5000 samples)

    Returns:
        mean, std (float)
    """
    print(f"Loading mel spectrograms from {parquet_path}...")
    df = pq.read_table(parquet_path).to_pandas()

    if sample_limit is not None and len(df) > sample_limit:
        print(f"Sampling {sample_limit} recordings (out of {len(df)})")
        df = df.sample(sample_limit, random_state=42)

    print(f"Computing normalization stats from {len(df)} spectrograms...")

    all_values = []
    for mel in tqdm(df['mel_spectrogram'], desc="Collecting values"):
        mel_array = np.array(mel, dtype=np.float32)  # [128, T]
        all_values.append(mel_array.flatten())

    # Concatenate all values
    all_values = np.concatenate(all_values)

    mean = float(np.mean(all_values))
    std = float(np.std(all_values))

    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Compute mel spectrogram normalization stats")
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
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of samples for faster computation (default: use all)"
    )

    args = parser.parse_args()

    # Default paths
    if args.dataset == "adult":
        default_input = "adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/mel_128bin_50hz.parquet"
        output_prefix = "adult"
    else:  # pediatric
        default_input = "features/mel_128bin_50hz.parquet"
        output_prefix = "pediatric"

    input_path = args.input if args.input else default_input

    # Compute stats
    mean, std = compute_normalization_stats(input_path, sample_limit=args.sample_limit)

    print(f"\n{'='*60}")
    print(f"{args.dataset.upper()} NORMALIZATION STATISTICS")
    print(f"{'='*60}")
    print(f"Mean: {mean:.6f}")
    print(f"Std:  {std:.6f}")
    print(f"{'='*60}")

    # Save to numpy file
    output_path = f"{output_prefix}_mel128_stats.npy"
    np.save(output_path, {'mean': mean, 'std': std})
    print(f"\n✅ Saved to {output_path}")

    print(f"\nUsage in dataset loader:")
    print(f"  stats = np.load('{output_path}', allow_pickle=True).item()")
    print(f"  mean, std = stats['mean'], stats['std']")


if __name__ == "__main__":
    main()
