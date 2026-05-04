"""
Gradient verification on small subsample to validate architecture and training logic.

Trains on 50 samples for 10 iterations and plots:
1. Loss curve (should decrease)
2. Gradient norms (should be stable, typically 0.1-10)

This catches bugs early before expensive full training runs.

Usage:
    python scripts/sanity_check_gradients.py --dataset adult
    python scripts/sanity_check_gradients.py --dataset pediatric
"""

import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# Add ssast to path (assumes ssast folder is in repo root)
sys.path.insert(0, 'ssast/src')
from models.ast_models import ASTModel

# Add bridge2ai_ssast to path
sys.path.insert(0, '.')
from bridge2ai_ssast.datasets import AdultBridge2AIDataset, PediatricBridge2AIDataset


def run_gradient_verification(dataset, dataset_type):
    """
    Run gradient verification on small subsample.

    Args:
        dataset: PyTorch Dataset instance
        dataset_type: 'adult' or 'pediatric'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model with pretrained weights
    model = ASTModel(
        label_dim=8,  # 8 overlapping conditions
        fshape=128,   # full frequency axis (frame-based)
        tshape=2,
        fstride=128,  # no overlap
        tstride=1,    # frame-level
        input_fdim=128,
        input_tdim=200,
        model_size='tiny',
        pretrain_stage=False,
        load_pretrained_mdl_path='ssast/pretrained_model/SSAST-Tiny-Frame-400.pth'
    )
    model = model.to(device)

    # Use only 50 samples from dataset
    small_dataset = Subset(dataset, range(min(50, len(dataset))))
    small_loader = DataLoader(small_dataset, batch_size=10, shuffle=True)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Track metrics
    losses = []
    gradient_norms = []

    print("\n" + "="*60)
    print(f"Running gradient verification on {len(small_dataset)}-sample subsample")
    print("="*60)

    # Train for 10 iterations
    model.train()
    for i, (mels, labels) in enumerate(small_loader):
        if i >= 10:
            break

        mels, labels = mels.to(device), labels.to(device)

        # Forward pass
        outputs = model(mels, task='ft_avgtok')
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Track gradient norm across all parameters
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        gradient_norms.append(total_norm)
        losses.append(loss.item())

        optimizer.step()

        print(f"Iter {i+1}/10: Loss={loss.item():.4f}, Gradient Norm={total_norm:.4f}")

    print("="*60)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(losses, marker='o', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss (BCEWithLogits)', fontsize=12)
    ax1.set_title(f'Loss Over 10 Iterations ({len(small_dataset)} Samples)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Gradient norm
    ax2.plot(gradient_norms, marker='o', linewidth=2, color='orange')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Gradient Norm (L2)', fontsize=12)
    ax2.set_title('Gradient Flow Verification', fontsize=14)
    ax2.axhline(y=0.1, color='red', linestyle='--', label='Min threshold (0.1)')
    ax2.axhline(y=10, color='red', linestyle='--', label='Max threshold (10)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f'gradient_verification_{dataset_type}.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Plot saved to {output_path}")

    # Verification checks
    print("\n" + "="*60)
    print("VERIFICATION RESULTS:")
    print("="*60)

    loss_decreasing = losses[-1] < losses[0]
    grad_stable = all(0.01 < g < 100 for g in gradient_norms)
    no_nan = not any(np.isnan(losses)) and not any(np.isnan(gradient_norms))

    print(f"✅ Loss decreasing: {loss_decreasing} (start={losses[0]:.4f}, end={losses[-1]:.4f})")
    print(f"✅ Gradients stable (0.01-100): {grad_stable} (mean={np.mean(gradient_norms):.4f})")
    print(f"✅ No NaN/Inf values: {no_nan}")

    if not loss_decreasing:
        print("\n⚠️  WARNING: Loss not decreasing!")
        print("    Possible causes:")
        print("    - Learning rate too low")
        print("    - Data normalization incorrect")
        print("    - Labels incorrectly encoded")

    if not grad_stable:
        print("\n⚠️  WARNING: Gradient instability detected!")
        if np.mean(gradient_norms) < 0.1:
            print("    Issue: Vanishing gradients")
            print("    Solutions: Check normalization stats, layer initialization")
        else:
            print("    Issue: Exploding gradients")
            print("    Solutions: Reduce learning rate, add gradient clipping")

    if no_nan and loss_decreasing and grad_stable:
        print("\n🎉 All checks passed! Ready for full training.")
    else:
        print("\n❌ Some checks failed. Debug before full training.")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Gradient verification on small subsample")
    parser.add_argument(
        "--dataset",
        choices=["adult", "pediatric"],
        required=True,
        help="Which dataset to verify"
    )

    args = parser.parse_args()

    # Load appropriate dataset
    if args.dataset == "adult":
        print("Loading adult dataset...")

        # Load normalization stats
        stats = np.load('adult_mel128_stats.npy', allow_pickle=True).item()
        mean, std = stats['mean'], stats['std']

        # Example condition mapping (update with actual column names)
        condition_mapping = {
            'asthma': 'asthma',
            'allergies': 'seasonal_allergies',
            'hearing_loss': 'hearing_loss',
            'voice_disorder': 'voice_disorder',
            'neurological': 'neurological_disorder',
        }

        dataset = AdultBridge2AIDataset(
            mel_parquet_path='adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/mel_128bin_50hz.parquet',
            phenotype_path='adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/phenotype.tsv',
            condition_mapping=condition_mapping,
            target_length=200,
            normalize=True,
            mean=mean,
            std=std,
        )

    else:  # pediatric
        print("Loading pediatric dataset...")

        # Load normalization stats
        stats = np.load('pediatric_mel128_stats.npy', allow_pickle=True).item()
        mean, std = stats['mean'], stats['std']

        # Example condition mapping (update after running parse_pediatric_conditions.py)
        condition_mapping = {
            'asthma': 'has_asthma',
            'allergies': 'had_allergies',
            'hearing_loss': 'has_hearing_loss',
            'voice_disorder': 'has_voice_disorder',
            'neurological': 'has_neurological_disorder',
        }

        dataset = PediatricBridge2AIDataset(
            mel_parquet_path='features/mel_128bin_50hz.parquet',
            conditions_expanded_path='phenotype/pediatric/pediatric_conditions_expanded.tsv',
            condition_mapping=condition_mapping,
            target_length=200,
            normalize=True,
            mean=mean,
            std=std,
        )

    print(f"Dataset loaded: {len(dataset)} recordings")

    # Run verification
    run_gradient_verification(dataset, args.dataset)


if __name__ == "__main__":
    main()
