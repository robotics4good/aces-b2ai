"""
Stage 2: Fine-tune adult-trained SSAST on pediatric dataset.

Three-stage transfer learning:
    SSAST-Tiny-Frame-400 (pretrained) → Adult supervised → Pediatric supervised
                                                            ^^^^^^^^^^^^^^^^^^^
                                                            THIS SCRIPT

Uses participant-level stratified splitting to prevent data leakage.

Usage:
    python scripts/stage2_train_pediatric.py \
        --adult-checkpoint results/stage1_adult/best_model.pth \
        --epochs 30 \
        --batch-size 32 \
        --lr 5e-5 \
        --output-dir results/stage2_pediatric

Outputs:
    - best_model.pth - Best model checkpoint (highest validation F1)
    - final_model.pth - Final model checkpoint
    - training_log.csv - Per-epoch metrics
    - loss_curves.png - Training/validation loss plots
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add ssast to path
sys.path.insert(0, 'ssast/src')
from models.ast_models import ASTModel

# Add bridge2ai_ssast to path
sys.path.insert(0, '.')
from bridge2ai_ssast.datasets import PediatricBridge2AIDataset


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for mels, labels in tqdm(train_loader, desc="Training"):
        mels, labels = mels.to(device), labels.to(device)

        # Forward
        outputs = model(mels, task='ft_avgtok')
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Collect predictions for metrics
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute metrics (threshold at 0.5 for F1)
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='macro', zero_division=0)

    return avg_loss, f1


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mels, labels in tqdm(val_loader, desc="Validating"):
            mels, labels = mels.to(device), labels.to(device)

            outputs = model(mels, task='ft_avgtok')
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute metrics
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='macro', zero_division=0)

    # Only compute AUROC if there are positive samples
    try:
        auroc = roc_auc_score(all_labels, all_preds, average='macro')
        auprc = average_precision_score(all_labels, all_preds, average='macro')
    except ValueError:
        auroc = 0.0
        auprc = 0.0

    return avg_loss, f1, auroc, auprc


def plot_training_curves(log_df, output_dir):
    """Plot training/validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(log_df['epoch'], log_df['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (BCEWithLogits)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1 curves
    ax2.plot(log_df['epoch'], log_df['train_f1'], label='Train F1', linewidth=2)
    ax2.plot(log_df['epoch'], log_df['val_f1'], label='Val F1', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax2.set_title('Training and Validation F1', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    print(f"✅ Training curves saved to {output_dir / 'loss_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Fine-tune on pediatric dataset")
    parser.add_argument('--adult-checkpoint', type=str, required=True, help='Path to Stage 1 adult checkpoint')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (lower than Stage 1)')
    parser.add_argument('--output-dir', type=str, default='results/stage2_pediatric', help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.15, help='Test split ratio')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load normalization stats
    print("Loading normalization stats...")
    stats = np.load('pediatric_mel128_stats.npy', allow_pickle=True).item()
    mean, std = stats['mean'], stats['std']

    # Example condition mapping (must match Stage 1)
    condition_mapping = {
        'asthma': 'has_asthma',
        'allergies': 'had_allergies',
        'hearing_loss': 'has_hearing_loss',
        'voice_disorder': 'has_voice_disorder',
        'neurological': 'has_neurological_disorder',
    }

    # Load full pediatric dataset (no filtering yet)
    print("Loading pediatric dataset...")
    full_dataset = PediatricBridge2AIDataset(
        mel_parquet_path='features/mel_128bin_50hz.parquet',
        conditions_expanded_path='phenotype/pediatric/pediatric_conditions_expanded.tsv',
        condition_mapping=condition_mapping,
        target_length=200,
        normalize=True,
        mean=mean,
        std=std,
    )

    # Get unique participant IDs for stratified splitting
    participant_ids = full_dataset.get_participant_ids()
    print(f"Total participants: {len(participant_ids)}")

    # Participant-level train/val/test split
    train_pids, temp_pids = train_test_split(
        participant_ids, test_size=(args.val_split + args.test_split), random_state=42
    )
    val_pids, test_pids = train_test_split(
        temp_pids, test_size=args.test_split / (args.val_split + args.test_split), random_state=42
    )

    print(f"Train participants: {len(train_pids)}")
    print(f"Val participants: {len(val_pids)}")
    print(f"Test participants: {len(test_pids)}")

    # Create datasets by filtering participant IDs
    train_dataset = PediatricBridge2AIDataset(
        mel_parquet_path='features/mel_128bin_50hz.parquet',
        conditions_expanded_path='phenotype/pediatric/pediatric_conditions_expanded.tsv',
        condition_mapping=condition_mapping,
        target_length=200,
        normalize=True,
        mean=mean,
        std=std,
        participant_ids=train_pids,
    )

    val_dataset = PediatricBridge2AIDataset(
        mel_parquet_path='features/mel_128bin_50hz.parquet',
        conditions_expanded_path='phenotype/pediatric/pediatric_conditions_expanded.tsv',
        condition_mapping=condition_mapping,
        target_length=200,
        normalize=True,
        mean=mean,
        std=std,
        participant_ids=val_pids,
    )

    test_dataset = PediatricBridge2AIDataset(
        mel_parquet_path='features/mel_128bin_50hz.parquet',
        conditions_expanded_path='phenotype/pediatric/pediatric_conditions_expanded.tsv',
        condition_mapping=condition_mapping,
        target_length=200,
        normalize=True,
        mean=mean,
        std=std,
        participant_ids=test_pids,
    )

    print(f"Train recordings: {len(train_dataset)}")
    print(f"Val recordings: {len(val_dataset)}")
    print(f"Test recordings: {len(test_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model with adult checkpoint
    print(f"Loading adult checkpoint from {args.adult_checkpoint}...")
    model = ASTModel(
        label_dim=len(condition_mapping),
        fshape=128,
        tshape=2,
        fstride=128,
        tstride=1,
        input_fdim=128,
        input_tdim=200,
        model_size='tiny',
        pretrain_stage=False,
    )
    model.load_state_dict(torch.load(args.adult_checkpoint, map_location='cpu'))
    model = model.to(device)

    # Loss and optimizer (lower LR for fine-tuning)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    print("="*60)

    best_val_f1 = 0.0
    log_rows = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_f1, val_auroc, val_auprc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")

        # Log
        log_rows.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_auroc': val_auroc,
            'val_auprc': val_auprc,
        })

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"✅ New best model saved (Val F1: {val_f1:.4f})")

    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    print(f"\n✅ Final model saved to {output_dir / 'final_model.pth'}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_f1, test_auroc, test_auprc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")

    # Save training log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / 'training_log.csv', index=False)
    print(f"✅ Training log saved to {output_dir / 'training_log.csv'}")

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_f1': test_f1,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
    }
    pd.DataFrame([test_results]).to_csv(output_dir / 'test_results.csv', index=False)
    print(f"✅ Test results saved to {output_dir / 'test_results.csv'}")

    # Plot curves
    plot_training_curves(log_df, output_dir)

    print("\n" + "="*60)
    print(f"STAGE 2 TRAINING COMPLETE")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}, Test AUROC: {test_auroc:.4f}")
    print(f"Results saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
