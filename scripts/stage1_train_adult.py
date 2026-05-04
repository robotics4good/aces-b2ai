"""
Stage 1: Fine-tune SSAST on adult dataset for medical condition classification.

Three-stage transfer learning:
    SSAST-Tiny-Frame-400 (pretrained) → Adult supervised → Pediatric supervised
                                        ^^^^^^^^^^^^^^^^
                                        THIS SCRIPT

Usage:
    python scripts/stage1_train_adult.py \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-4 \
        --output-dir results/stage1_adult

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
from bridge2ai_ssast.datasets import AdultBridge2AIDataset


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
    parser = argparse.ArgumentParser(description="Stage 1: Train SSAST on adult dataset")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='results/stage1_adult', help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--condition-mapping-file', type=str, help='Path to condition mapping JSON (optional)')
    parser.add_argument('--age-min', type=int, default=None, help='Minimum age for filtering (optional)')
    parser.add_argument('--age-max', type=int, default=None, help='Maximum age for filtering (optional)')

    args = parser.parse_args()

    # Parse age range
    age_range = None
    if args.age_min is not None and args.age_max is not None:
        age_range = (args.age_min, args.age_max)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load normalization stats
    print("Loading normalization stats...")
    stats = np.load('adult_mel128_stats.npy', allow_pickle=True).item()
    mean, std = stats['mean'], stats['std']

    # Example condition mapping (TODO: load from file if provided)
    condition_mapping = {
        'asthma': 'asthma',
        'allergies': 'seasonal_allergies',
        'hearing_loss': 'hearing_loss',
        'voice_disorder': 'voice_disorder',
        'neurological': 'neurological_disorder',
    }

    # Load full dataset
    print("Loading adult dataset...")
    full_dataset = AdultBridge2AIDataset(
        mel_parquet_path='adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/mel_128bin_50hz.parquet',
        phenotype_path='adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/phenotype.tsv',
        condition_mapping=condition_mapping,
        target_length=200,
        normalize=True,
        mean=mean,
        std=std,
        age_range=age_range,
    )

    # Train/val split
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=args.val_split, random_state=42
    )

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Train: {len(train_dataset)} recordings")
    print(f"Val: {len(val_dataset)} recordings")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    print("Initializing SSAST-Tiny-Frame-400...")
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
        load_pretrained_mdl_path='ssast/pretrained_model/SSAST-Tiny-Frame-400.pth'
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
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

    # Save training log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / 'training_log.csv', index=False)
    print(f"✅ Training log saved to {output_dir / 'training_log.csv'}")

    # Plot curves
    plot_training_curves(log_df, output_dir)

    print("\n" + "="*60)
    print(f"STAGE 1 TRAINING COMPLETE")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Results saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
