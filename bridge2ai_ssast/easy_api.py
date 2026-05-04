"""
High-level wrapper API for training and evaluating SSAST models on Bridge2AI datasets.

This module provides user-friendly functions for:
- Training medical condition classifiers with custom filtering (age, task, condition)
- Evaluating models on specific subgroups
- Running experiments with minimal boilerplate

Example usage:
    from bridge2ai_ssast.easy_api import train_condition_classifier, evaluate_model

    # Train on pediatric data, ages 7-12, asthma + allergies only
    model, metrics = train_condition_classifier(
        dataset='pediatric',
        conditions=['asthma', 'allergies'],
        age_range=(7, 12),
        tasks=['long-sounds'],  # sustained phonation
        epochs=30,
        output_dir='results/elementary_respiratory/'
    )

    # Evaluate on teenagers
    test_metrics = evaluate_model(
        model_path='results/elementary_respiratory/best_model.pth',
        dataset='pediatric',
        conditions=['asthma', 'allergies'],
        age_range=(13, 17),
        tasks=['long-sounds']
    )
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import numpy as np

# Add ssast to path
sys.path.insert(0, 'ssast/src')
from models.ast_models import ASTModel

# Import datasets
from bridge2ai_ssast.datasets import AdultBridge2AIDataset, PediatricBridge2AIDataset


def load_condition_mapping(mapping_path='condition_mapping.json'):
    """Load condition mapping from JSON file"""
    with open(mapping_path, 'r') as f:
        data = json.load(f)
    return data['adult_to_pediatric_mapping'], data['label_order']


def filter_conditions(all_mappings, condition_subset: Optional[List[str]]):
    """
    Filter condition mapping to a subset of conditions.

    Args:
        all_mappings: Full condition mapping dict
        condition_subset: List of condition names to include (e.g., ['asthma', 'adhd'])
                         If None, uses all conditions

    Returns:
        Filtered mapping dict
    """
    if condition_subset is None:
        return all_mappings

    filtered = {}
    for cond in condition_subset:
        if cond not in all_mappings:
            raise ValueError(
                f"Condition '{cond}' not found in mapping. "
                f"Available conditions: {list(all_mappings.keys())}"
            )
        filtered[cond] = all_mappings[cond]

    return filtered


def train_condition_classifier(
    dataset: str = 'pediatric',
    conditions: Optional[List[str]] = None,
    age_range: Optional[Tuple[int, int]] = None,
    tasks: Optional[List[str]] = None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    output_dir: str = 'results/',
    validation_split: float = 0.2,
    random_seed: int = 42,
    device: str = 'auto',
    pretrained_checkpoint: Optional[str] = None,
) -> Tuple[ASTModel, Dict]:
    """
    Train an SSAST medical condition classifier with custom filtering.

    Args:
        dataset: 'pediatric' or 'adult'
        conditions: List of conditions to predict (e.g., ['asthma', 'adhd'])
                   If None, uses all 8 overlapping conditions
        age_range: Tuple (min_age, max_age) for age filtering
                   Examples: (7, 12) for elementary school, (30, 50) for middle-aged adults
        tasks: List of acoustic tasks to include (e.g., ['long-sounds', 'passage'])
               If None, uses all tasks
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate (5e-5 recommended for fine-tuning)
        output_dir: Directory to save model checkpoints and metrics
        validation_split: Fraction of data for validation (0.0-1.0)
        random_seed: Random seed for reproducibility
        device: 'cuda', 'cpu', or 'auto' (auto-detect GPU)
        pretrained_checkpoint: Path to pretrained checkpoint (e.g., from adult training)
                              If None, loads SSAST-Tiny-Frame-400 pretrained weights

    Returns:
        (model, metrics): Trained model and dict of training metrics

    Example:
        # Train on elementary school kids with asthma or allergies
        model, metrics = train_condition_classifier(
            dataset='pediatric',
            conditions=['asthma', 'allergies'],
            age_range=(7, 12),
            epochs=30,
            output_dir='results/elementary_respiratory/'
        )
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load condition mapping
    all_mappings, label_order = load_condition_mapping()
    filtered_mappings = filter_conditions(all_mappings, conditions)

    # Extract condition mappings for dataset type
    if dataset == 'pediatric':
        condition_cols = {cond: mapping['pediatric_column']
                         for cond, mapping in filtered_mappings.items()}
        mel_path = 'features/mel_128bin_50hz.parquet'
        phenotype_path = 'phenotype/pediatric/pediatric_conditions_expanded.tsv'
        demographics_path = 'phenotype/pediatric/pediatric_demographics.tsv'

        # Load full dataset
        full_dataset = PediatricBridge2AIDataset(
            mel_parquet_path=mel_path,
            conditions_expanded_path=phenotype_path,
            condition_mapping=condition_cols,
            tasks=tasks,
            age_range=age_range,
            demographics_path=demographics_path,
            normalize=False,  # Will compute stats below
        )

        # Participant-level stratified split
        participant_ids = full_dataset.get_participant_ids()
        train_pids, val_pids = train_test_split(
            participant_ids,
            test_size=validation_split,
            random_state=random_seed
        )

        # Create train/val datasets
        train_dataset = PediatricBridge2AIDataset(
            mel_parquet_path=mel_path,
            conditions_expanded_path=phenotype_path,
            condition_mapping=condition_cols,
            participant_ids=train_pids,
            tasks=tasks,
            age_range=age_range,
            demographics_path=demographics_path,
            normalize=False,
        )

        val_dataset = PediatricBridge2AIDataset(
            mel_parquet_path=mel_path,
            conditions_expanded_path=phenotype_path,
            condition_mapping=condition_cols,
            participant_ids=val_pids,
            tasks=tasks,
            age_range=age_range,
            demographics_path=demographics_path,
            normalize=False,
        )

    elif dataset == 'adult':
        condition_cols = {cond: mapping['adult_column']
                         for cond, mapping in filtered_mappings.items()}
        mel_path = 'adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/mel_128bin_50hz.parquet'
        phenotype_path = 'adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/phenotype.tsv'

        # Load and split
        full_dataset = AdultBridge2AIDataset(
            mel_parquet_path=mel_path,
            phenotype_path=phenotype_path,
            condition_mapping=condition_cols,
            tasks=tasks,
            age_range=age_range,
            normalize=False,
        )

        # Random split (adult dataset doesn't have participant-level splitting concern)
        train_size = int((1 - validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

    else:
        raise ValueError(f"dataset must be 'pediatric' or 'adult', got '{dataset}'")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_conditions = len(filtered_mappings)
    model = ASTModel(
        label_dim=num_conditions,
        fshape=128,
        tshape=2,
        fstride=128,
        tstride=1,
        input_fdim=128,
        input_tdim=200,
        model_size='tiny',
        pretrain_stage=False,
        load_pretrained_mdl_path=pretrained_checkpoint or 'ssast/pretrained_model/SSAST-Tiny-Frame-400.pth'
    )
    model = model.to(device)

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\n{'='*80}")
    print(f"Training Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Conditions: {list(filtered_mappings.keys())}")
    print(f"  Age range: {age_range if age_range else 'All ages'}")
    print(f"  Tasks: {tasks if tasks else 'All tasks'}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")

    best_val_f1 = 0.0
    metrics_history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)

            outputs = model(mels, task='ft_avgtok')
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for mels, labels in val_loader:
                mels, labels = mels.to(device), labels.to(device)

                outputs = model(mels, task='ft_avgtok')
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Metrics
        val_f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='macro', zero_division=0)

        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_f1': val_f1,
        })

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_path / 'best_model.pth')
            print(f"  ✓ Saved new best model (F1: {val_f1:.4f})")

    # Save final model
    torch.save(model.state_dict(), output_path / 'final_model.pth')

    # Save metrics
    pd.DataFrame(metrics_history).to_csv(output_path / 'training_log.csv', index=False)

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"  Best validation F1: {best_val_f1:.4f}")
    print(f"  Models saved to: {output_path}")
    print(f"{'='*80}\n")

    return model, {'best_val_f1': best_val_f1, 'history': metrics_history}


def evaluate_model(
    model_path: str,
    dataset: str = 'pediatric',
    conditions: Optional[List[str]] = None,
    age_range: Optional[Tuple[int, int]] = None,
    tasks: Optional[List[str]] = None,
    batch_size: int = 32,
    device: str = 'auto',
) -> Dict:
    """
    Evaluate a trained model on a specific subset of data.

    Args:
        model_path: Path to saved model checkpoint (.pth file)
        dataset: 'pediatric' or 'adult'
        conditions: List of conditions (must match training conditions)
        age_range: Age filter for evaluation
        tasks: Task filter for evaluation
        batch_size: Batch size for evaluation
        device: 'cuda', 'cpu', or 'auto'

    Returns:
        Dict of evaluation metrics (F1, AUROC, AUPRC per condition + macro averages)

    Example:
        # Evaluate model trained on ages 7-12, test on ages 13-17
        metrics = evaluate_model(
            model_path='results/elementary_respiratory/best_model.pth',
            dataset='pediatric',
            conditions=['asthma', 'allergies'],
            age_range=(13, 17),
            tasks=['long-sounds']
        )
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load condition mapping
    all_mappings, _ = load_condition_mapping()
    filtered_mappings = filter_conditions(all_mappings, conditions)

    # Load dataset (similar to training, but no split)
    if dataset == 'pediatric':
        condition_cols = {cond: mapping['pediatric_column']
                         for cond, mapping in filtered_mappings.items()}
        mel_path = 'features/mel_128bin_50hz.parquet'
        phenotype_path = 'phenotype/pediatric/pediatric_conditions_expanded.tsv'
        demographics_path = 'phenotype/pediatric/pediatric_demographics.tsv'

        test_dataset = PediatricBridge2AIDataset(
            mel_parquet_path=mel_path,
            conditions_expanded_path=phenotype_path,
            condition_mapping=condition_cols,
            tasks=tasks,
            age_range=age_range,
            demographics_path=demographics_path,
            normalize=False,
        )
    elif dataset == 'adult':
        condition_cols = {cond: mapping['adult_column']
                         for cond, mapping in filtered_mappings.items()}
        mel_path = 'adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/mel_128bin_50hz.parquet'
        phenotype_path = 'adult/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-2.0.1 2/phenotype.tsv'

        test_dataset = AdultBridge2AIDataset(
            mel_parquet_path=mel_path,
            phenotype_path=phenotype_path,
            condition_mapping=condition_cols,
            tasks=tasks,
            age_range=age_range,
            normalize=False,
        )
    else:
        raise ValueError(f"dataset must be 'pediatric' or 'adult', got '{dataset}'")

    # Load model
    num_conditions = len(filtered_mappings)
    model = ASTModel(
        label_dim=num_conditions,
        fshape=128,
        tshape=2,
        fstride=128,
        tstride=1,
        input_fdim=128,
        input_tdim=200,
        model_size='tiny',
        pretrain_stage=False,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluate
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mels, labels in test_loader:
            mels = mels.to(device)
            outputs = model(mels, task='ft_avgtok')
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute metrics per condition
    condition_names = list(filtered_mappings.keys())
    metrics = {'condition_metrics': {}}

    for i, cond_name in enumerate(condition_names):
        cond_labels = all_labels[:, i]
        cond_preds = all_preds[:, i]

        f1 = f1_score(cond_labels, (cond_preds > 0.5).astype(int), zero_division=0)
        auroc = roc_auc_score(cond_labels, cond_preds) if cond_labels.sum() > 0 else 0.0
        auprc = average_precision_score(cond_labels, cond_preds) if cond_labels.sum() > 0 else 0.0

        metrics['condition_metrics'][cond_name] = {
            'f1': f1,
            'auroc': auroc,
            'auprc': auprc,
            'prevalence': cond_labels.mean(),
            'positive_count': int(cond_labels.sum()),
        }

    # Macro averages
    metrics['macro_f1'] = np.mean([m['f1'] for m in metrics['condition_metrics'].values()])
    metrics['macro_auroc'] = np.mean([m['auroc'] for m in metrics['condition_metrics'].values()])
    metrics['macro_auprc'] = np.mean([m['auprc'] for m in metrics['condition_metrics'].values()])

    print(f"\n{'='*80}")
    print(f"Evaluation Results ({dataset}, age {age_range if age_range else 'all'})")
    print(f"{'='*80}")
    for cond, m in metrics['condition_metrics'].items():
        print(f"{cond:20s} | F1: {m['f1']:.3f} | AUROC: {m['auroc']:.3f} | AUPRC: {m['auprc']:.3f} | Prev: {m['prevalence']:.1%}")
    print(f"{'='*80}")
    print(f"Macro Average       | F1: {metrics['macro_f1']:.3f} | AUROC: {metrics['macro_auroc']:.3f} | AUPRC: {metrics['macro_auprc']:.3f}")
    print(f"{'='*80}\n")

    return metrics
