#!/usr/bin/env python3
"""Binary classifier training script — MLP / TCN / MelHuBERT.

Feature modes
-------------
  mlp       — 132 OpenSMILE scalar features → regularised MLP
  tcn       — 12-channel EMA time-series → dilated TCN
  melhubert — 60-bin mel spectrogram (zero-padded to 80) →
              MelHuBERT backbone (frozen encoder) + optional
              sparse scalar cross-attention conditioning

Usage examples
--------------
MelHuBERT, hearing cochlear vs healthy, all tasks::

    python scripts/train_binary_classifier.py \\
        --target hearing_cochlear \\
        --negative-class healthy_only \\
        --feature-mode melhubert \\
        --ckpt-path weights/melhubert-20ms-stg2.ckpt \\
        --exclude-ill \\
        --epochs 10 \\
        --log-every 2

MelHuBERT with sparse scalar conditioning (cross-attention)::

    python scripts/train_binary_classifier.py \\
        --target hearing_cochlear \\
        --feature-mode melhubert \\
        --ckpt-path weights/melhubert-20ms-stg2.ckpt \\
        --sparse-keys f0_mean mean_hnr_db speaking_rate \\
        --unfreeze-after 5 \\
        --epochs 15

MLP quick baseline::

    python scripts/train_binary_classifier.py \\
        --target hearing_cochlear --negative-class healthy_only \\
        --feature-mode mlp --exclude-ill --epochs 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aces_b2ai.models.branches.timeseries_branch import TimeSeriesBranch      # noqa: E402
from aces_b2ai.models.branches.transformer_branch import TransformerBranch    # noqa: E402
from aces_b2ai.models.model_config import TransformerBranchConfig             # noqa: E402

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
from build_training_manifest import build_manifest  # type: ignore  # noqa: E402

DEFAULT_DATASET_DIR = Path("~/Downloads/bridge2ai-voice-pediatric-dataset-1.0.0").expanduser()
DEFAULT_CKPT        = Path("weights/melhubert-20ms-stg2.ckpt")


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_static_features(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "features" / "static_features.tsv"
    df = pd.read_csv(path, sep="\t")
    return df.drop(columns=["session_id", "transcription"], errors="ignore")


def load_ema_features(dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(dataset_dir / "features" / "sparc_ema.parquet")
    df["ema_2d"] = df["ema"].apply(lambda a: np.stack(a).astype(np.float32))
    return df[["participant_id", "task_name", "ema_2d"]]


def load_mel_features(dataset_dir: Path, target_bins: int = 80) -> pd.DataFrame:
    """Load mel spectrograms, returning (bins, T) float32 arrays.

    The parquet stores (60_bins, T) — zero-pad to ``target_bins`` (80) so the
    pretrained MelHuBERT pre_extract_proj (80→768) loads cleanly.
    """
    df = pd.read_parquet(dataset_dir / "features" / "torchaudio_mel_spectrogram.parquet")

    def _to_mel(arr: np.ndarray) -> np.ndarray:
        # arr is object array of freq_bin arrays: shape (n_bins, T)
        mel = np.stack(arr).astype(np.float32)  # (n_bins, T)
        if mel.shape[0] < target_bins:
            pad = np.zeros((target_bins - mel.shape[0], mel.shape[1]), dtype=np.float32)
            mel = np.concatenate([mel, pad], axis=0)
        return mel  # (target_bins, T)

    df["mel_2d"] = df["mel_spectrogram"].apply(_to_mel)
    return df[["participant_id", "task_name", "mel_2d", "n_frames"]]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class StaticDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return torch.tensor(r["features"], dtype=torch.float32), \
               torch.tensor(r["label"],    dtype=torch.float32)


class EMADataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return torch.tensor(r["ema_2d"].T, dtype=torch.float32), \
               torch.tensor(r["label"], dtype=torch.float32)


class MelDataset(Dataset):
    """Returns (mel, sparse_vector, label).

    mel          : (80, T) float32
    sparse_vector: (D,)    float32   — utterance-level scalars for cross-attn
    label        : scalar  float32
    """

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        mel = torch.tensor(r["mel"], dtype=torch.float32)           # (80, T)
        sv  = torch.tensor(r["sparse_vector"], dtype=torch.float32) # (D,)
        lbl = torch.tensor(r["label"], dtype=torch.float32)
        return mel, sv, lbl


def _pad_collate_2d(batch):
    """Pad (C, T_i) tensors to (C, T_max)."""
    xs, ys = zip(*batch)
    t_max = max(x.size(1) for x in xs)
    padded = [
        torch.cat([x, torch.zeros(x.size(0), t_max - x.size(1))], dim=1)
        if x.size(1) < t_max else x
        for x in xs
    ]
    return torch.stack(padded), torch.stack(ys)


def mel_collate(batch):
    """Pad mel tensors; sparse_vector is fixed-width."""
    mels, svecs, lbls = zip(*batch)
    t_max = max(m.size(1) for m in mels)
    padded_mels = [
        torch.cat([m, torch.zeros(m.size(0), t_max - m.size(1))], dim=1)
        if m.size(1) < t_max else m
        for m in mels
    ]
    # pad_mask: 1 = valid, 0 = padded
    pad_masks = [
        torch.cat([torch.ones(m.size(1)), torch.zeros(t_max - m.size(1))])
        for m in mels
    ]
    return (
        torch.stack(padded_mels),   # (B, 80, T_max)
        torch.stack(pad_masks),     # (B, T_max)
        torch.stack(svecs),         # (B, D)
        torch.stack(lbls),          # (B,)
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BinaryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def l1_penalty(self):
        return self.net[0].weight.abs().mean()


class BinaryTCN(nn.Module):
    def __init__(self, n_channels=12, embed_dim=128, n_blocks=4, dropout=0.4) -> None:
        super().__init__()
        self.encoder = TimeSeriesBranch(n_channels, embed_dim, n_blocks, dropout)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)

    def l1_penalty(self):
        return torch.tensor(0.0)


class BinaryMelHuBERT(nn.Module):
    """MelHuBERT backbone + optional sparse conditioning + binary head.

    Parameters
    ----------
    cfg : TransformerBranchConfig
    sparse_dim : int  — dimension of the sparse_vector fed to cross-attention.
                        0 = no cross-attention (sparse_cond disabled).
    """

    def __init__(self, cfg: TransformerBranchConfig) -> None:
        super().__init__()
        self.branch = TransformerBranch(cfg)
        self.head   = nn.Linear(cfg.embed_dim, 1)

    def forward(
        self,
        mel: torch.Tensor,
        pad_mask: torch.Tensor | None,
        sparse_vector: torch.Tensor | None,
    ) -> torch.Tensor:
        embed = self.branch(
            mel=mel, pad_mask=pad_mask,
            sparc_series=None, sparse_vector=sparse_vector,
        )
        return self.head(embed).squeeze(-1)

    def l1_penalty(self):
        return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    lbls  = labels.cpu().numpy().astype(int)

    n_pos, n_neg = int(lbls.sum()), len(lbls) - int(lbls.sum())
    tp = int(((preds == 1) & (lbls == 1)).sum())
    tn = int(((preds == 0) & (lbls == 0)).sum())
    fp = int(((preds == 1) & (lbls == 0)).sum())
    fn = int(((preds == 0) & (lbls == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(lbls, probs)) if n_pos > 0 and n_neg > 0 else float("nan")
    except Exception:
        auc = float("nan")

    pos_probs = probs[lbls == 1] if n_pos > 0 else np.array([0.5])
    neg_probs = probs[lbls == 0] if n_neg > 0 else np.array([0.5])

    return {
        "acc": float((preds == lbls).mean()), "auc": auc, "f1": f1,
        "prec": prec, "rec": rec,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_pos": n_pos, "n_neg": n_neg,
        "mean_prob_pos": float(pos_probs.mean()),
        "mean_prob_neg": float(neg_probs.mean()),
        "logit_mean": float(logits.mean()), "logit_std": float(logits.std()),
        "dead_neg": preds.sum() == 0,
        "dead_pos": preds.sum() == len(preds),
    }


def grad_norm(model: nn.Module) -> float:
    total = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    )
    return total ** 0.5


# ---------------------------------------------------------------------------
# MelHuBERT-specific diagnostics
# ---------------------------------------------------------------------------

def melhubert_diagnostics(model: BinaryMelHuBERT, epoch: int, log_layers: bool = True) -> None:
    branch = model.branch
    print("\n  ┌─ MelHuBERT Diagnostics ─────────────────────────────────────────")

    # Encoder frozen status
    enc_trainable = sum(p.requires_grad for p in branch.backbone.parameters())
    enc_total     = sum(1 for _ in branch.backbone.parameters())
    frozen = enc_trainable == 0
    print(f"  │ Encoder: {'FROZEN' if frozen else 'UNFROZEN'}  "
          f"({enc_trainable}/{enc_total} params require grad)")

    # input_proj
    if hasattr(branch.input_proj, "weight"):
        ip_w = branch.input_proj.weight
        ip_gnorm = ip_w.grad.norm().item() if ip_w.grad is not None else 0.0
        print(f"  │ input_proj:    weight_norm={ip_w.norm():.4f}  "
              f"grad_norm={ip_gnorm:.4f}")
    else:
        print("  │ input_proj:    Identity (no-op)")

    # Sparse conditioning (FiLM: scale + shift)
    if branch.sparse_cond is not None:
        sc = branch.sparse_cond
        sc_gnorm = sum(
            p.grad.norm().item() ** 2
            for p in sc.parameters() if p.grad is not None
        ) ** 0.5
        scale_wnorm = sc.scale_proj.weight.norm().item()
        shift_wnorm = sc.shift_proj.weight.norm().item()
        norm_gnorm = sc.norm.weight.grad.norm().item() if sc.norm.weight.grad is not None else 0.0
        print(f"  │ sparse_cond(FiLM): grad={sc_gnorm:.4f}  "
              f"scale_w={scale_wnorm:.4f}  shift_w={shift_wnorm:.4f}  norm_grad={norm_gnorm:.4f}")
        if sc_gnorm == 0.0:
            print("  │   [!] FiLM conditioning has zero gradients — "
                  "check sparse_vector is not all zeros")
    else:
        print("  │ sparse_cond:   disabled (no --sparse-keys given)")

    # Head
    head_gnorm = model.head.weight.grad.norm().item() if model.head.weight.grad is not None else 0.0
    print(f"  │ binary head:   weight_norm={model.head.weight.norm():.4f}  "
          f"grad_norm={head_gnorm:.4f}")

    # Per-encoder-layer gradient + weight norms
    if log_layers and hasattr(branch.backbone, "encoder") and \
            hasattr(branch.backbone.encoder, "layers"):
        print("  │")
        print("  │  Layer  WeightNorm  GradNorm   Status")
        print("  │  ─────  ──────────  ─────────  ──────")
        for i, layer in enumerate(branch.backbone.encoder.layers):
            params = list(layer.parameters())
            wnorm = sum(p.norm().item() ** 2 for p in params) ** 0.5
            gnorm_v = sum(
                p.grad.norm().item() ** 2 for p in params if p.grad is not None
            ) ** 0.5
            status = "grad ✓" if gnorm_v > 0 else "frozen" if not params[0].requires_grad else "no grad yet"
            print(f"  │  {i:>5}  {wnorm:>10.4f}  {gnorm_v:>9.4f}  {status}")

    # Layer weights (if weighted_sum pool)
    if branch.layer_weights is not None:
        w = torch.softmax(branch.layer_weights, dim=0).detach()
        top3 = w.topk(3).indices.tolist()
        print(f"  │ layer_weights: top-3 layers by weight: {top3}")

    print("  └────────────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch_mlp_tcn(model, loader, optimizer, pos_weight, device, l1_lambda):
    training = optimizer is not None
    model.train(training)
    total_loss, all_logits, all_labels = 0.0, [], []
    last_gnorm = 0.0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight.to(device))
            if l1_lambda > 0 and hasattr(model, "l1_penalty"):
                loss = loss + l1_lambda * model.l1_penalty()
            if training:
                optimizer.zero_grad(); loss.backward()
                last_gnorm = grad_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    all_logits_t = torch.cat(all_logits)
    all_labels_t = torch.cat(all_labels)
    return total_loss / len(all_labels_t), compute_metrics(all_logits_t, all_labels_t), last_gnorm


def run_epoch_melhubert(model: BinaryMelHuBERT, loader, optimizer, pos_weight, device):
    training = optimizer is not None
    model.train(training)
    total_loss, all_logits, all_labels = 0.0, [], []
    last_gnorm = 0.0

    with torch.set_grad_enabled(training):
        for mel, pad_mask, sv, y in loader:
            mel      = mel.to(device)
            pad_mask = pad_mask.to(device)
            sv_in    = sv.to(device) if sv.shape[-1] > 0 else None
            y        = y.to(device)

            logits = model(mel, pad_mask, sv_in)
            loss   = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight.to(device))

            if training:
                optimizer.zero_grad(); loss.backward()
                last_gnorm = grad_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    all_logits_t = torch.cat(all_logits)
    all_labels_t = torch.cat(all_labels)
    return total_loss / len(all_labels_t), compute_metrics(all_logits_t, all_labels_t), last_gnorm


# ---------------------------------------------------------------------------
# Helper printers
# ---------------------------------------------------------------------------

def print_confusion(m: dict, prefix: str = "") -> None:
    print(
        f"  {prefix}CM: TP={m['tp']:>3} TN={m['tn']:>3} FP={m['fp']:>3} FN={m['fn']:>3}  "
        f"prec={m['prec']:.3f}  rec={m['rec']:.3f}  "
        f"P(y=1|pos)={m['mean_prob_pos']:.3f}  P(y=1|neg)={m['mean_prob_neg']:.3f}"
    )
    if m.get("dead_neg"):
        print(f"  {prefix}[!] DEAD — all negative predictions")
    if m.get("dead_pos"):
        print(f"  {prefix}[!] DEAD — all positive predictions")


def print_feature_importance(model: BinaryMLP, feat_cols: list[str], top_n: int = 15) -> None:
    w = model.net[0].weight.detach().cpu().numpy()
    importance = np.linalg.norm(w, axis=0)
    top_idx = np.argsort(importance)[::-1][:top_n]
    print(f"\n  Top {top_n} features by first-layer weight magnitude:")
    for rank, idx in enumerate(top_idx, 1):
        name = feat_cols[idx] if idx < len(feat_cols) else f"feat_{idx}"
        print(f"    {rank:>2}. {name:<55}  {importance[idx]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: C901 (intentionally long)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Data
    parser.add_argument("--dataset-dir",    type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--target",         type=str,  required=True)
    parser.add_argument("--feature-mode",   choices=["mlp", "tcn", "melhubert"], default="mlp")
    parser.add_argument("--negative-class", choices=["all", "healthy_only"], default="all")
    parser.add_argument("--age-min",  type=float, default=None)
    parser.add_argument("--age-max",  type=float, default=None)
    parser.add_argument("--exclude-ill", action="store_true")
    parser.add_argument("--tasks",    nargs="*", default=None)
    # MelHuBERT-specific
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT,
                        help="Path to melhubert-20ms-stg2.ckpt")
    # Top acoustic scalar features from MLP importance analysis — used as
    # default sparse conditioning vector for the cross-attention layer.
    _DEFAULT_SPARSE_KEYS = [
        "HNRdBACF_sma3nz_stddevNorm",           # HNR variability
        "speaking_rate",                          # fluency
        "jitterLocal_sma3nz_stddevNorm",         # pitch perturbation variability
        "F0semitoneFrom27.5Hz_sma3nz_amean",     # mean F0
        "loudness_sma3_amean",                   # mean loudness
        "alphaRatioV_sma3nz_amean",              # spectral tilt (voiced)
        "VoicedSegmentsPerSec",                  # voicing density
    ]
    parser.add_argument(
        "--sparse-keys",
        nargs="*",
        default=None,
        help=(
            "Static feature keys for cross-attention conditioning. "
            "Defaults to 7 top acoustic features when --feature-mode melhubert is used. "
            "Pass --sparse-keys '' to disable cross-attention entirely. "
            "E.g. --sparse-keys f0_mean mean_hnr_db speaking_rate"
        ),
    )
    parser.add_argument("--unfreeze-after", type=int, default=None,
                        help="Epoch at which to unfreeze the MelHuBERT encoder. "
                             "None = stay frozen.")
    parser.add_argument("--unfreeze-top-n", type=int, default=None,
                        help="When unfreezing, only thaw the last N transformer layers "
                             "(+ final LayerNorm). None = unfreeze all. "
                             "Recommended: 2 or 3 for datasets < 500 clips.")
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                        help="Start with frozen encoder (default True).")
    # Training
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.4)
    parser.add_argument("--l1-lambda",  type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int,   default=64)
    parser.add_argument("--val-frac",   type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out-dir",    type=Path,  default=Path("runs"))
    parser.add_argument("--log-every",  type=int,   default=1,
                        help="Print detailed diagnostics every N epochs.")
    parser.add_argument("--patience",   type=int,   default=None,
                        help="Early stopping: stop if val AUC does not improve for "
                             "this many epochs. None = disabled.")
    parser.add_argument("--no-layer-log", action="store_true",
                        help="Skip per-layer gradient table (speeds up MelHuBERT logging).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Compute device. Options: auto (default), cpu, cuda, cuda:0, mps. "
            "'auto' picks cuda if available, then mps, then cpu."
        ),
    )

    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device resolution
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Resolve sparse_keys for cross-attention:
    #   None  (not passed)     → use defaults when melhubert mode
    #   []    (--sparse-keys passed with no args, i.e. '') → disabled
    #   [...]                  → user-specified keys
    if args.feature_mode == "melhubert":
        if args.sparse_keys is None:
            args.sparse_keys = _DEFAULT_SPARSE_KEYS
            print(f"[info] --sparse-keys not set; using {len(args.sparse_keys)} default keys")
        elif args.sparse_keys == [""]:
            # user passed --sparse-keys '' to explicitly disable
            args.sparse_keys = []
    else:
        if args.sparse_keys is None:
            args.sparse_keys = []

    print("\n" + "=" * 72)
    print(f"  TARGET       : {args.target}")
    print(f"  MODE         : {args.feature_mode}")
    print(f"  NEG CLASS    : {args.negative_class}")
    print(f"  EXCLUDE ILL  : {args.exclude_ill}")
    print(f"  TASKS        : {args.tasks or 'all'}")
    print(f"  AGE          : {args.age_min or '—'} – {args.age_max or '—'}")
    if args.feature_mode == "melhubert":
        print(f"  CKPT         : {args.ckpt_path}")
        xattn_status = (f"ENABLED (FiLM) — {len(args.sparse_keys)} keys: {args.sparse_keys}"
                        if args.sparse_keys else "DISABLED (pass --sparse-keys to enable)")
        print(f"  CONDITIONING : {xattn_status}")
        unfreeze_desc = "never"
        if args.unfreeze_after:
            layers = f"top-{args.unfreeze_top_n}" if args.unfreeze_top_n else "ALL"
            unfreeze_desc = f"epoch {args.unfreeze_after} ({layers} layers, encoder LR={args.lr*0.1:.1e})"
        print(f"  UNFREEZE     : {unfreeze_desc}")
    print(f"  DROPOUT      : {args.dropout}   L1λ: {args.l1_lambda}   HIDDEN: {args.hidden_dim}")
    print(f"  DEVICE       : {device}  (--device {args.device})")
    print("=" * 72)

    # ── 1. Manifest ────────────────────────────────────────────────────────────
    manifest = build_manifest(
        dataset_dir=args.dataset_dir,
        target=args.target,
        negative_class=args.negative_class,
        age_min=args.age_min,
        age_max=args.age_max,
        exclude_ill=args.exclude_ill,
        tasks=args.tasks,
        feature_source="sparc_ema",
    )
    manifest["participant_id"] = manifest["participant_id"].astype(str)
    n_pos = int((manifest["label"] == 1).sum())
    n_neg = int((manifest["label"] == 0).sum())
    print(f"\nTotal clips: {len(manifest)}  Pos: {n_pos}  Neg: {n_neg}  "
          f"Ratio 1:{n_neg//max(n_pos,1)}")
    if n_pos < 3:
        print("ERROR: too few positives."); return

    # ── 2. Stratified participant split ────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    pos_pids = manifest[manifest["label"] == 1]["participant_id"].unique()
    neg_pids = manifest[manifest["label"] == 0]["participant_id"].unique()
    rng.shuffle(pos_pids); rng.shuffle(neg_pids)
    val_pids = set(pos_pids[:max(1, int(len(pos_pids)*args.val_frac))]) | \
               set(neg_pids[:max(1, int(len(neg_pids)*args.val_frac))])
    train_pids = set(manifest["participant_id"].unique()) - val_pids
    tr_mf = manifest[manifest["participant_id"].isin(train_pids)].copy()
    va_mf = manifest[manifest["participant_id"].isin(val_pids)].copy()
    print(f"Train: {len(tr_mf)} clips ({len(train_pids)} participants)  "
          f"Pos={int((tr_mf['label']==1).sum())}  Neg={int((tr_mf['label']==0).sum())}")
    print(f"Val:   {len(va_mf)} clips ({len(val_pids)} participants)  "
          f"Pos={int((va_mf['label']==1).sum())}  Neg={int((va_mf['label']==0).sum())}")

    # ── 3. Feature loading ─────────────────────────────────────────────────────
    feat_cols: list[str] = []
    collate_fn = None

    if args.feature_mode == "mlp":
        print("\nLoading OpenSMILE static features...")
        feat_df = load_static_features(args.dataset_dir)
        feat_cols = [c for c in feat_df.columns if c not in ("participant_id", "task_name")]
        feat_df[feat_cols] = feat_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feat_df["participant_id"] = feat_df["participant_id"].astype(str)

        def _make_static_rows(mf):
            merged = mf.merge(feat_df[["participant_id","task_name"]+feat_cols],
                              on=["participant_id","task_name"], how="left")
            merged[feat_cols] = merged[feat_cols].fillna(0.0)
            return [{"features": row[feat_cols].values.astype(np.float32),
                     "label": int(row["label"])} for _, row in merged.iterrows()]

        train_rows = _make_static_rows(tr_mf)
        val_rows   = _make_static_rows(va_mf)
        mu  = np.stack([r["features"] for r in train_rows]).mean(axis=0)
        std = np.stack([r["features"] for r in train_rows]).std(axis=0) + 1e-8
        for r in train_rows + val_rows:
            r["features"] = (r["features"] - mu) / std

        train_ds = StaticDataset(train_rows)
        val_ds   = StaticDataset(val_rows)
        model    = BinaryMLP(len(feat_cols), args.hidden_dim, args.dropout).to(device)

    elif args.feature_mode == "tcn":
        print("\nLoading EMA time-series features...")
        ema_df = load_ema_features(args.dataset_dir)
        ema_df["participant_id"] = ema_df["participant_id"].astype(str)

        def _make_ema_rows(mf):
            merged = mf.merge(ema_df, on=["participant_id","task_name"], how="left")
            rows = []
            for _, row in merged.iterrows():
                arr = row.get("ema_2d")
                if arr is None or not isinstance(arr, np.ndarray):
                    arr = np.zeros((1, 12), dtype=np.float32)
                rows.append({"ema_2d": arr.astype(np.float32), "label": int(row["label"])})
            return rows

        train_rows = _make_ema_rows(tr_mf)
        val_rows   = _make_ema_rows(va_mf)
        train_ds   = EMADataset(train_rows)
        val_ds     = EMADataset(val_rows)
        collate_fn = _pad_collate_2d
        model      = BinaryTCN(12, 128, 4, args.dropout).to(device)

    else:  # melhubert
        print("\nLoading mel spectrograms (60-bin → zero-pad to 80)...")
        mel_df = load_mel_features(args.dataset_dir, target_bins=80)
        mel_df["participant_id"] = mel_df["participant_id"].astype(str)

        # Optional sparse vector from static features
        sparse_dim = len(args.sparse_keys)
        if sparse_dim > 0:
            print(f"Loading sparse conditioning features: {args.sparse_keys}")
            sf_df = load_static_features(args.dataset_dir)
            sf_df["participant_id"] = sf_df["participant_id"].astype(str)
            # Normalise sparse features on train participants only

        def _make_mel_rows(mf: pd.DataFrame, sf_means=None, sf_stds=None):
            merged = mf.merge(mel_df, on=["participant_id","task_name"], how="left")
            if sparse_dim > 0:
                merged = merged.merge(
                    sf_df[["participant_id","task_name"] + args.sparse_keys],
                    on=["participant_id","task_name"], how="left"
                )
            rows = []
            for _, row in merged.iterrows():
                arr = row.get("mel_2d")
                if arr is None or not isinstance(arr, np.ndarray):
                    arr = np.zeros((80, 1), dtype=np.float32)

                sv = np.zeros(sparse_dim, dtype=np.float32)
                if sparse_dim > 0:
                    for j, k in enumerate(args.sparse_keys):
                        raw = row.get(k)
                        # pd.isna catches float NaN, None, and pd.NA;
                        # "nan or 0.0" would return nan because nan is truthy
                        v = 0.0 if pd.isna(raw) else float(raw)
                        if sf_means is not None:
                            v = (v - sf_means[j]) / (sf_stds[j] + 1e-8)
                            # Clamp to ±5σ — HNR stddev has extreme outliers (-1812 → +624)
                            v = float(np.clip(v, -5.0, 5.0))
                        sv[j] = v

                rows.append({"mel": arr, "sparse_vector": sv, "label": int(row["label"])})
            return rows

        train_rows_raw = _make_mel_rows(tr_mf)
        # Fit sparse normalisation on train (NaNs already replaced with 0 in _make_mel_rows)
        sf_means = sf_stds = None
        if sparse_dim > 0:
            svecs = np.stack([r["sparse_vector"] for r in train_rows_raw])
            sf_means = np.nanmean(svecs, axis=0)
            sf_stds  = np.nanstd(svecs, axis=0)
            # Re-make with normalisation
            train_rows = _make_mel_rows(tr_mf, sf_means, sf_stds)
            val_rows   = _make_mel_rows(va_mf, sf_means, sf_stds)
        else:
            train_rows = train_rows_raw
            val_rows   = _make_mel_rows(va_mf)

        train_ds   = MelDataset(train_rows)
        val_ds     = MelDataset(val_rows)
        collate_fn = mel_collate

        # Build TransformerBranchConfig
        tf_cfg = TransformerBranchConfig(
            ckpt_path      = None,           # loaded below
            freeze_encoder = True,
            input_mel_bins = 80,
            melhubert_feat_dim = 80,
            sparc_channels = 0,
            sparse_keys    = args.sparse_keys,
            sparse_dim     = sparse_dim if sparse_dim > 0 else 64,
            embed_dim      = 768,
            n_encoder_layers = 2,
            pool           = "mean",
        )
        model = BinaryMelHuBERT(tf_cfg).to(device)

        ckpt = Path(args.ckpt_path)
        if ckpt.exists():
            print(f"Loading checkpoint: {ckpt}  ({ckpt.stat().st_size/1e9:.2f} GB)")
            model.branch.load_pretrained(str(ckpt))
            print("Checkpoint loaded ✓")
        else:
            print(f"[warn] Checkpoint not found at {ckpt} — training from scratch")

    # ── 4. DataLoaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Class reweighting
    train_labels = [r["label"] for r in train_rows]
    n_tr_pos = sum(train_labels)
    n_tr_neg = len(train_labels) - n_tr_pos
    pos_weight = torch.tensor([n_tr_neg / max(n_tr_pos, 1)], dtype=torch.float32)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel       : {model.__class__.__name__}")
    if args.feature_mode == "mlp":
        print(f"  Arch      : Linear({len(feat_cols)}→{args.hidden_dim}→{args.hidden_dim//2}→1)")
    elif args.feature_mode == "tcn":
        print(f"  Arch      : TCN(12ch, 4 blocks, embed=128) → Linear(128→1)")
    else:
        print(f"  Arch      : MelHuBERT(80-bin mel, 12-layer enc, embed=768)")
        print(f"  Cross-attn: {'✓ sparse_dim='+str(sparse_dim) if sparse_dim > 0 else '✗ disabled'}")
        print(f"  Encoder   : FROZEN (unfreeze at epoch {args.unfreeze_after or 'never'})")
    print(f"  Params    : {n_params:,}  (trainable)")
    print(f"  pos_weight: {pos_weight.item():.2f}  (train pos={n_tr_pos}, neg={n_tr_neg})")

    # ── 5. Optimizer + scheduler ───────────────────────────────────────────────
    if args.feature_mode == "melhubert":
        # Separate LR groups: encoder 0.1×, rest 1×
        param_groups = model.branch.get_param_groups()
        opt_groups = [{"params": pg["params"], "lr": args.lr * pg["lr_scale"]}
                      for pg in param_groups]
        # head params
        head_params = list(model.head.parameters())
        opt_groups.append({"params": head_params, "lr": args.lr})
        optimizer = torch.optim.AdamW(opt_groups, weight_decay=1e-3)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── 6. Training loop ───────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_name  = f"{args.target}_{args.feature_mode}"
    ckpt_best = args.out_dir / f"{run_name}_best.pt"
    metrics_csv = args.out_dir / f"{run_name}_metrics.csv"

    best_val_auc = -1.0
    no_improve_count = 0
    all_metrics: list[dict] = []

    header = (
        f"\n{'─'*80}\n"
        f"{'Ep':>3}  {'TrLoss':>7}  {'TrAUC':>6}  {'VaLoss':>7}  {'VaAUC':>6}  "
        f"{'VaF1':>5}  {'GNorm':>7}  {'LogitMu':>8}  {'LogitSD':>7}\n"
        f"{'─'*80}"
    )
    print(header)

    for epoch in range(1, args.epochs + 1):
        # Progressive unfreeze at specified epoch
        if args.feature_mode == "melhubert" and \
                args.unfreeze_after and epoch == args.unfreeze_after:
            model.branch.unfreeze_encoder(top_n=args.unfreeze_top_n)
            n_label = f"top-{args.unfreeze_top_n}" if args.unfreeze_top_n else "ALL"
            n_trainable = sum(p.requires_grad for p in model.branch.backbone.parameters())
            print(f"\n  [epoch {epoch}] Encoder {n_label} layers UNFROZEN "
                  f"({n_trainable} backbone params now trainable)")

            # Rebuild optimizer so unfrozen encoder params are included at 0.1× LR.
            # Using get_param_groups() which handles the encoder/non-encoder split.
            param_groups = model.branch.get_param_groups()
            opt_groups = [{"params": pg["params"], "lr": args.lr * pg["lr_scale"]}
                          for pg in param_groups]
            opt_groups.append({"params": list(model.head.parameters()), "lr": args.lr})
            optimizer = torch.optim.AdamW(opt_groups, weight_decay=1e-3)
            # Reset scheduler over remaining epochs so cosine annealing restarts cleanly
            remaining = args.epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining, 1), eta_min=args.lr * 0.001
            )
            print(f"  [epoch {epoch}] Optimizer rebuilt — encoder LR={args.lr*0.1:.2e}  "
                  f"head LR={args.lr:.2e}")

        t0 = time.time()

        if args.feature_mode == "melhubert":
            tr_loss, tr_m, tr_gnorm = run_epoch_melhubert(
                model, train_loader, optimizer, pos_weight, device
            )
            va_loss, va_m, _ = run_epoch_melhubert(
                model, val_loader, None, pos_weight, device
            )
        else:
            tr_loss, tr_m, tr_gnorm = run_epoch_mlp_tcn(
                model, train_loader, optimizer, pos_weight, device, args.l1_lambda
            )
            va_loss, va_m, _ = run_epoch_mlp_tcn(
                model, val_loader, None, pos_weight, device, 0.0
            )

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"{epoch:>3}  {tr_loss:>7.4f}  {tr_m['auc']:>6.3f}  "
            f"{va_loss:>7.4f}  {va_m['auc']:>6.3f}  {va_m['f1']:>5.3f}  "
            f"{tr_gnorm:>7.4f}  {va_m['logit_mean']:>8.3f}  {va_m['logit_std']:>7.3f}"
            f"  [{elapsed:.1f}s]"
        )

        if epoch % args.log_every == 0 or epoch == args.epochs:
            print_confusion(tr_m, "Train ")
            print_confusion(va_m, "Val   ")
            gap = tr_m["auc"] - va_m["auc"] if not (np.isnan(tr_m["auc"]) or np.isnan(va_m["auc"])) else 0
            if gap > 0.15:
                print(f"  [!] AUC gap={gap:.3f} → overfitting. "
                      "↑dropout / ↑L1λ / ↓hidden-dim / more tasks")

            if args.feature_mode == "melhubert":
                melhubert_diagnostics(model, epoch, log_layers=not args.no_layer_log)

        save_val = va_m["auc"] if not np.isnan(va_m["auc"]) else -va_loss
        if save_val > best_val_auc:
            best_val_auc = save_val
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auc": va_m["auc"], "val_f1": va_m["f1"],
                "args": vars(args),
                "feat_cols": feat_cols,
                "sparse_keys": args.sparse_keys if args.feature_mode == "melhubert" else [],
            }, ckpt_best)
            print(f"  ★ best checkpoint  val_auc={va_m['auc']:.4f}  f1={va_m['f1']:.4f}")
        else:
            no_improve_count += 1
            if args.patience and no_improve_count >= args.patience:
                print(f"\n  [early stop] val AUC did not improve for {args.patience} epochs "
                      f"(best={best_val_auc:.4f} @ epoch {epoch - args.patience})")
                break

        all_metrics.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr_loss, "train_auc": tr_m["auc"], "train_acc": tr_m["acc"],
            "train_f1": tr_m["f1"], "train_gnorm": tr_gnorm,
            "val_loss": va_loss, "val_auc": va_m["auc"], "val_acc": va_m["acc"],
            "val_f1": va_m["f1"], "val_prec": va_m["prec"], "val_rec": va_m["rec"],
            "val_tp": va_m["tp"], "val_tn": va_m["tn"], "val_fp": va_m["fp"], "val_fn": va_m["fn"],
            "val_logit_mean": va_m["logit_mean"], "val_logit_std": va_m["logit_std"],
            "val_mean_prob_pos": va_m["mean_prob_pos"], "val_mean_prob_neg": va_m["mean_prob_neg"],
        })

    print(f"{'─'*80}")

    # ── 7. End-of-run report ───────────────────────────────────────────────────
    if args.feature_mode == "mlp" and feat_cols:
        print_feature_importance(model, feat_cols, top_n=20)
    elif args.feature_mode == "melhubert":
        print("\n── Final weight norms ──")
        melhubert_diagnostics(model, args.epochs, log_layers=True)

    pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
    print(f"\nMetrics → {metrics_csv}")
    print(f"Best ckpt → {ckpt_best}")
    print(f"Best val AUC: {best_val_auc:.4f}\n")


if __name__ == "__main__":
    main()
