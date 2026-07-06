"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 1C: Train the Learnable Fuser
═══════════════════════════════════════════════════════════════════

Trains the MLFuser (ml_fuser.py) on the fusion dataset built by
fusion_data.py. Trains one model PER gating granularity arm
("per_modality" primary, "per_emotion" ablation) so ablation.py can compare
both against the rule-based fusers.

Data (from data/fusion/*.npz):
    text_probs      (N, 15)   Phase 1A outputs
    voice_probs     (N, 15)   Phase 1B outputs
    conf_features   (N, 11)   confidence/agreement features
    modality_mask   (N, 2)    [text_present, voice_present]
    targets         (N,)      gold Aether emotion id (0..14)

Loss: CE(fused, target) + λ·KL(fused ‖ rule_fused), where rule_fused is the
FixedWeightFuser's output — precomputed once per split. The distillation term
anchors behavior where labels are absent (see ml_fuser.py).

Class imbalance: MELD is calm-heavy, so training uses class-weighted sampling
(WeightedRandomSampler) plus we report macro-F1 (not accuracy) as the model
selection metric — accuracy would reward predicting "calm" constantly.

Outputs (to models/fusion/):
    ml_fuser_per_modality.pt   (+ .config.json)
    ml_fuser_per_emotion.pt    (+ .config.json)
    train_history_<arm>.json   (loss/metric curves)

Usage:
    python train_fuser.py                      # trains both arms, default HPs
    python train_fuser.py --epochs 40 --lr 1e-3
    python train_fuser.py --arm per_modality   # train just one arm
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Local modules
from ml_fuser import (
    MLFuser, MLFuserConfig, FusionLoss, NUM_EMOTIONS, VALID_GRANULARITIES,
)
from rule_fuser import FixedWeightFuser

try:
    from config import AETHER_EMOTIONS  # type: ignore
except Exception:  # pragma: no cover
    from ml_fuser import AETHER_EMOTIONS  # falls back to the same list

_EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class FusionDataset(Dataset):
    """Wraps a fusion .npz and precomputes the rule-fuser distributions.

    Precomputing rule_fused once (rather than per epoch) keeps training fast,
    since the rule fuser is deterministic.
    """

    def __init__(self, npz_path: str, rule_fuser: FixedWeightFuser):
        data = np.load(npz_path)
        self.text = data["text_probs"].astype(np.float32)      # (N,15)
        self.voice = data["voice_probs"].astype(np.float32)    # (N,15)
        self.conf = data["conf_features"].astype(np.float32)   # (N,11)
        self.mask = data["modality_mask"].astype(np.float32)   # (N,2)
        self.targets = data["targets"].astype(np.int64)        # (N,)

        n = len(self.targets)
        if not (len(self.text) == len(self.voice) == len(self.conf)
                == len(self.mask) == n):
            raise ValueError(f"Ragged arrays in {npz_path}.")

        # Precompute rule-fuser output for the distillation target.
        self.rule = rule_fuser.predict_batch(
            self.text, self.voice, self.mask
        ).astype(np.float32)                                    # (N,15)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.text[i]),
            torch.from_numpy(self.voice[i]),
            torch.from_numpy(self.conf[i]),
            torch.from_numpy(self.mask[i]),
            torch.tensor(self.targets[i]),
            torch.from_numpy(self.rule[i]),
        )

    def class_sample_weights(self) -> np.ndarray:
        """Per-sample weights = inverse class frequency (for balanced sampling)."""
        counts = np.bincount(self.targets, minlength=NUM_EMOTIONS).astype(np.float64)
        inv = 1.0 / np.clip(counts, 1.0, None)
        return inv[self.targets]


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════
def macro_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """Unweighted mean of per-class F1 over classes present in `targets`.

    Macro-F1 (not accuracy) is our selection metric because the data is
    class-imbalanced; accuracy would over-reward the majority class.
    """
    present = sorted(set(targets.tolist()))
    f1s = []
    for c in present:
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ═══════════════════════════════════════════════════════════════════
# Train / evaluate one arm
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model: MLFuser, loader: DataLoader, device: str) -> tuple[float, float]:
    """Return (macro_f1, mean_ce) on a loader."""
    model.eval()
    all_preds, all_tgts, ce_sum, n = [], [], 0.0, 0
    for text, voice, conf, mask, tgt, _rule in loader:
        text, voice, conf, mask = (t.to(device) for t in (text, voice, conf, mask))
        tgt = tgt.to(device)
        fused = model(text, voice, conf, mask)
        logp = torch.log(fused.clamp_min(_EPS))
        ce_sum += float(nn.functional.nll_loss(logp, tgt, reduction="sum"))
        n += tgt.numel()
        all_preds.append(fused.argmax(dim=-1).cpu().numpy())
        all_tgts.append(tgt.cpu().numpy())
    preds = np.concatenate(all_preds)
    tgts = np.concatenate(all_tgts)
    return macro_f1(preds, tgts, NUM_EMOTIONS), ce_sum / max(n, 1)


def train_arm(
    granularity: str,
    train_ds: FusionDataset,
    val_ds: FusionDataset,
    args: argparse.Namespace,
    out_dir: Path,
    device: str,
) -> dict:
    """Train one granularity arm; save best-by-val-macro-F1 weights."""
    print(f"\n{'═'*60}\n  Training arm: {granularity}\n{'═'*60}")

    cfg = MLFuserConfig(
        gating_granularity=granularity,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.layers,
        dropout=args.dropout,
        distill_weight=args.distill_weight,
    )
    model = MLFuser(cfg).to(device)
    print(f"   params: {sum(p.numel() for p in model.parameters()):,}")

    # Balanced sampling to counter class imbalance.
    weights = train_ds.class_sample_weights()
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    loss_fn = FusionLoss(distill_weight=args.distill_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_f1 = -1.0
    best_path = out_dir / f"ml_fuser_{granularity}.pt"
    history = {"train_loss": [], "val_macro_f1": [], "val_ce": []}
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, batches = 0.0, 0
        for text, voice, conf, mask, tgt, rule in train_loader:
            text, voice, conf, mask, tgt, rule = (
                x.to(device) for x in (text, voice, conf, mask, tgt, rule)
            )
            optim.zero_grad()
            fused = model(text, voice, conf, mask)
            loss, _comp = loss_fn(fused, tgt, rule)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            epoch_loss += float(loss.detach())
            batches += 1
        scheduler.step()

        train_loss = epoch_loss / max(batches, 1)
        val_f1, val_ce = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_ce"].append(val_ce)

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save(str(best_path))
            patience_left = args.patience
            marker = "  ★ (best, saved)"
        else:
            patience_left -= 1

        if epoch == 1 or epoch % args.log_every == 0 or marker:
            print(f"   epoch {epoch:3d} | train_loss {train_loss:.4f} "
                  f"| val_macroF1 {val_f1:.4f} | val_ce {val_ce:.4f}{marker}")

        if patience_left <= 0:
            print(f"   early stop at epoch {epoch} (no val improvement in "
                  f"{args.patience} epochs).")
            break

    # Save history.
    with open(out_dir / f"train_history_{granularity}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"   ✅ best val macro-F1: {best_f1:.4f} → {best_path.name}")
    return {"granularity": granularity, "best_val_macro_f1": best_f1,
            "weights": str(best_path)}


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Aether ML fuser.")
    parser.add_argument("--data-dir", default="./data/fusion")
    parser.add_argument("--out-dir", default="./models/fusion")
    parser.add_argument("--arm", default="both",
                        choices=["both", *VALID_GRANULARITIES])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--distill-weight", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_npz = data_dir / "meld_train.npz"
    val_npz = data_dir / "meld_val.npz"
    for p in (train_npz, val_npz):
        if not p.exists():
            print(f"❌ Missing {p}. Build the dataset first (build_local.py).")
            return 1

    # Rule fuser used to precompute the distillation target.
    rule = FixedWeightFuser()
    print("Loading data + precomputing rule-fuser targets…")
    train_ds = FusionDataset(str(train_npz), rule)
    val_ds = FusionDataset(str(val_npz), rule)
    print(f"   train: {len(train_ds):,} | val: {len(val_ds):,}")

    arms = list(VALID_GRANULARITIES) if args.arm == "both" else [args.arm]
    results = [train_arm(a, train_ds, val_ds, args, out_dir, device) for a in arms]

    print(f"\n{'═'*60}\n  SUMMARY\n{'═'*60}")
    for r in results:
        print(f"   {r['granularity']:12} → val macro-F1 {r['best_val_macro_f1']:.4f}")
    print("\n🎉 Training done. Next: python ablation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
