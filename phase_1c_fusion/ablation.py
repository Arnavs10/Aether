"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 1C: Fusion Ablation Study
═══════════════════════════════════════════════════════════════════

The decisive experiment. Evaluates every fusion strategy on the HELD-OUT test
set (meld_test.npz) with a rigorous metric suite, then picks the winner on
evidence — not vibes.

Baselines compared
------------------
  1. text_only        — argmax of the text model alone (reference)
  2. voice_only       — argmax of the voice model alone (reference)
  3. rule_fixed       — FixedWeightFuser (0.60/0.40, the handoff spec)
  4. rule_entropy     — EntropyGatedFuser (the tuned/gated rule)
  5. ml_per_modality  — trained MLFuser, 2 gating weights (PRIMARY arm)
  6. ml_per_emotion   — trained MLFuser, 30 gating weights (ablation arm)

Metrics (per baseline)
----------------------
  • accuracy
  • macro-F1  (primary selection metric — robust to class imbalance)
  • weighted-F1
  • per-emotion F1
  • per-regime breakdown (text_and_voice / text_only / voice_only)
  • ECE (Expected Calibration Error) — are the probabilities trustworthy?

Significance
------------
  • McNemar's test between each fuser and the fixed-rule baseline, so a
    reported "win" is statistically real rather than noise.

Decision rule (stated upfront, applied mechanically)
----------------------------------------------------
  Choose the fuser with the highest test macro-F1. If the best ML fuser does
  not beat the best rule fuser by a statistically significant margin
  (McNemar p < 0.05), PREFER THE RULE — it's simpler, explainable, and better
  for the downstream RAG layer. Either outcome is a strong result.

Usage:
    python ablation.py
    python ablation.py --data-dir ./data/fusion --models-dir ./models/fusion
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from rule_fuser import FixedWeightFuser, EntropyGatedFuser
from ml_fuser import MLFuser, to_tensors, NUM_EMOTIONS

try:
    from config import AETHER_EMOTIONS  # type: ignore
except Exception:  # pragma: no cover
    from ml_fuser import AETHER_EMOTIONS

import torch

_EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════
def per_class_f1(preds: np.ndarray, targets: np.ndarray) -> dict[int, float]:
    """F1 for each class that appears in `targets`."""
    out = {}
    for c in sorted(set(targets.tolist())):
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[c] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return out


def macro_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    f1s = list(per_class_f1(preds, targets).values())
    return float(np.mean(f1s)) if f1s else 0.0


def weighted_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    f1 = per_class_f1(preds, targets)
    total = len(targets)
    return float(sum(
        f1[c] * int(np.sum(targets == c)) / total for c in f1
    )) if total else 0.0


def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(preds == targets)) if len(targets) else 0.0


def expected_calibration_error(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: weighted gap between confidence and accuracy across bins.

    Low ECE = the model's stated probabilities match its real hit rate.
    Rows with no probability mass (both modalities absent) are skipped.
    """
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    valid = probs.sum(axis=1) > _EPS
    conf, preds, tgt = conf[valid], preds[valid], targets[valid]
    if len(tgt) == 0:
        return 0.0

    correct = (preds == tgt).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(tgt)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def mcnemar_pvalue(preds_a: np.ndarray, preds_b: np.ndarray,
                   targets: np.ndarray) -> float:
    """McNemar's test on the discordant predictions of two models.

    Returns a two-sided p-value. Small p = the two models differ
    significantly in which examples they get right.
    """
    a_correct = preds_a == targets
    b_correct = preds_b == targets
    b01 = int(np.sum(a_correct & ~b_correct))   # a right, b wrong
    b10 = int(np.sum(~a_correct & b_correct))   # a wrong, b right
    n = b01 + b10
    if n == 0:
        return 1.0
    # Exact binomial two-sided p-value (robust for small discordant counts).
    from math import comb
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


# ═══════════════════════════════════════════════════════════════════
# Prediction producers  (each returns (probs (N,15), preds (N,)))
# ═══════════════════════════════════════════════════════════════════
def single_modality_probs(modality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use one modality's distribution directly."""
    probs = modality.copy()
    preds = probs.argmax(axis=1)
    return probs, preds


def rule_probs(fuser, text, voice, mask) -> tuple[np.ndarray, np.ndarray]:
    probs = fuser.predict_batch(text, voice, mask)
    return probs, probs.argmax(axis=1)


def ml_probs(model: MLFuser, text, voice, conf, mask
             ) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        t, v, c, m = to_tensors(text, voice, conf, mask)
        fused = model(t, v, c, m).cpu().numpy()
    return fused, fused.argmax(axis=1)


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════
def evaluate_baseline(
    name: str,
    probs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Full metric bundle for one baseline, including per-regime."""
    result = {
        "name": name,
        "accuracy": accuracy(preds, targets),
        "macro_f1": macro_f1(preds, targets),
        "weighted_f1": weighted_f1(preds, targets),
        "ece": expected_calibration_error(probs, targets),
        "per_emotion_f1": {
            AETHER_EMOTIONS[c]: round(v, 4)
            for c, v in per_class_f1(preds, targets).items()
        },
    }

    # Per-regime macro-F1.
    text_p = mask[:, 0] > 0.5
    voice_p = mask[:, 1] > 0.5
    regimes = {
        "text_and_voice": text_p & voice_p,
        "text_only": text_p & ~voice_p,
        "voice_only": ~text_p & voice_p,
    }
    result["per_regime_macro_f1"] = {}
    for rname, rmask in regimes.items():
        if rmask.sum() > 0:
            result["per_regime_macro_f1"][rname] = round(
                macro_f1(preds[rmask], targets[rmask]), 4
            )
        else:
            result["per_regime_macro_f1"][rname] = None
    return result


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    parser = argparse.ArgumentParser(description="Aether fusion ablation study.")
    parser.add_argument("--data-dir", default="./data/fusion")
    parser.add_argument("--models-dir", default="./models/fusion")
    parser.add_argument("--out", default="./data/fusion/ablation_results.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)

    test_npz = data_dir / "meld_test.npz"
    if not test_npz.exists():
        print(f"❌ Missing {test_npz}. Build the dataset first.")
        return 1

    data = np.load(test_npz)
    text = data["text_probs"].astype(np.float32)
    voice = data["voice_probs"].astype(np.float32)
    conf = data["conf_features"].astype(np.float32)
    mask = data["modality_mask"].astype(np.float32)
    targets = data["targets"].astype(np.int64)
    print(f"Test set: {len(targets):,} examples\n")

    results: dict[str, dict] = {}
    preds_by_name: dict[str, np.ndarray] = {}

    # ── Reference single-modality baselines ──
    for name, mod in [("text_only", text), ("voice_only", voice)]:
        p, pr = single_modality_probs(mod)
        results[name] = evaluate_baseline(name, p, pr, targets, mask)
        preds_by_name[name] = pr

    # ── Rule fusers ──
    for name, fuser in [
        ("rule_fixed", FixedWeightFuser()),
        ("rule_entropy", EntropyGatedFuser(gate_strength=0.5)),
    ]:
        p, pr = rule_probs(fuser, text, voice, mask)
        results[name] = evaluate_baseline(name, p, pr, targets, mask)
        preds_by_name[name] = pr

    # ── ML fusers (load whichever arms exist) ──
    for name, fname in [
        ("ml_per_modality", "ml_fuser_per_modality.pt"),
        ("ml_per_emotion", "ml_fuser_per_emotion.pt"),
    ]:
        wpath = models_dir / fname
        if not wpath.exists():
            print(f"⚠️  {wpath} not found — skipping {name}. "
                  "(Run train_fuser.py first.)")
            continue
        model = MLFuser.load(str(wpath))
        p, pr = ml_probs(model, text, voice, conf, mask)
        results[name] = evaluate_baseline(name, p, pr, targets, mask)
        preds_by_name[name] = pr

    # ── Headline table ──
    print("═" * 78)
    print(f"{'baseline':<18}{'accuracy':>10}{'macroF1':>10}{'wF1':>10}{'ECE':>8}")
    print("─" * 78)
    for name, r in results.items():
        print(f"{name:<18}{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}"
              f"{r['weighted_f1']:>10.4f}{r['ece']:>8.4f}")
    print("═" * 78)

    # ── Per-regime (the honest view) ──
    print("\nPer-regime macro-F1:")
    print(f"{'baseline':<18}{'text+voice':>12}{'text_only':>12}{'voice_only':>12}")
    for name, r in results.items():
        reg = r["per_regime_macro_f1"]
        print(f"{name:<18}"
              f"{_fmt(reg['text_and_voice']):>12}"
              f"{_fmt(reg['text_only']):>12}"
              f"{_fmt(reg['voice_only']):>12}")

    # ── Significance vs the fixed rule ──
    print("\nMcNemar vs rule_fixed (p<0.05 = significantly different):")
    base = preds_by_name.get("rule_fixed")
    for name, pr in preds_by_name.items():
        if name == "rule_fixed" or base is None:
            continue
        p = mcnemar_pvalue(pr, base, targets)
        sig = "significant" if p < 0.05 else "not significant"
        print(f"   {name:<18} p={p:.4f}  ({sig})")

    # ── Evidence-based decision ──
    fuser_names = [n for n in results
                   if n not in ("text_only", "voice_only")]
    best = max(fuser_names, key=lambda n: results[n]["macro_f1"])
    best_rule = max(
        [n for n in fuser_names if n.startswith("rule")],
        key=lambda n: results[n]["macro_f1"], default=None,
    )
    best_ml = max(
        [n for n in fuser_names if n.startswith("ml")],
        key=lambda n: results[n]["macro_f1"], default=None,
    )

    print("\n" + "═" * 78)
    print("DECISION")
    print("═" * 78)
    winner = best
    rationale = f"highest test macro-F1 ({results[best]['macro_f1']:.4f})"

    if best_ml and best_rule and best == best_ml:
        p = mcnemar_pvalue(preds_by_name[best_ml],
                           preds_by_name[best_rule], targets)
        if p >= 0.05:
            winner = best_rule
            rationale = (
                f"{best_ml} had the top macro-F1 but did NOT beat {best_rule} "
                f"significantly (McNemar p={p:.4f}). Per the pre-stated rule, "
                f"prefer the simpler, explainable rule fuser."
            )
        else:
            rationale = (
                f"{best_ml} beats the best rule ({best_rule}) significantly "
                f"(McNemar p={p:.4f})."
            )

    print(f"  🏆 SHIP: {winner}")
    print(f"     {rationale}")
    print("═" * 78)

    # ── Persist ──
    out = {
        "test_size": int(len(targets)),
        "results": results,
        "decision": {"winner": winner, "rationale": rationale},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Full results → {args.out}")
    print("Next: python fusion.py  (unified inference interface for the winner)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
