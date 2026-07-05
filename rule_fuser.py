"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 1C: Rule-Based Fusion
═══════════════════════════════════════════════════════════════════

Two deterministic (non-learned) fusers that combine the Phase 1A (text) and
Phase 1B (voice) 15-dim probability outputs into one unified 15-dim
distribution over the Aether emotions.

Both fusers operate in DISTRIBUTION-SPACE — inputs are already-softmaxed
probability vectors, not raw audio/text. They never see a model; they only
combine two opinions.

Variant 1 — FixedWeightFuser
    The exact handoff spec:
        text + voice : 0.60 * text + 0.40 * voice
        text only    : 1.00 * text
        voice only   : 1.00 * voice        (acoustic; see note below)
    It switches weighting based on which modalities are present.

Variant 2 — EntropyGatedFuser  ("smart rule")
    Same modality switching, but when both modalities are present it does NOT
    trust them 0.60/0.40 blindly. It down-weights whichever modality is
    UNCERTAIN, where uncertainty = normalized entropy of that modality's
    distribution (flat/spread-out = uncertain; peaked = confident). This is
    the fair middle baseline in the ablation: it's the best a hand-tuned rule
    can reasonably do, so if the learnable ML fuser can't beat THIS, the rule
    wins on simplicity.

Why two? The ablation (ablation.py) compares:
    (1) FixedWeightFuser        — the literal spec
    (2) EntropyGatedFuser       — the tuned/gated rule
    (3) the learnable ML fuser  — ml_fuser.py
Comparing the ML fuser only against (1) would be an unfair, easy win; (2)
keeps the comparison honest.

A note on "voice only":
    config.FUSION_WEIGHTS["voice_only"] splits 0.5 acoustic / 0.5
    voice-transcription-text. In THIS module we receive two already-computed
    15-dim vectors (text_probs, voice_probs). When only voice is present, the
    "voice transcription text" would have been produced by running Phase 1A on
    Whisper's transcript — which, if available, arrives to us AS text_probs
    with the text modality marked present. So at the fusion-vector level,
    "voice only" here means: we have a voice vector but no text vector, and we
    use the voice (acoustic) vector directly. The 0.5/0.5 acoustic/transcript
    split lives upstream, not here.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────
# Emotion system — must match config.py / Phase 1A / 1B exactly.
# Imported from config when available; falls back to a local copy so this
# module is runnable standalone (e.g. in a notebook) without the package.
# ─────────────────────────────────────────────────────────────
try:
    from config import AETHER_EMOTIONS, FUSION_WEIGHTS  # type: ignore
except Exception:  # pragma: no cover - fallback for standalone use
    AETHER_EMOTIONS = [
        "happy", "sad", "angry", "calm", "anxious",
        "energetic", "focused", "nostalgic", "romantic",
        "melancholic", "confident", "hopeful", "frustrated",
        "lonely", "dreamy",
    ]
    FUSION_WEIGHTS = {
        "text_voice": {"text": 0.60, "voice": 0.40},
        "text_only": {"text": 1.0},
        "voice_only": {"voice_acoustic": 0.50, "voice_text": 0.50},
    }

NUM_EMOTIONS = len(AETHER_EMOTIONS)  # 15
_EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def _as_vector(probs: Optional[np.ndarray]) -> np.ndarray:
    """Coerce an input to a clean (15,) float64 vector.

    Args:
        probs: a probability-like array, or None.

    Returns:
        A (15,) non-negative float64 array. None or an all-zero input returns
        all zeros (meaning "modality absent / no signal").

    Raises:
        ValueError: if a non-None input is not length 15.
    """
    if probs is None:
        return np.zeros(NUM_EMOTIONS, dtype=np.float64)
    arr = np.asarray(probs, dtype=np.float64).flatten()
    if arr.shape[0] != NUM_EMOTIONS:
        raise ValueError(
            f"Expected a {NUM_EMOTIONS}-dim vector, got shape {arr.shape}. "
            "Text and voice vectors must be in AETHER_EMOTIONS order."
        )
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, 0.0, None)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a non-negative vector to sum to 1. All-zero stays all-zero."""
    total = float(vec.sum())
    if total <= _EPS:
        return vec
    return vec / total


def _normalized_entropy(p: np.ndarray) -> float:
    """Entropy of a distribution scaled to [0, 1] (1 = maximally uncertain).

    Args:
        p: a (15,) probability vector (need not be pre-normalized).

    Returns:
        Normalized entropy in [0, 1]; 0.0 for an all-zero (absent) vector.
    """
    s = float(p.sum())
    if s <= _EPS:
        return 0.0
    q = p / s
    q = np.clip(q, _EPS, 1.0)
    ent = float(-np.sum(q * np.log(q)))
    return ent / float(np.log(len(q)))


@dataclass
class FusionResult:
    """Output of a fuser for one example.

    Attributes:
        probs: (15,) fused probability distribution (sums to 1, unless both
            modalities were absent — then all zeros).
        top_emotion: name of the argmax emotion, or None if no signal.
        top_prob: probability of the top emotion.
        weights_used: the (text_weight, voice_weight) actually applied, for
            transparency/debugging and interview explainability.
        regime: "text_voice" | "text_only" | "voice_only" | "none".
    """
    probs: np.ndarray
    top_emotion: Optional[str]
    top_prob: float
    weights_used: tuple[float, float]
    regime: str


# ═══════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════
class BaseRuleFuser:
    """Shared plumbing: input validation, modality detection, result build."""

    name: str = "base"

    def _resolve_weights(
        self,
        text_probs: np.ndarray,
        voice_probs: np.ndarray,
        text_present: bool,
        voice_present: bool,
    ) -> tuple[float, float]:
        """Return (text_weight, voice_weight). Subclasses override for gating."""
        raise NotImplementedError

    def fuse(
        self,
        text_probs: Optional[np.ndarray],
        voice_probs: Optional[np.ndarray],
        text_present: Optional[bool] = None,
        voice_present: Optional[bool] = None,
    ) -> FusionResult:
        """Fuse one text vector and one voice vector into a unified distribution.

        Args:
            text_probs: (15,) text distribution, or None if text absent.
            voice_probs: (15,) voice distribution, or None if voice absent.
            text_present: optional explicit presence flag. If None, inferred
                from whether text_probs has any mass.
            voice_present: optional explicit presence flag. If None, inferred
                from whether voice_probs has any mass.

        Returns:
            FusionResult with the fused distribution and metadata.
        """
        t = _as_vector(text_probs)
        v = _as_vector(voice_probs)

        # Infer presence from mass if not explicitly told.
        t_present = (t.sum() > _EPS) if text_present is None else bool(text_present)
        v_present = (v.sum() > _EPS) if voice_present is None else bool(voice_present)

        # Normalize each present modality so weights mean what they say.
        if t_present:
            t = _normalize(t)
        if v_present:
            v = _normalize(v)

        # ── Modality regimes ──
        if not t_present and not v_present:
            zeros = np.zeros(NUM_EMOTIONS, dtype=np.float64)
            return FusionResult(zeros, None, 0.0, (0.0, 0.0), "none")

        if t_present and not v_present:
            return self._build(t, (1.0, 0.0), "text_only")

        if v_present and not t_present:
            return self._build(v, (0.0, 1.0), "voice_only")

        # Both present → subclass decides the weighting.
        w_text, w_voice = self._resolve_weights(t, v, t_present, v_present)
        fused = w_text * t + w_voice * v
        return self._build(fused, (w_text, w_voice), "text_voice")

    def _build(
        self,
        vec: np.ndarray,
        weights: tuple[float, float],
        regime: str,
    ) -> FusionResult:
        """Normalize the fused vector and package a FusionResult."""
        probs = _normalize(np.clip(vec, 0.0, None))
        if probs.sum() <= _EPS:
            return FusionResult(probs, None, 0.0, weights, regime)
        idx = int(np.argmax(probs))
        return FusionResult(
            probs=probs,
            top_emotion=AETHER_EMOTIONS[idx],
            top_prob=float(probs[idx]),
            weights_used=weights,
            regime=regime,
        )

    def predict_batch(
        self,
        text_probs: np.ndarray,
        voice_probs: np.ndarray,
        modality_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fuse a whole dataset at once and return (N, 15) fused probabilities.

        Args:
            text_probs: (N, 15) array of text distributions.
            voice_probs: (N, 15) array of voice distributions.
            modality_mask: optional (N, 2) array [text_present, voice_present].
                If None, presence is inferred from mass per row.

        Returns:
            (N, 15) array of fused distributions.
        """
        text_probs = np.asarray(text_probs, dtype=np.float64)
        voice_probs = np.asarray(voice_probs, dtype=np.float64)
        if text_probs.shape != voice_probs.shape or text_probs.shape[1] != NUM_EMOTIONS:
            raise ValueError(
                f"text_probs and voice_probs must both be (N, {NUM_EMOTIONS}); "
                f"got {text_probs.shape} and {voice_probs.shape}."
            )

        n = text_probs.shape[0]
        out = np.zeros((n, NUM_EMOTIONS), dtype=np.float64)
        for i in range(n):
            tp = bool(modality_mask[i, 0]) if modality_mask is not None else None
            vp = bool(modality_mask[i, 1]) if modality_mask is not None else None
            out[i] = self.fuse(text_probs[i], voice_probs[i], tp, vp).probs
        return out


# ═══════════════════════════════════════════════════════════════════
# Variant 1 — Fixed-weight fuser (the literal handoff spec)
# ═══════════════════════════════════════════════════════════════════
class FixedWeightFuser(BaseRuleFuser):
    """0.60*text + 0.40*voice when both present; single modality otherwise.

    Weights come from config.FUSION_WEIGHTS["text_voice"] so there is one
    source of truth. This is the deterministic baseline in the ablation.
    """

    name = "fixed_weight"

    def __init__(
        self,
        text_weight: Optional[float] = None,
        voice_weight: Optional[float] = None,
    ):
        """Args:
            text_weight / voice_weight: override the config weights (e.g. for a
                grid search). If None, read from config.FUSION_WEIGHTS.
        """
        tv = FUSION_WEIGHTS.get("text_voice", {"text": 0.60, "voice": 0.40})
        self.text_weight = float(tv["text"]) if text_weight is None else float(text_weight)
        self.voice_weight = float(tv["voice"]) if voice_weight is None else float(voice_weight)

        # Defensive: weights should be non-negative. Normalize if they don't
        # sum to 1 so the fused vector stays a proper distribution.
        if self.text_weight < 0 or self.voice_weight < 0:
            raise ValueError("Fusion weights must be non-negative.")
        s = self.text_weight + self.voice_weight
        if s <= _EPS:
            raise ValueError("Fusion weights sum to zero.")
        if abs(s - 1.0) > 1e-6:
            self.text_weight /= s
            self.voice_weight /= s

    def _resolve_weights(self, t, v, t_present, v_present) -> tuple[float, float]:
        return self.text_weight, self.voice_weight


# ═══════════════════════════════════════════════════════════════════
# Variant 2 — Entropy-gated fuser (the "smart rule" baseline)
# ═══════════════════════════════════════════════════════════════════
class EntropyGatedFuser(BaseRuleFuser):
    """Down-weights whichever modality is more uncertain (higher entropy).

    Intuition: a confident (peaked) distribution should count more than an
    uncertain (flat) one. We convert each modality's normalized entropy into a
    confidence = 1 - entropy, then blend the base fixed weights with these
    confidences via `gate_strength`:

        gate_strength = 0.0  → behaves exactly like FixedWeightFuser
        gate_strength = 1.0  → weights driven purely by confidence
        in between           → a mix (default 0.5)

    This is the strongest reasonable hand-built rule, and thus the fair bar the
    learnable ML fuser must clear in the ablation.
    """

    name = "entropy_gated"

    def __init__(
        self,
        base_text_weight: Optional[float] = None,
        base_voice_weight: Optional[float] = None,
        gate_strength: float = 0.5,
    ):
        """Args:
            base_text_weight / base_voice_weight: the fixed prior weights
                (default from config). The gate nudges away from these.
            gate_strength: in [0, 1]; how much confidence overrides the prior.
        """
        tv = FUSION_WEIGHTS.get("text_voice", {"text": 0.60, "voice": 0.40})
        bt = float(tv["text"]) if base_text_weight is None else float(base_text_weight)
        bv = float(tv["voice"]) if base_voice_weight is None else float(base_voice_weight)
        s = bt + bv
        if s <= _EPS:
            raise ValueError("Base fusion weights sum to zero.")
        self.base_text_weight = bt / s
        self.base_voice_weight = bv / s

        if not (0.0 <= gate_strength <= 1.0):
            raise ValueError("gate_strength must be in [0, 1].")
        self.gate_strength = float(gate_strength)

    def _resolve_weights(self, t, v, t_present, v_present) -> tuple[float, float]:
        # Confidence = 1 - normalized entropy (peaked => confident => ~1).
        conf_t = 1.0 - _normalized_entropy(t)
        conf_v = 1.0 - _normalized_entropy(v)

        conf_sum = conf_t + conf_v
        if conf_sum <= _EPS:
            # Both maximally uncertain → fall back to the base prior.
            return self.base_text_weight, self.base_voice_weight

        # Confidence-driven weights.
        gate_text = conf_t / conf_sum
        gate_voice = conf_v / conf_sum

        # Blend prior with gate by gate_strength.
        g = self.gate_strength
        w_text = (1.0 - g) * self.base_text_weight + g * gate_text
        w_voice = (1.0 - g) * self.base_voice_weight + g * gate_voice

        # Renormalize (guards against float drift).
        s = w_text + w_voice
        return w_text / s, w_voice / s


# ═══════════════════════════════════════════════════════════════════
# Self-test  (run: python rule_fuser.py)
# ═══════════════════════════════════════════════════════════════════
def _selftest() -> None:
    print("Rule fuser self-test")
    print("-" * 40)

    def onehot(i, peak=0.9):
        p = np.full(NUM_EMOTIONS, (1 - peak) / (NUM_EMOTIONS - 1))
        p[i] = peak
        return p

    fixed = FixedWeightFuser()
    gated = EntropyGatedFuser(gate_strength=0.5)

    # 1) Agreement: both say "happy" → fused should say happy.
    t = onehot(0); v = onehot(0)
    r = fixed.fuse(t, v)
    assert r.top_emotion == "happy" and abs(r.probs.sum() - 1) < 1e-6
    print(f"  agreement      → {r.top_emotion} ({r.top_prob:.2f})  weights={r.weights_used}")

    # 2) Text-only regime.
    r = fixed.fuse(onehot(7), None)
    assert r.regime == "text_only" and r.top_emotion == "nostalgic"
    print(f"  text-only      → {r.top_emotion}  regime={r.regime}")

    # 3) Voice-only regime.
    r = fixed.fuse(None, onehot(2))
    assert r.regime == "voice_only" and r.top_emotion == "angry"
    print(f"  voice-only     → {r.top_emotion}  regime={r.regime}")

    # 4) Both absent → none.
    r = fixed.fuse(None, None)
    assert r.regime == "none" and r.top_emotion is None
    print(f"  both absent    → regime={r.regime}")

    # 5) Gating: confident text vs. uncertain voice should tilt toward text.
    confident_text = onehot(0, peak=0.95)
    uncertain_voice = np.full(NUM_EMOTIONS, 1.0 / NUM_EMOTIONS)  # max entropy
    r = gated.fuse(confident_text, uncertain_voice)
    wt, wv = r.weights_used
    assert wt > wv, "gated fuser should favor the confident modality"
    print(f"  gated tilt     → text_w={wt:.2f} voice_w={wv:.2f} (text confident)")

    # 6) Fixed fuser ignores confidence (always 0.6/0.4).
    r = fixed.fuse(confident_text, uncertain_voice)
    assert abs(r.weights_used[0] - 0.6) < 1e-6
    print(f"  fixed weights  → {r.weights_used} (constant, as expected)")

    # 7) Batch path.
    N = 5
    tb = np.stack([onehot(i % NUM_EMOTIONS) for i in range(N)])
    vb = np.stack([onehot((i + 1) % NUM_EMOTIONS) for i in range(N)])
    out = fixed.predict_batch(tb, vb)
    assert out.shape == (N, NUM_EMOTIONS)
    assert np.allclose(out.sum(axis=1), 1.0)
    print(f"  batch          → shape {out.shape}, rows sum to 1 ✓")

    print("-" * 40)
    print("✅ All rule-fuser self-tests passed.")


if __name__ == "__main__":
    _selftest()
