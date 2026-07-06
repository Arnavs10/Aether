"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 1C: Unified Fusion Interface
═══════════════════════════════════════════════════════════════════

The single entry point downstream phases (2–8) use to turn text and/or voice
emotion predictions into ONE unified 15-dim Aether emotion distribution.

It hides every Phase 1C detail behind one method:

    fusion = AetherFusion()               # loads the ablation winner
    result = fusion.fuse(text_probs=..., voice_probs=...)
    result.top_emotion    # e.g. "happy"
    result.distribution   # (15,) numpy array over AETHER_EMOTIONS

Backend selection (evidence-driven)
-----------------------------------
By default the interface loads whichever fuser the ablation study chose (read
from ablation_results.json → decision.winner). This keeps the shipped model
tied to the evidence rather than hard-coded. You can override with `backend=`.

  • "ml_per_modality" / "ml_per_emotion" → trained MLFuser (needs weights)
  • "rule_fixed"                          → FixedWeightFuser
  • "rule_entropy"                        → EntropyGatedFuser
  • "auto" (default)                      → ablation winner, else best rule

Missing-modality handling
-------------------------
`fuse()` accepts either or both modalities. If one is None it's treated as
absent (mask=0); the fuser routes all weight to the present modality. If both
are None it raises — there is nothing to fuse.

This module recomputes the SAME confidence features the fuser was trained on
(via fusion_data.compute_conf_features) so the ML fuser sees consistent input.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from rule_fuser import FixedWeightFuser, EntropyGatedFuser
from fusion_data import compute_conf_features, NUM_EMOTIONS, AETHER_EMOTIONS

# ML fuser is optional at import time (rule-only deployments shouldn't need torch).
try:
    import torch  # noqa: F401
    from ml_fuser import MLFuser, to_tensors
    _ML_AVAILABLE = True
except Exception:  # pragma: no cover
    _ML_AVAILABLE = False

_EPS = 1e-8

_ML_BACKENDS = {"ml_per_modality", "ml_per_emotion"}
_RULE_BACKENDS = {"rule_fixed", "rule_entropy"}
_ML_WEIGHT_FILES = {
    "ml_per_modality": "ml_fuser_per_modality.pt",
    "ml_per_emotion": "ml_fuser_per_emotion.pt",
}


@dataclass
class FusionOutput:
    """Result of fusing one example.

    Attributes:
        distribution: (15,) fused probability vector over AETHER_EMOTIONS.
        top_emotion: name of the argmax emotion (None if no signal).
        top_prob: probability of the top emotion.
        regime: "text_and_voice" | "text_only" | "voice_only" | "none".
        backend: which fuser produced this ("ml_per_modality", ...).
    """
    distribution: np.ndarray
    top_emotion: Optional[str]
    top_prob: float
    regime: str
    backend: str

    def as_dict(self) -> dict:
        """JSON-serializable summary (handy for the REST API in Phase 8)."""
        return {
            "top_emotion": self.top_emotion,
            "top_prob": round(self.top_prob, 4),
            "regime": self.regime,
            "backend": self.backend,
            "distribution": {
                AETHER_EMOTIONS[i]: round(float(p), 4)
                for i, p in enumerate(self.distribution)
            },
        }


class AetherFusion:
    """Unified inference wrapper around the Phase 1C fusion layer."""

    def __init__(
        self,
        backend: str = "auto",
        models_dir: str = "./models/fusion",
        ablation_results: str = "./data/fusion/ablation_results.json",
        device: str = "cpu",
    ):
        """Load the chosen fusion backend.

        Args:
            backend: one of the backend names above, or "auto" to use the
                ablation winner (falls back to the best rule if unavailable).
            models_dir: directory holding the trained ML fuser weights.
            ablation_results: path to ablation_results.json (for "auto").
            device: torch device for the ML fuser.

        Raises:
            ValueError: on an unknown backend name.
            FileNotFoundError: if an ML backend is chosen but weights are missing.
        """
        self.models_dir = Path(models_dir)
        self.device = device

        if backend == "auto":
            backend = self._resolve_auto_backend(ablation_results)
        self.backend = backend

        self._ml_model = None
        self._rule = None
        self._load_backend(backend)

    # ─────────────────────────────────────────────────────────
    def _resolve_auto_backend(self, ablation_results: str) -> str:
        """Pick the backend the ablation study chose; fall back gracefully."""
        p = Path(ablation_results)
        if p.exists():
            try:
                with open(p) as f:
                    winner = json.load(f).get("decision", {}).get("winner")
                if winner:
                    return winner
            except Exception:
                pass
        # No ablation file → prefer a trained ML fuser if present, else a rule.
        if _ML_AVAILABLE and (self.models_dir / _ML_WEIGHT_FILES["ml_per_modality"]).exists():
            return "ml_per_modality"
        return "rule_entropy"

    def _load_backend(self, backend: str) -> None:
        if backend in _RULE_BACKENDS:
            self._rule = (
                FixedWeightFuser() if backend == "rule_fixed"
                else EntropyGatedFuser(gate_strength=0.5)
            )
            return

        if backend in _ML_BACKENDS:
            if not _ML_AVAILABLE:
                raise RuntimeError(
                    f"Backend {backend!r} needs PyTorch + ml_fuser.py, which "
                    "failed to import."
                )
            wpath = self.models_dir / _ML_WEIGHT_FILES[backend]
            if not wpath.exists():
                raise FileNotFoundError(
                    f"Trained weights not found: {wpath}. Run train_fuser.py, "
                    f"or use backend='rule_entropy'."
                )
            self._ml_model = MLFuser.load(str(wpath), map_location=self.device)
            self._ml_model.eval()
            return

        raise ValueError(
            f"Unknown backend {backend!r}. Choose from "
            f"{sorted(_ML_BACKENDS | _RULE_BACKENDS)} or 'auto'."
        )

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _clean(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Validate/normalize a probability vector, or return None if absent."""
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=np.float32).flatten()
        if arr.shape[0] != NUM_EMOTIONS:
            raise ValueError(
                f"Expected a {NUM_EMOTIONS}-dim vector, got shape {arr.shape}."
            )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        s = float(arr.sum())
        return arr / s if s > _EPS else None  # all-zero → treat as absent

    def fuse(
        self,
        text_probs: Optional[np.ndarray] = None,
        voice_probs: Optional[np.ndarray] = None,
    ) -> FusionOutput:
        """Fuse text and/or voice emotion distributions into one.

        Args:
            text_probs: (15,) Phase 1A output, or None if text absent.
            voice_probs: (15,) Phase 1B acoustic output, or None if voice absent.

        Returns:
            FusionOutput with the fused distribution + metadata.

        Raises:
            ValueError: if both modalities are absent.
        """
        t = self._clean(text_probs)
        v = self._clean(voice_probs)

        text_present = t is not None
        voice_present = v is not None
        if not text_present and not voice_present:
            raise ValueError("fuse() needs at least one modality; both were absent/empty.")

        # Zero-fill the absent modality so shapes are consistent downstream.
        t_vec = t if text_present else np.zeros(NUM_EMOTIONS, dtype=np.float32)
        v_vec = v if voice_present else np.zeros(NUM_EMOTIONS, dtype=np.float32)

        regime = (
            "text_and_voice" if text_present and voice_present
            else "text_only" if text_present
            else "voice_only"
        )

        if self.backend in _RULE_BACKENDS:
            fused = self._fuse_rule(t_vec, v_vec, text_present, voice_present)
        else:
            fused = self._fuse_ml(t_vec, v_vec, text_present, voice_present)

        fused = np.nan_to_num(fused, nan=0.0)
        s = float(fused.sum())
        fused = fused / s if s > _EPS else fused

        if fused.sum() <= _EPS:
            return FusionOutput(fused, None, 0.0, regime, self.backend)
        idx = int(np.argmax(fused))
        return FusionOutput(
            distribution=fused,
            top_emotion=AETHER_EMOTIONS[idx],
            top_prob=float(fused[idx]),
            regime=regime,
            backend=self.backend,
        )

    # ── backend-specific fusion ──
    def _fuse_rule(self, t, v, tp, vp) -> np.ndarray:
        result = self._rule.fuse(
            text_probs=t if tp else None,
            voice_probs=v if vp else None,
            text_present=tp,
            voice_present=vp,
        )
        return result.probs

    def _fuse_ml(self, t, v, tp, vp) -> np.ndarray:
        conf = compute_conf_features(t, v, tp, vp)                     # (11,)
        mask = np.array([1.0 if tp else 0.0, 1.0 if vp else 0.0], dtype=np.float32)
        # Add batch dim for the model.
        tt, vv, cc, mm = to_tensors(
            t[None, :], v[None, :], conf[None, :], mask[None, :], device=self.device
        )
        with torch.no_grad():
            out = self._ml_model(tt, vv, cc, mm).cpu().numpy()[0]      # (15,)
        return out


# ═══════════════════════════════════════════════════════════════════
# Self-test  (python fusion.py)
# ═══════════════════════════════════════════════════════════════════
def _selftest() -> None:
    print("Fusion interface self-test")
    print("-" * 50)

    def onehot(i, peak=0.9):
        p = np.full(NUM_EMOTIONS, (1 - peak) / (NUM_EMOTIONS - 1), dtype=np.float32)
        p[i] = peak
        return p

    # Rule backend always works (no weights needed) — test with it.
    fusion = AetherFusion(backend="rule_entropy")
    print(f"  backend loaded: {fusion.backend}")

    # both present
    r = fusion.fuse(text_probs=onehot(0), voice_probs=onehot(0))
    assert r.regime == "text_and_voice" and r.top_emotion == "happy"
    print(f"  both present  → {r.top_emotion} ({r.top_prob:.2f}) [{r.regime}]")

    # text only
    r = fusion.fuse(text_probs=onehot(7))
    assert r.regime == "text_only" and r.top_emotion == "nostalgic"
    print(f"  text only     → {r.top_emotion} [{r.regime}]")

    # voice only
    r = fusion.fuse(voice_probs=onehot(2))
    assert r.regime == "voice_only" and r.top_emotion == "angry"
    print(f"  voice only    → {r.top_emotion} [{r.regime}]")

    # both absent → raises
    try:
        fusion.fuse()
        raise AssertionError("should have raised on both-absent")
    except ValueError:
        print("  both absent   → correctly raised ValueError")

    # as_dict shape
    d = fusion.fuse(text_probs=onehot(0)).as_dict()
    assert set(d.keys()) == {"top_emotion", "top_prob", "regime", "backend", "distribution"}
    assert len(d["distribution"]) == NUM_EMOTIONS
    print("  as_dict()     → JSON-ready ✓")

    # If a trained ML fuser exists, smoke-test the auto/ML path too.
    if _ML_AVAILABLE and Path("./models/fusion/ml_fuser_per_modality.pt").exists():
        mlf = AetherFusion(backend="ml_per_modality")
        r = mlf.fuse(text_probs=onehot(0), voice_probs=onehot(3))
        assert r.distribution.shape == (NUM_EMOTIONS,)
        print(f"  ml backend    → {r.top_emotion} [{r.backend}] ✓")
    else:
        print("  ml backend    → (no weights present; skipped, rule path verified)")

    print("-" * 50)
    print("✅ All fusion-interface self-tests passed.")


if __name__ == "__main__":
    _selftest()
