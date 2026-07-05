"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 1C: Learnable Fusion Network (ML Fuser)
═══════════════════════════════════════════════════════════════════

A small neural network that learns HOW to combine the Phase 1A (text) and
Phase 1B (voice) 15-dim probability outputs, instead of using the fixed
0.60/0.40 rule. It is the "ML" arm of the Phase 1C ablation, compared against
the rule-based fusers in rule_fuser.py.

Why a learnable fuser can beat a fixed rule
-------------------------------------------
The network sees CONFIDENCE FEATURES (entropy, max-prob, top-2 gap,
agreement, cosine, JS-divergence, modality mask) alongside the two
distributions. This lets it *dynamically* weight each modality per example —
trusting voice when voice is confident, discounting it when it's uncertain or
absent — which a static weight cannot do.

Architecture (gating network)
------------------------------
    input  = [ text_probs(15), voice_probs(15), conf_features(11) ]  → 41-dim
    body   = MLP( 41 → hidden → hidden )
    head   = produces GATING WEIGHTS, then fuses:

    gating_granularity:
      • "per_modality" (PRIMARY): head → 2 logits → softmax → (w_text, w_voice)
            fused = w_text * text_probs + w_voice * voice_probs
        One weight per modality, shared across all 15 emotions. Fewer params,
        matches the data we have, less overfitting.

      • "per_emotion" (ABLATION ARM): head → 30 logits → reshape (15, 2) →
            softmax over the 2 modalities per emotion →
            fused[e] = w_text[e]*text_probs[e] + w_voice[e]*voice_probs[e]
        Emotion-specific trust (e.g. trust voice more for "angry"). More
        expressive but needs more data; here it's a secondary arm to MEASURE,
        not the default.

    output = fused 15-dim distribution (renormalized to sum to 1)

Modality masking
----------------
When a modality is absent (mask=0), its gating weight is forced to 0 BEFORE
the softmax renormalization, so an absent modality never contributes. This
mirrors the rule fuser's dynamic switching and is critical for the 8 Aether
emotions that voice never trained on.

Loss
----
    L = CE(fused, target)  +  λ · KL(fused ‖ rule_fused)

The KL term is DISTILLATION toward the rule-based fuser. It anchors the
network's behavior where we have no gold supervision (agreement cases, and
the emotions MELD can't label), preventing the fuser from doing something
wild on unsupervised regions. λ (distill_weight) trades off "fit the labels"
vs "stay close to the sane rule".

Standalone usage
----------------
    python ml_fuser.py            # runs a self-test (forward + backward)

Training happens in train_fuser.py; this file only defines the model + loss.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Constants — must match config.py / fusion_data.py exactly.
# ─────────────────────────────────────────────────────────────
try:
    from config import AETHER_EMOTIONS  # type: ignore
except Exception:  # pragma: no cover
    AETHER_EMOTIONS = [
        "happy", "sad", "angry", "calm", "anxious",
        "energetic", "focused", "nostalgic", "romantic",
        "melancholic", "confident", "hopeful", "frustrated",
        "lonely", "dreamy",
    ]

NUM_EMOTIONS = len(AETHER_EMOTIONS)      # 15
NUM_CONF_FEATURES = 11                    # must match fusion_data.CONF_FEATURE_NAMES
INPUT_DIM = NUM_EMOTIONS * 2 + NUM_CONF_FEATURES  # 15 + 15 + 11 = 41
_EPS = 1e-8

VALID_GRANULARITIES = ("per_modality", "per_emotion")


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
@dataclass
class MLFuserConfig:
    """Hyperparameters for the learnable fuser. Saved alongside weights."""
    gating_granularity: str = "per_modality"   # "per_modality" | "per_emotion"
    hidden_dim: int = 64
    num_hidden_layers: int = 2
    dropout: float = 0.1
    distill_weight: float = 0.3                 # λ for the KL-to-rule term
    input_dim: int = INPUT_DIM
    num_emotions: int = NUM_EMOTIONS

    def validate(self) -> None:
        if self.gating_granularity not in VALID_GRANULARITIES:
            raise ValueError(
                f"gating_granularity must be one of {VALID_GRANULARITIES}, "
                f"got {self.gating_granularity!r}"
            )
        if self.hidden_dim <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("hidden_dim and num_hidden_layers must be positive.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if self.distill_weight < 0:
            raise ValueError("distill_weight must be >= 0.")


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════
class MLFuser(nn.Module):
    """Learnable confidence-gated fusion network.

    Forward inputs (all batched, shape (B, ...)):
        text_probs:    (B, 15) text distributions
        voice_probs:   (B, 15) voice distributions
        conf_features: (B, 11) confidence/agreement features
        modality_mask: (B, 2)  [text_present, voice_present] in {0,1}

    Forward output:
        fused_probs:   (B, 15) fused distributions (each row sums to 1)
    """

    def __init__(self, config: Optional[MLFuserConfig] = None):
        super().__init__()
        self.config = config or MLFuserConfig()
        self.config.validate()

        cfg = self.config
        # Number of gating logits the head must produce.
        #   per_modality → 2   (one weight per modality)
        #   per_emotion  → 30  (a text/voice pair for each of 15 emotions)
        self._num_gate_logits = 2 if cfg.gating_granularity == "per_modality" else NUM_EMOTIONS * 2

        # ── Body: MLP over the 41-dim input ──
        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for _ in range(cfg.num_hidden_layers):
            layers += [
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = cfg.hidden_dim
        self.body = nn.Sequential(*layers)

        # ── Head: body → gating logits ──
        self.gate_head = nn.Linear(in_dim, self._num_gate_logits)

    # ─────────────────────────────────────────────────────────
    def _compute_gate_weights(
        self,
        gate_logits: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Turn raw head logits into per-modality (or per-emotion) weights.

        Absent modalities (mask=0) are forced to zero weight BEFORE
        normalization, so they can never contribute to the fused output.

        Args:
            gate_logits: (B, 2) or (B, 30) raw logits from the head.
            modality_mask: (B, 2) with 1 = present, 0 = absent.

        Returns:
            weights of shape (B, 1, 2) for per_modality, or (B, 15, 2) for
            per_emotion — broadcastable against stacked (B, 15, 2) probs.
        """
        B = gate_logits.shape[0]

        if self.config.gating_granularity == "per_modality":
            # (B, 2) → mask → softmax over the 2 modalities → (B, 1, 2)
            logits = gate_logits                                   # (B, 2)
            masked = logits.masked_fill(modality_mask < 0.5, float("-inf"))
            weights = self._safe_softmax(masked, dim=-1)           # (B, 2)
            return weights.unsqueeze(1)                            # (B, 1, 2)

        # per_emotion: (B, 30) → (B, 15, 2)
        logits = gate_logits.view(B, NUM_EMOTIONS, 2)              # (B, 15, 2)
        # Broadcast the (B, 2) mask across all 15 emotions.
        mask = modality_mask.unsqueeze(1).expand(B, NUM_EMOTIONS, 2)
        masked = logits.masked_fill(mask < 0.5, float("-inf"))
        weights = self._safe_softmax(masked, dim=-1)              # (B, 15, 2)
        return weights

    @staticmethod
    def _safe_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
        """Softmax that tolerates rows where every entry is -inf (both
        modalities absent). Such rows return all zeros instead of NaN.
        """
        all_neg_inf = torch.isinf(logits).all(dim=dim, keepdim=True)
        safe = torch.where(all_neg_inf, torch.zeros_like(logits), logits)
        out = F.softmax(safe, dim=dim)
        # Zero out the degenerate rows (no modality present).
        out = torch.where(all_neg_inf, torch.zeros_like(out), out)
        return out

    # ─────────────────────────────────────────────────────────
    def forward(
        self,
        text_probs: torch.Tensor,
        voice_probs: torch.Tensor,
        conf_features: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse a batch. Returns (B, 15) distributions summing to 1."""
        self._check_shapes(text_probs, voice_probs, conf_features, modality_mask)

        x = torch.cat([text_probs, voice_probs, conf_features], dim=-1)  # (B, 41)
        h = self.body(x)
        gate_logits = self.gate_head(h)

        weights = self._compute_gate_weights(gate_logits, modality_mask)  # (B,1,2)|(B,15,2)

        # Stack the two distributions along a new "modality" axis → (B, 15, 2)
        stacked = torch.stack([text_probs, voice_probs], dim=-1)          # (B, 15, 2)

        # Weighted combine across the modality axis.
        fused = (stacked * weights).sum(dim=-1)                            # (B, 15)

        # Renormalize to a valid distribution (guards float drift; rows where
        # both modalities were absent stay all-zero, handled by callers).
        denom = fused.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        fused = fused / denom
        return fused

    @staticmethod
    def _check_shapes(t, v, c, m) -> None:
        if t.shape[-1] != NUM_EMOTIONS or v.shape[-1] != NUM_EMOTIONS:
            raise ValueError(f"text/voice must have last dim {NUM_EMOTIONS}.")
        if c.shape[-1] != NUM_CONF_FEATURES:
            raise ValueError(f"conf_features must have last dim {NUM_CONF_FEATURES}.")
        if m.shape[-1] != 2:
            raise ValueError("modality_mask must have last dim 2.")

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save weights (.pt) and a sidecar config JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)
        with open(p.with_suffix(".config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "MLFuser":
        """Load a saved fuser (reads the sidecar config to rebuild the arch)."""
        p = Path(path)
        cfg_path = p.with_suffix(".config.json")
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = MLFuserConfig(**json.load(f))
        else:
            cfg = MLFuserConfig()
        model = cls(cfg)
        model.load_state_dict(torch.load(p, map_location=map_location))
        model.eval()
        return model


# ═══════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════
class FusionLoss(nn.Module):
    """Cross-entropy to the gold label + KL distillation to the rule fuser.

        L = CE(fused, target) + λ · KL(fused ‖ rule_fused)

    The distillation term keeps the network sensible where labels are missing
    (agreement cases, unsupervised emotions). Pass rule_fused=None to disable.
    """

    def __init__(self, distill_weight: float = 0.3):
        super().__init__()
        if distill_weight < 0:
            raise ValueError("distill_weight must be >= 0.")
        self.distill_weight = distill_weight

    def forward(
        self,
        fused_probs: torch.Tensor,
        target: torch.Tensor,
        rule_fused: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total loss.

        Args:
            fused_probs: (B, 15) network output distributions.
            target: (B,) int64 gold emotion ids.
            rule_fused: (B, 15) rule-fuser distributions, or None.

        Returns:
            (total_loss, components_dict) — components for logging.
        """
        # CE expects log-probabilities; our output is already a distribution.
        log_probs = torch.log(fused_probs.clamp_min(_EPS))
        ce = F.nll_loss(log_probs, target)

        components = {"ce": float(ce.detach())}
        total = ce

        if rule_fused is not None and self.distill_weight > 0:
            # KL(fused ‖ rule): does the network agree with the sane rule?
            kl = F.kl_div(
                log_probs,
                rule_fused.clamp_min(_EPS),
                reduction="batchmean",
            )
            total = total + self.distill_weight * kl
            components["kl_distill"] = float(kl.detach())

        components["total"] = float(total.detach())
        return total, components


# ═══════════════════════════════════════════════════════════════════
# Convenience: numpy → tensors (for eval / inference glue)
# ═══════════════════════════════════════════════════════════════════
def to_tensors(
    text_probs: np.ndarray,
    voice_probs: np.ndarray,
    conf_features: np.ndarray,
    modality_mask: np.ndarray,
    device: str = "cpu",
) -> tuple[torch.Tensor, ...]:
    """Convert the numpy arrays from fusion_data's .npz into float tensors."""
    def _t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    return _t(text_probs), _t(voice_probs), _t(conf_features), _t(modality_mask)


# ═══════════════════════════════════════════════════════════════════
# Self-test  (python ml_fuser.py)
# ═══════════════════════════════════════════════════════════════════
def _selftest() -> None:
    torch.manual_seed(0)
    print("ML fuser self-test")
    print("-" * 50)

    B = 8

    def fake_batch():
        t = F.softmax(torch.randn(B, NUM_EMOTIONS), dim=-1)
        v = F.softmax(torch.randn(B, NUM_EMOTIONS), dim=-1)
        c = torch.rand(B, NUM_CONF_FEATURES)
        m = torch.ones(B, 2)
        # Consistent single-modality rows: when a modality is absent, BOTH its
        # mask entry is 0 AND its probability row is zeros (as produced by
        # fusion_data for real absent modalities).
        m[0, 0] = 0.0; t[0] = 0.0    # row 0: text absent  → voice-only
        m[1, 1] = 0.0; v[1] = 0.0    # row 1: voice absent → text-only
        y = torch.randint(0, NUM_EMOTIONS, (B,))
        rule = F.softmax(torch.randn(B, NUM_EMOTIONS), dim=-1)
        return t, v, c, m, y, rule

    for gran in VALID_GRANULARITIES:
        cfg = MLFuserConfig(gating_granularity=gran, hidden_dim=32)
        model = MLFuser(cfg)
        n_params = sum(p.numel() for p in model.parameters())

        t, v, c, m, y, rule = fake_batch()
        fused = model(t, v, c, m)

        # shape + validity
        assert fused.shape == (B, NUM_EMOTIONS), fused.shape
        row_sums = fused.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-4), row_sums
        assert not torch.isnan(fused).any(), "NaNs in output"

        # masking: text-absent row (index 0) must equal voice_probs exactly
        # (all weight forced onto the present modality)
        assert torch.allclose(fused[0], v[0], atol=1e-4), \
            "text-absent row should equal voice distribution"
        # voice-absent row (index 1) must equal text_probs
        assert torch.allclose(fused[1], t[1], atol=1e-4), \
            "voice-absent row should equal text distribution"

        # loss + backward
        loss_fn = FusionLoss(distill_weight=0.3)
        loss, comps = loss_fn(fused, y, rule)
        loss.backward()
        grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.parameters())
        assert grad_ok, "no gradients flowed"

        print(f"  [{gran:12}] params={n_params:5d} "
              f"gate_logits={model._num_gate_logits:2d} "
              f"loss={comps['total']:.4f} (ce={comps['ce']:.3f}, "
              f"kl={comps.get('kl_distill', 0):.3f})  ✓ mask+grad ok")

    # save/load roundtrip
    import tempfile, os
    cfg = MLFuserConfig(gating_granularity="per_modality")
    m1 = MLFuser(cfg)
    m1.eval()   # eval mode so dropout is off and outputs are deterministic
    tmp = os.path.join(tempfile.gettempdir(), "mlfuser_test.pt")
    m1.save(tmp)
    m2 = MLFuser.load(tmp)   # load() already sets eval()
    t, v, c, mask, *_ = fake_batch()
    with torch.no_grad():
        o1, o2 = m1(t, v, c, mask), m2(t, v, c, mask)
    assert torch.allclose(o1, o2, atol=1e-5), "save/load mismatch"
    print("  save/load roundtrip ✓")

    print("-" * 50)
    print("✅ All ML-fuser self-tests passed.")


if __name__ == "__main__":
    _selftest()
