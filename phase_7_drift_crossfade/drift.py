"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 7 · Emotion Drift Detector
═══════════════════════════════════════════════════════════════════
The Live Emotion Player watches the listener's mood while music plays. Each time
they speak/type, Phase 1C emits a fresh 15-emotion distribution. This detector
keeps a sliding window of those distributions and fires a DRIFT event when the
newest mood has moved far enough from the recent baseline — the signal to pick a
new song and cross-fade into it.

"Far enough" is measured with **Jensen–Shannon divergence** (symmetric, bounded
0–1 with log base 2) between the newest distribution and the mean of the prior
window. JS is the right tool here: it compares whole probability distributions
(not just the top label), so a mood *broadening* or *splitting* registers too,
not only a hard label flip.

After a drift fires, the window re-anchors to the new state, so a sustained new
mood doesn't keep re-triggering — it only fires again on the *next* real shift.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from config import AETHER_EMOTIONS                        # noqa: E402
except Exception:  # pragma: no cover
    AETHER_EMOTIONS = []


@dataclass
class DriftEvent:
    """The outcome of observing one new emotion distribution."""
    drifted: bool
    distance: float                     # JS divergence vs. baseline, 0–1
    from_emotion: str                   # dominant emotion of the baseline
    to_emotion: str                     # dominant emotion of the newest reading
    to_distribution: np.ndarray         # the newest distribution (for downstream)
    threshold: float


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).clip(min=0.0)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))


def _kl(a: np.ndarray, b: np.ndarray) -> float:
    mask = a > 0
    return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence in [0, 1] (log base 2). Symmetric."""
    p, q = _normalize(p), _normalize(q)
    m = 0.5 * (p + q)
    return max(0.0, min(1.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


class EmotionDriftDetector:
    """Fires when the listener's mood moves beyond `threshold` from baseline."""

    def __init__(self, window: int = 5, threshold: float = 0.22,
                 min_history: int = 1) -> None:
        """
        Args:
            window: how many recent readings form the baseline.
            threshold: JS distance (0–1) above which drift fires.
            min_history: readings required before drift can fire (warm-up).
        """
        self.window = max(2, window)
        self.threshold = threshold
        self.min_history = max(1, min_history)
        self._hist: deque[np.ndarray] = deque(maxlen=self.window)

    def _dominant(self, dist: np.ndarray) -> str:
        i = int(np.argmax(dist))
        return AETHER_EMOTIONS[i] if 0 <= i < len(AETHER_EMOTIONS) else str(i)

    def observe(self, distribution: np.ndarray) -> DriftEvent:
        """Feed a new 15-emotion distribution; return whether the mood drifted."""
        cur = _normalize(distribution)

        # Warm-up: not enough history to judge drift yet.
        if len(self._hist) < self.min_history:
            self._hist.append(cur)
            return DriftEvent(False, 0.0, self._dominant(cur),
                              self._dominant(cur), cur, self.threshold)

        baseline = _normalize(np.mean(np.stack(self._hist), axis=0))
        dist = js_divergence(cur, baseline)
        drifted = dist > self.threshold

        from_emo = self._dominant(baseline)
        to_emo = self._dominant(cur)

        if drifted:
            # Re-anchor: the new mood becomes the baseline going forward.
            self._hist.clear()
        self._hist.append(cur)

        return DriftEvent(drifted, dist, from_emo, to_emo, cur, self.threshold)

    def reset(self) -> None:
        """Clear all history (e.g. when a new session/song starts fresh)."""
        self._hist.clear()


# ─────────────────────────────────────────────────────────────
# Self-test — synthetic emotion streams
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Emotion drift detector self-test")
    print("-" * 55)

    n = len(AETHER_EMOTIONS) or 15

    def peak(emo: str, strength: float = 0.85) -> np.ndarray:
        v = np.full(n, (1 - strength) / (n - 1))
        v[AETHER_EMOTIONS.index(emo)] = strength
        return v

    # 1. Stable mood → no drift after warm-up.
    d = EmotionDriftDetector(window=4, threshold=0.22)
    events = [d.observe(peak("sad")) for _ in range(4)]
    assert not any(e.drifted for e in events), [e.distance for e in events]
    print(f"  stable 'sad' stream → no drift (max dist "
          f"{max(e.distance for e in events):.3f}) ✓")

    # 2. Hard switch sad → energetic → drift fires with correct from/to.
    ev = d.observe(peak("energetic"))
    assert ev.drifted, ev.distance
    assert ev.from_emotion == "sad" and ev.to_emotion == "energetic", ev
    print(f"  sad → energetic → DRIFT (dist {ev.distance:.3f}, "
          f"{ev.from_emotion}→{ev.to_emotion}) ✓")

    # 3. Re-anchored: staying energetic does NOT keep firing.
    ev2 = d.observe(peak("energetic"))
    assert not ev2.drifted, ev2.distance
    print(f"  sustained 'energetic' → no re-fire (dist {ev2.distance:.3f}) ✓")

    # 4. Warm-up: very first reading never drifts.
    d2 = EmotionDriftDetector(min_history=2)
    first = d2.observe(peak("angry"))
    assert not first.drifted
    print("  first reading (warm-up) → no drift ✓")

    # 5. JS divergence basic properties.
    assert abs(js_divergence(peak("calm"), peak("calm"))) < 1e-9
    assert js_divergence(peak("calm"), peak("energetic")) > 0.5
    print("  JS(self,self)=0, JS(calm,energetic)>0.5 ✓")

    print("-" * 55)
    print("✅ All drift-detector self-tests passed.")


if __name__ == "__main__":
    _selftest()
