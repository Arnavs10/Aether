"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 6 · Arc Planner
═══════════════════════════════════════════════════════════════════
The "autonomous playlist-arc planning" core the agent uses when a request
describes an emotional JOURNEY ("take me from anxious to calm") rather than a
single mood.

Idea (interview-defensible)
---------------------------
Every Aether emotion has a position in valence–arousal space — we read it from
the SAME normalized feature target the matcher scores against (valence and
energy dimensions). Planning an arc from a start emotion to a target emotion is
then literally finding a smooth PATH between two points: we drop in the nuanced
emotions that lie near the straight line between start and target, ordered by
how far along that line they sit. The result is a short sequence of waypoint
emotions the playlist moves through — e.g. anxious → focused → calm — instead of
an abrupt jump.

This mirrors the music-therapy "iso principle": meet the listener where they
are, then guide them, step by step, toward the goal.

Pure, deterministic, offline — no deps beyond NumPy and config.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Path bootstrap: root config + Phase 2 normalizer/schema ──
_ROOT = Path(__file__).resolve().parent.parent
_P2 = _ROOT / "phase_2_music_data"
for _p in (_ROOT, _P2):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from config import AETHER_EMOTIONS, EMOTION_MUSIC_TARGETS  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    AETHER_EMOTIONS, EMOTION_MUSIC_TARGETS = [], {}

try:
    from feature_store import normalize_emotion_target        # type: ignore  # noqa: E402
    from schema import MATCH_FEATURES                          # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    normalize_emotion_target = None                            # type: ignore
    MATCH_FEATURES = ["tempo", "energy", "valence", "danceability",
                      "acousticness", "instrumentalness"]

_ENERGY_I = MATCH_FEATURES.index("energy")
_VALENCE_I = MATCH_FEATURES.index("valence")


# ──────────────────────────────────────────────
# Emotion coordinates in (valence, energy) space
# ──────────────────────────────────────────────
def _coords() -> dict[str, np.ndarray]:
    """Map each emotion → normalized (valence, energy) point."""
    out: dict[str, np.ndarray] = {}
    for emo, target in EMOTION_MUSIC_TARGETS.items():
        if normalize_emotion_target is not None:
            v = normalize_emotion_target(target)
            out[emo] = np.array([v[_VALENCE_I], v[_ENERGY_I]], dtype=np.float64)
        else:  # pragma: no cover — degrade using raw 0–1 features
            out[emo] = np.array([target.get("valence", 0.5),
                                 target.get("energy", 0.5)], dtype=np.float64)
    return out


_COORDS = _coords()


@dataclass
class ArcPlan:
    """A planned emotional journey."""
    waypoints: list[str]                      # ordered emotions, start → target
    direction: str                            # ascending | descending | steady (by energy)
    notes: list[str] = field(default_factory=list)

    @property
    def is_journey(self) -> bool:
        return len(self.waypoints) > 1

    def describe(self) -> str:
        return " → ".join(self.waypoints)


# ──────────────────────────────────────────────
# Planning
# ──────────────────────────────────────────────
def energy_of(emotion: str) -> float:
    """Normalized energy (arousal) coordinate of an emotion, 0.5 if unknown."""
    c = _COORDS.get(emotion)
    return float(c[1]) if c is not None else 0.5


def _direction(start: str, target: str) -> str:
    """Energy trend of the journey (drives arc shape + per-segment sequencing)."""
    de = energy_of(target) - energy_of(start)
    if de > 0.08:
        return "ascending"
    if de < -0.08:
        return "descending"
    return "steady"


def plan_arc(start: str, target: str, max_waypoints: int = 4) -> ArcPlan:
    """
    Plan a smooth emotional arc from `start` to `target`.

    Args:
        start: the emotion the listener is in now.
        target: the emotion to guide them toward.
        max_waypoints: hard cap on total waypoints (incl. start & target).

    Returns:
        An ArcPlan whose waypoints trace start → …intermediates… → target,
        with intermediates chosen as the nuanced emotions lying nearest the
        straight line between the two endpoints, ordered along that line.
    """
    if start not in _COORDS or target not in _COORDS:
        # Unknown emotion(s): fall back to the endpoints we have.
        wp = [e for e in (start, target) if e in _COORDS] or [start]
        uniq = list(dict.fromkeys(wp))
        return ArcPlan(uniq, _direction(start, target),
                       notes=["unknown emotion → endpoints only"])

    if start == target:
        return ArcPlan([start], "steady", notes=["single mood (no journey)"])

    s, t = _COORDS[start], _COORDS[target]
    d = t - s
    L2 = float(d @ d) or 1e-9

    # Score every other emotion by (a) projection u along s→t and (b) how far
    # off the line it sits. Keep the ones between the endpoints, close to line.
    candidates: list[tuple[float, float, str]] = []   # (u, perp, emotion)
    for emo, c in _COORDS.items():
        if emo in (start, target):
            continue
        u = float((c - s) @ d) / L2
        if not (0.12 < u < 0.88):          # strictly between, with margin
            continue
        perp = float(np.linalg.norm((c - s) - u * d))
        if perp <= 0.22:                    # near the line (tunable corridor)
            candidates.append((u, perp, emo))

    # Prefer the closest-to-line candidates, then order the chosen ones by u so
    # the journey is monotonic in progress.
    candidates.sort(key=lambda x: x[1])                # by perpendicular distance
    keep = candidates[: max(0, max_waypoints - 2)]
    keep.sort(key=lambda x: x[0])                      # re-order by progress u
    intermediates = [e for _, _, e in keep]

    waypoints = [start, *intermediates, target]
    notes = [f"path in valence–arousal space, {len(intermediates)} intermediate(s)"]
    return ArcPlan(waypoints, _direction(start, target), notes=notes)


# ─────────────────────────────────────────────────────────────
# Self-test — pure, offline
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Arc planner self-test")
    print("-" * 55)

    assert _COORDS, "no emotion coordinates (config missing?)"

    # 1. Journey inserts sensible, ordered intermediates.
    p = plan_arc("anxious", "calm")
    assert p.waypoints[0] == "anxious" and p.waypoints[-1] == "calm", p.waypoints
    assert p.is_journey, p.waypoints
    # monotonic progress: each waypoint's projection u is non-decreasing.
    s, t = _COORDS["anxious"], _COORDS["calm"]
    d = t - s; L2 = float(d @ d)
    us = [float((_COORDS[w] - s) @ d) / L2 for w in p.waypoints]
    assert us == sorted(us), us
    print(f"  anxious→calm  → {p.describe()}  [{p.direction}]")

    # 2. Direction reflects the energy trend (anxious↑ → calm↓ = descending).
    assert p.direction == "descending", p.direction
    assert plan_arc("sad", "energetic").direction == "ascending"
    print(f"  sad→energetic → {plan_arc('sad','energetic').describe()} "
          f"[{plan_arc('sad','energetic').direction}]")

    # 3. Same start/target → single mood, no journey.
    p1 = plan_arc("happy", "happy")
    assert p1.waypoints == ["happy"] and not p1.is_journey
    print(f"  happy→happy   → {p1.describe()} (single mood)")

    # 4. max_waypoints is respected.
    p2 = plan_arc("melancholic", "energetic", max_waypoints=3)
    assert len(p2.waypoints) <= 3, p2.waypoints
    print(f"  melancholic→energetic (cap 3) → {p2.describe()}")

    # 5. All waypoints are valid Aether emotions.
    assert all(w in AETHER_EMOTIONS for w in p.waypoints)

    print("-" * 55)
    print("✅ All arc-planner self-tests passed.")


if __name__ == "__main__":
    _selftest()
