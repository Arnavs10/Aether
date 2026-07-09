"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 7 · Transition Demo
═══════════════════════════════════════════════════════════════════
Simulate the Live Emotion Player: a song starts, the listener's mood is read
repeatedly, and the engine holds or cross-fades into a compatible next track as
the mood drifts. Then show the Playlist Curator's crossfade plan for a curated
set.

    python phase_7_drift_crossfade/demo_transition.py

Uses a tiny built-in library (no data files). This is the decision brain; the
actual audio crossfade is rendered by Phase 8's Web Audio player.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import AETHER_EMOTIONS                            # noqa: E402
from engine import LiveTransitionEngine, _build_songs        # noqa: E402


def peak(emotion: str, strength: float = 0.85) -> np.ndarray:
    n = len(AETHER_EMOTIONS)
    v = np.full(n, (1 - strength) / (n - 1))
    v[AETHER_EMOTIONS.index(emotion)] = strength
    return v


def main() -> None:
    engine = LiveTransitionEngine.from_songs(_build_songs())

    print("=" * 68)
    print("LIVE EMOTION PLAYER  —  the listener's mood steers the music")
    print("=" * 68)

    engine.start("sad1")
    print(f"\n▶ now playing: sad1  ({engine.harmonic.get('sad1').camelot}, "
          f"{engine.harmonic.get('sad1').bpm:.0f} BPM)\n")

    # A mood journey the listener expresses over time.
    journey = ["sad", "sad", "energetic", "energetic", "calm", "calm"]
    for i, mood in enumerate(journey, 1):
        d = engine.observe(peak(mood))
        if d.triggered:
            print(f"  [{i}] felt '{mood}'  →  ✦ TRANSITION  "
                  f"({d.drift.from_emotion}→{d.drift.to_emotion}, "
                  f"drift {d.drift.distance:.2f})")
            print(f"        next: {d.next.track_id}  {d.next.camelot}  "
                  f"{d.next.bpm:.0f} BPM   (emotion {d.next.emotion_score:.2f}, "
                  f"harmony {d.next.harmonic_score:.2f})")
            print(f"        crossfade: {d.crossfade.duration_s:.1f}s "
                  f"{d.crossfade.curve}  (~{d.crossfade.beats:.0f} beats)")
        else:
            print(f"  [{i}] felt '{mood}'  →  hold  ({d.reason})")
    print()

    print("=" * 68)
    print("PLAYLIST CURATOR  —  smooth crossfades across a curated set")
    print("=" * 68)
    curated = ["sad1", "sad2", "ca1", "en1"]
    plans = engine.plan_playlist(curated)
    print(f"\n  playlist: {' → '.join(curated)}\n")
    for p in plans:
        print(f"    {p.out_track_id} → {p.in_track_id}: "
              f"{p.duration_s:.1f}s {p.curve}  ({p.notes})")


if __name__ == "__main__":
    main()
