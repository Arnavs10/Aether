"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 7 · Transition Engine
═══════════════════════════════════════════════════════════════════
The orchestrator that both Aether features call:

FUN FEATURE — Live Emotion Player
    A song is playing. The listener keeps expressing how they feel; each reading
    goes to `observe(distribution)`. The engine runs drift detection, and the
    moment the mood shifts far enough it picks a harmonically- and tempo-
    compatible next track for the NEW mood and returns a crossfade plan. The
    Phase 8 Web Audio player renders that fade. Repeat → an endless, self-
    steering set that follows the listener's emotions.

MAIN FEATURE — Playlist Curator
    A finished playlist just needs smooth playback. `plan_playlist(track_ids)`
    returns the crossfade plan between each consecutive pair — same engine, no
    drift logic — so the curated set plays gapless and gliding, not hard-cut.

This is the decision brain only; all audio rendering lives in Phase 8.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_P2 = _ROOT / "phase_2_music_data"
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _P2, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schema import Song                                       # Phase 2  # noqa: E402
from feature_store import FeatureStore                        # Phase 2  # noqa: E402

from harmonic import HarmonicIndex, HarmonicProfile           # noqa: E402
from drift import EmotionDriftDetector, DriftEvent            # noqa: E402
from transition import TransitionSelector, TransitionCandidate  # noqa: E402
from crossfade import CrossfadePlanner, CrossfadePlan         # noqa: E402


@dataclass
class TransitionDecision:
    """What the engine decided after one emotion reading."""
    triggered: bool                     # did a live transition fire?
    reason: str
    drift: DriftEvent
    next: Optional[TransitionCandidate] = None
    crossfade: Optional[CrossfadePlan] = None

    def as_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "drift": {
                "drifted": self.drift.drifted,
                "distance": round(self.drift.distance, 3),
                "from": self.drift.from_emotion,
                "to": self.drift.to_emotion,
            },
            "next": self.next.as_dict() if self.next else None,
            "crossfade": self.crossfade.as_dict() if self.crossfade else None,
        }


class LiveTransitionEngine:
    """Drift-driven live song transitions + curated-playlist crossfades."""

    def __init__(
        self,
        store: FeatureStore,
        harmonic_index: HarmonicIndex,
        selector: Optional[TransitionSelector] = None,
        planner: Optional[CrossfadePlanner] = None,
        detector: Optional[EmotionDriftDetector] = None,
    ) -> None:
        self.store = store
        self.harmonic = harmonic_index
        self.selector = selector or TransitionSelector(store, harmonic_index)
        self.planner = planner or CrossfadePlanner()
        self.detector = detector or EmotionDriftDetector()
        self.current_track_id: Optional[str] = None
        self.played: set[str] = set()

    # ── construction from a song stream (builds store + harmonic index) ──
    @classmethod
    def from_songs(cls, songs: Iterable[Song], **kwargs) -> "LiveTransitionEngine":
        songs = list(songs)
        store = FeatureStore().build_from_songs(songs)
        hidx = HarmonicIndex().build_from_songs(songs)
        return cls(store, hidx, **kwargs)

    # ── FUN FEATURE ──
    def start(self, track_id: str) -> None:
        """Begin a live session on a starting track."""
        self.current_track_id = str(track_id)
        self.played = {str(track_id)}
        self.detector.reset()

    def observe(self, distribution: np.ndarray) -> TransitionDecision:
        """
        Feed one emotion reading. If the mood has drifted, choose a compatible
        next track for the new mood and return a crossfade plan.
        """
        if self.current_track_id is None:
            raise RuntimeError("call start(track_id) before observe().")

        ev = self.detector.observe(distribution)
        if not ev.drifted:
            return TransitionDecision(False, "no drift — hold current track", ev)

        nxt = self.selector.select_next(
            self.current_track_id, ev.to_emotion, exclude=self.played)
        if nxt is None:
            return TransitionDecision(False, f"drift to {ev.to_emotion}, "
                                      "but no compatible track found", ev)

        out_prof = self.harmonic.get(self.current_track_id)
        in_prof = self.harmonic.get(nxt.track_id)
        crossfade = self.planner.plan(
            out_track_id=self.current_track_id, in_track_id=nxt.track_id,
            out_bpm=out_prof.bpm if out_prof else 0.0, in_bpm=in_prof.bpm,
            out_energy=out_prof.energy if out_prof else 0.5,
            in_energy=in_prof.energy,
        )

        # advance session state
        self.current_track_id = nxt.track_id
        self.played.add(nxt.track_id)

        return TransitionDecision(
            True, f"mood drifted {ev.from_emotion}→{ev.to_emotion} "
            f"(dist {ev.distance:.2f}) — mixing into {nxt.camelot}",
            ev, next=nxt, crossfade=crossfade,
        )

    # ── MAIN FEATURE ──
    def plan_playlist(self, track_ids: list[str]) -> list[CrossfadePlan]:
        """Crossfade plans between each consecutive pair of a curated playlist."""
        plans: list[CrossfadePlan] = []
        for a_id, b_id in zip(track_ids, track_ids[1:]):
            a, b = self.harmonic.get(a_id), self.harmonic.get(b_id)
            if a is None or b is None:
                continue
            plans.append(self.planner.plan_profiles(a, b))
        return plans


# ─────────────────────────────────────────────────────────────
# Self-test — a small multi-emotion library, end to end
# ─────────────────────────────────────────────────────────────
def _build_songs() -> list[Song]:
    def mk(tid, key, mode, tempo, energy, valence, extra=None):
        rf = {"danceability": 0.4, "energy": energy, "key": key, "loudness": 0.5,
              "mode": mode, "speechiness": 0.1, "acousticness": 0.5,
              "instrumentalness": 0.3, "liveness": 0.1, "valence": valence,
              "tempo": tempo}
        if extra:
            rf.update(extra)
        return Song(tid, f"song-{tid}", [f"Artist{tid}"], 2000, "2000-01-01", rf)

    return [
        # sad-ish, low energy
        mk("sad1", 9, 0, 72, 0.20, 0.15),   # A minor  8A
        mk("sad2", 2, 0, 70, 0.22, 0.18),   # D minor  7A
        # energetic, high energy, various keys
        mk("en1", 7, 1, 128, 0.85, 0.80),   # G major  9B
        mk("en2", 0, 1, 126, 0.82, 0.78),   # C major  8B
        mk("en3", 5, 1, 130, 0.88, 0.75),   # F major  7B
        # calm, mid
        mk("ca1", 4, 0, 80, 0.25, 0.50),    # E minor  9A
        mk("ca2", 11, 1, 82, 0.28, 0.52),   # B major  1B
    ]


def _selftest() -> None:
    print("Transition engine self-test")
    print("-" * 55)

    from config import AETHER_EMOTIONS
    n = len(AETHER_EMOTIONS)

    def peak(emo, s=0.85):
        v = np.full(n, (1 - s) / (n - 1))
        v[AETHER_EMOTIONS.index(emo)] = s
        return v

    engine = LiveTransitionEngine.from_songs(_build_songs())

    # FUN FEATURE: start on a sad track, hold through sad, then drift to energetic.
    engine.start("sad1")
    d0 = engine.observe(peak("sad"))
    d1 = engine.observe(peak("sad"))
    assert not d0.triggered and not d1.triggered, "sad stream should hold"
    print(f"  start sad1 · stable 'sad' → hold ({d1.reason})")

    d2 = engine.observe(peak("energetic"))
    assert d2.triggered, d2.reason
    assert d2.drift.from_emotion == "sad" and d2.drift.to_emotion == "energetic"
    assert d2.next is not None and d2.next.track_id in {"en1", "en2", "en3"}
    assert d2.crossfade is not None and 3.0 <= d2.crossfade.duration_s <= 5.0
    print(f"  → DRIFT: {d2.reason}")
    print(f"    next: {d2.next.track_id} ({d2.next.camelot}, {d2.next.bpm:.0f}bpm, "
          f"combined {d2.next.combined_score:.2f})")
    print(f"    crossfade: {d2.crossfade.duration_s:.1f}s {d2.crossfade.curve} "
          f"({d2.crossfade.beats:.1f} beats)")

    # played-set: the engine won't pick a track it already used.
    assert engine.current_track_id == d2.next.track_id
    assert d2.next.track_id in engine.played

    # drift again energetic → calm
    d3 = engine.observe(peak("calm"))
    assert d3.triggered and d3.next.track_id in {"ca1", "ca2"}, d3.reason
    print(f"  → DRIFT: energetic→calm, next {d3.next.track_id} ({d3.next.camelot})")

    # MAIN FEATURE: crossfade plan for a curated playlist.
    plans = engine.plan_playlist(["sad1", "sad2", "ca1", "en1"])
    assert len(plans) == 3, len(plans)
    assert all(3.0 <= p.duration_s <= 5.0 for p in plans)
    print(f"  plan_playlist(4 tracks) → {len(plans)} crossfades: "
          f"{[round(p.duration_s,1) for p in plans]}s")

    print("-" * 55)
    print("✅ All transition-engine self-tests passed.")


if __name__ == "__main__":
    _selftest()
