"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 7 · Transition Selector
═══════════════════════════════════════════════════════════════════
When drift is detected, we don't just grab the top emotion match — we grab the
one that *also mixes well* out of the current track. This selector scores every
candidate for the new mood on three axes and returns the best blend:

  • emotion   — how well it fits the new mood      (FeatureStore cosine score)
  • harmonic  — Camelot compatibility with the now-playing key  (0–1)
  • tempo     — BPM closeness to the now-playing track          (0–1)

combined = wₑ·emotion + w_h·harmonic + w_t·tempo   (weights tunable, default
0.50 / 0.30 / 0.20). This is exactly how a DJ chooses the next record: right
vibe, compatible key, mixable tempo — now automated over the 1.2M store.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
_P2 = _ROOT / "phase_2_music_data"
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _P2, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from feature_store import FeatureStore                        # Phase 2  # noqa: E402

from camelot import compatibility_score                       # noqa: E402
from harmonic import HarmonicIndex                            # noqa: E402


@dataclass
class TransitionCandidate:
    """A scored next-song option."""
    track_id: str
    name: str
    artist: str
    camelot: Optional[str]
    bpm: float
    emotion_score: float
    harmonic_score: float
    tempo_score: float
    combined_score: float

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id, "name": self.name, "artist": self.artist,
            "camelot": self.camelot, "bpm": round(self.bpm, 1),
            "emotion": round(self.emotion_score, 3),
            "harmonic": round(self.harmonic_score, 3),
            "tempo": round(self.tempo_score, 3),
            "combined": round(self.combined_score, 3),
        }


class TransitionSelector:
    """Scores emotion-matched candidates by harmonic + tempo mixability."""

    def __init__(
        self,
        store: FeatureStore,
        harmonic_index: HarmonicIndex,
        mix_weight: float = 0.15,      # how much harmony/tempo can sway a tie
        w_harmonic: float = 0.60,      # split of the mixability bonus …
        w_tempo: float = 0.40,         # … between key and tempo
        bpm_falloff: float = 16.0,     # BPM gap at which tempo_score hits ~0
    ) -> None:
        self.store = store
        self.harmonic = harmonic_index
        split = w_harmonic + w_tempo
        self.w_harmonic = w_harmonic / split
        self.w_tempo = w_tempo / split
        self.mix_weight = mix_weight
        self.bpm_falloff = max(1.0, bpm_falloff)

    def _tempo_score(self, bpm_a: float, bpm_b: float) -> float:
        if bpm_a <= 0 or bpm_b <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(bpm_a - bpm_b) / self.bpm_falloff)

    def rank_candidates(
        self,
        current_track_id: str,
        target_emotion: str,
        k_candidates: int = 40,
        exclude: Optional[set[str]] = None,
    ) -> list[TransitionCandidate]:
        """Return emotion-matched candidates, scored + sorted best-first."""
        exclude = set(exclude or set())
        exclude.add(str(current_track_id))

        cur = self.harmonic.get(current_track_id)
        cur_camelot = cur.camelot if cur else None
        cur_bpm = cur.bpm if cur else 0.0

        results = self.store.search_emotion(target_emotion, k=k_candidates)

        scored: list[TransitionCandidate] = []
        for r in results:
            if r.track_id in exclude:
                continue
            prof = self.harmonic.get(r.track_id)
            if prof is None:              # no key/tempo → can't assess a mix
                continue
            h = compatibility_score(cur_camelot, prof.camelot)
            t = self._tempo_score(cur_bpm, prof.bpm)
            # Emotion fit dominates; harmony+tempo add a bounded bonus that only
            # decides between candidates of comparable mood fit.
            combined = r.score + self.mix_weight * (
                self.w_harmonic * h + self.w_tempo * t)
            scored.append(TransitionCandidate(
                track_id=r.track_id, name=r.name, artist=r.artist,
                camelot=prof.camelot, bpm=prof.bpm,
                emotion_score=r.score, harmonic_score=h, tempo_score=t,
                combined_score=combined,
            ))

        scored.sort(key=lambda c: c.combined_score, reverse=True)
        return scored

    def select_next(
        self,
        current_track_id: str,
        target_emotion: str,
        k_candidates: int = 40,
        exclude: Optional[set[str]] = None,
    ) -> Optional[TransitionCandidate]:
        """Best next track to mix into, or None if nothing suitable."""
        ranked = self.rank_candidates(current_track_id, target_emotion,
                                      k_candidates, exclude)
        return ranked[0] if ranked else None


# ─────────────────────────────────────────────────────────────
# Self-test — key isolated so harmonic score decides the winner
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Transition selector self-test")
    print("-" * 55)

    from schema import Song

    def mk(tid, key, mode, tempo_bpm, calm_like=True):
        # Identical *match* features across candidates (so emotion_score ties);
        # only key/mode differ → harmonic score is the deciding axis.
        rf = {"danceability": 0.30, "energy": 0.22, "key": key, "loudness": 0.4,
              "mode": mode, "speechiness": 0.1, "acousticness": 0.80,
              "instrumentalness": 0.40, "liveness": 0.1, "valence": 0.50,
              "tempo": tempo_bpm}
        return Song(tid, f"cand-{tid}", ["Artist"], 2000, "2000-01-01", rf)

    current = mk("cur", 0, 1, 80.0)     # C major → 8B, 80 BPM
    cand_rel = mk("A", 9, 0, 80.0)      # A minor → 8A  (relative,  0.80)
    cand_adj = mk("B", 7, 1, 80.0)      # G major → 9B  (adjacent,  0.85)
    cand_clash = mk("C", 6, 1, 80.0)    # F# major → 2B (clash,     0.00)
    songs = [current, cand_rel, cand_adj, cand_clash]

    store = FeatureStore().build_from_songs(songs)
    hidx = HarmonicIndex().build_from_songs(songs)
    sel = TransitionSelector(store, hidx)

    ranked = sel.rank_candidates("cur", "calm", k_candidates=10)
    ids = [c.track_id for c in ranked]
    assert set(ids) == {"A", "B", "C"}, ids            # current excluded
    # Emotion + tempo tie across all three, so harmonic ranks them: B > A > C.
    assert ids[0] == "B", ids                           # 9B adjacent = best mix
    assert ids[-1] == "C", ids                          # 2B clash = worst
    best = sel.select_next("cur", "calm")
    assert best.track_id == "B" and best.camelot == "9B"
    print(f"  from 8B/80bpm → ranked {[(c.track_id, c.camelot, round(c.harmonic_score,2)) for c in ranked]}")
    print(f"  select_next → {best.track_id} ({best.camelot}), combined {best.combined_score:.3f}")

    # exclude already-played tracks.
    best2 = sel.select_next("cur", "calm", exclude={"B"})
    assert best2.track_id == "A", best2.track_id
    print(f"  exclude played 'B' → next best {best2.track_id} ({best2.camelot}) ✓")

    print("-" * 55)
    print("✅ All transition-selector self-tests passed.")


if __name__ == "__main__":
    _selftest()
