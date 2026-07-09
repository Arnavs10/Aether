"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 7 · Harmonic Index
═══════════════════════════════════════════════════════════════════
The FeatureStore (Phase 2) keeps only the 6 matching features — it drops
``key`` and ``mode``. Harmonic mixing needs them, so this module builds a light
side-index: ``track_id → HarmonicProfile`` (Camelot code, real BPM, energy),
read from the same source ``Song`` objects that built the store.

It's a one-time pass parallel to store construction — in production you build
both from the same song stream; in tests you build both from a handful of
songs. Lookups are O(1) dict hits, so the transition selector can check any
candidate's harmonic profile instantly.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_ROOT = Path(__file__).resolve().parent.parent
_P2 = _ROOT / "phase_2_music_data"
for _p in (_ROOT, _P2):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schema import Song                                       # Phase 2  # noqa: E402

from camelot import to_camelot                                # noqa: E402


@dataclass(frozen=True)
class HarmonicProfile:
    """A track's mix-relevant fingerprint."""
    track_id: str
    camelot: Optional[str]      # e.g. "8B"; None if key unknown
    bpm: float                  # real tempo in BPM (not normalized)
    energy: float               # 0–1

    @property
    def has_key(self) -> bool:
        return self.camelot is not None


class HarmonicIndex:
    """``track_id → HarmonicProfile`` for harmonic + tempo transition checks."""

    def __init__(self) -> None:
        self._by_id: dict[str, HarmonicProfile] = {}

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, track_id: str) -> bool:
        return track_id in self._by_id

    def build_from_songs(self, songs: Iterable[Song]) -> "HarmonicIndex":
        """Populate the index from source songs (reads key/mode/tempo/energy)."""
        for s in songs:
            rf = getattr(s, "raw_features", {}) or {}
            key = rf.get("key", -1)
            mode = rf.get("mode", -1)
            profile = HarmonicProfile(
                track_id=str(s.track_id),
                camelot=to_camelot(key, mode),
                bpm=float(rf.get("tempo", 0.0) or 0.0),
                energy=float(rf.get("energy", 0.0) or 0.0),
            )
            self._by_id[str(s.track_id)] = profile
        return self

    def get(self, track_id: str) -> Optional[HarmonicProfile]:
        """Return the profile for a track id, or None if not indexed."""
        return self._by_id.get(str(track_id))

    def add(self, profile: HarmonicProfile) -> None:
        """Insert/replace a single profile (useful for freshness-layer tracks)."""
        self._by_id[profile.track_id] = profile


# ─────────────────────────────────────────────────────────────
# Self-test — tiny songs, no data files
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Harmonic index self-test")
    print("-" * 55)

    def mk(tid, key, mode, tempo, energy):
        rf = {f: 0.5 for f in
              ["danceability", "energy", "key", "loudness", "mode",
               "speechiness", "acousticness", "instrumentalness",
               "liveness", "valence", "tempo"]}
        rf.update(key=key, mode=mode, tempo=tempo, energy=energy)
        return Song(tid, f"song {tid}", ["Artist"], 2000, "2000-01-01", rf)

    songs = [
        mk("t1", 0, 1, 120.0, 0.60),   # C major  → 8B
        mk("t2", 9, 0, 122.0, 0.55),   # A minor  → 8A
        mk("t3", 7, 1, 128.0, 0.70),   # G major  → 9B
        mk("t4", -1, 1, 100.0, 0.40),  # unknown key → camelot None
    ]
    idx = HarmonicIndex().build_from_songs(songs)

    assert len(idx) == 4
    assert idx.get("t1").camelot == "8B" and idx.get("t1").bpm == 120.0
    assert idx.get("t2").camelot == "8A"
    assert idx.get("t3").camelot == "9B" and idx.get("t3").energy == 0.70
    assert idx.get("t4").camelot is None and not idx.get("t4").has_key
    assert idx.get("nope") is None and "t1" in idx
    print(f"  indexed {len(idx)} tracks; "
          f"t1→{idx.get('t1').camelot} t2→{idx.get('t2').camelot} "
          f"t3→{idx.get('t3').camelot} t4→{idx.get('t4').camelot}")

    print("-" * 55)
    print("✅ All harmonic-index self-tests passed.")


if __name__ == "__main__":
    _selftest()
