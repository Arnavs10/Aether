"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 3: Emotion → Music Matcher
═══════════════════════════════════════════════════════════════════

The retrieval core. Given a MatchIntent (from intent.py) and the Phase 2
music feature store, it produces a ranked list of songs.

Two retrieval modes (chosen upstream by intent.py):

  • BLEND  → average the emotions' feature-targets into ONE target, then do a
             single nearest-neighbor search. Produces songs matching a single
             nuanced mood ("nostalgic but hopeful").

  • MIX    → search EACH emotion's target separately, then interleave the
             per-emotion result lists (round-robin by weight). Produces a
             playlist with DISTINCT songs of each emotion
             ("some nostalgic and some hopeful").

  • SINGLE → one emotion, one search (a degenerate blend).

On top of raw nearest-neighbor, the matcher adds two production concerns:

  • Artist diversity — cap how many songs one artist can contribute, so a
    playlist isn't 15 tracks by the same band.
  • De-duplication — never return the same track twice (matters especially in
    MIX mode, where a song could match two emotions).

The matcher stays PURE retrieval: it returns ranked SearchResult-like rows.
Final assembly (intensity scaling, ordering, target length) is playlist.py.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from intent import MatchIntent, MODE_BLEND, MODE_MIX, MODE_SINGLE

# Feature store + normalization live in Phase 2. We import them so the matcher
# speaks the exact same 6-dim match space the store was built in.
import sys
from pathlib import Path
_P2 = Path(__file__).resolve().parent.parent / "phase_2_music_data"
if _P2.exists() and str(_P2) not in sys.path:
    sys.path.insert(0, str(_P2))

from feature_store import FeatureStore, normalize_emotion_target, SearchResult  # noqa: E402

try:
    from config import EMOTION_MUSIC_TARGETS  # type: ignore
except Exception:  # pragma: no cover
    EMOTION_MUSIC_TARGETS = {}


@dataclass
class MatchedSong:
    """A song returned by the matcher, with provenance for transparency."""
    track_id: str
    name: str
    artist: str
    year: Optional[int]
    score: float
    source_emotion: str   # which emotion produced this match (mix mode)

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id, "name": self.name, "artist": self.artist,
            "year": self.year, "score": round(self.score, 4),
            "source_emotion": self.source_emotion,
        }


class Matcher:
    """Emotion→music retrieval over a Phase 2 FeatureStore."""

    def __init__(self, store: FeatureStore, max_per_artist: int = 2):
        """
        Args:
            store: a loaded Phase 2 FeatureStore.
            max_per_artist: cap on songs from a single artist in one result set
                (diversity). Set 0 to disable.
        """
        if len(store) == 0:
            raise ValueError("FeatureStore is empty — build it first (build_store.py).")
        self.store = store
        self.max_per_artist = max_per_artist

    # ─────────────────────────────────────────────────────────
    def _emotion_target(self, emotion: str) -> np.ndarray:
        """Normalized 6-dim target vector for an emotion (from config)."""
        if emotion not in EMOTION_MUSIC_TARGETS:
            raise KeyError(
                f"No music target for emotion {emotion!r}. Present: "
                f"{sorted(EMOTION_MUSIC_TARGETS)}"
            )
        return normalize_emotion_target(EMOTION_MUSIC_TARGETS[emotion])

    def _blended_target(self, emotions: list[str], weights: list[float]) -> np.ndarray:
        """Weighted average of several emotions' target vectors → one target."""
        vecs = np.stack([self._emotion_target(e) for e in emotions])   # (k, 6)
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        w = w / (w.sum() + 1e-8)
        return (vecs * w).sum(axis=0).astype(np.float32)

    def _apply_diversity(
        self, results: list[SearchResult], limit: int, source_emotion: str,
    ) -> list[MatchedSong]:
        """Dedup by track_id + cap songs per artist, keeping best scores first."""
        seen_tracks: set[str] = set()
        artist_counts: dict[str, int] = {}
        out: list[MatchedSong] = []
        for r in results:
            if r.track_id in seen_tracks:
                continue
            if self.max_per_artist:
                c = artist_counts.get(r.artist, 0)
                if c >= self.max_per_artist:
                    continue
                artist_counts[r.artist] = c + 1
            seen_tracks.add(r.track_id)
            out.append(MatchedSong(
                track_id=r.track_id, name=r.name, artist=r.artist,
                year=r.year, score=r.score, source_emotion=source_emotion,
            ))
            if len(out) >= limit:
                break
        return out

    # ─────────────────────────────────────────────────────────
    def match(
        self,
        intent: MatchIntent,
        limit: int = 20,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> list[MatchedSong]:
        """Retrieve songs for a parsed intent.

        Args:
            intent: MatchIntent from intent.parse_intent().
            limit: total number of songs to return.
            year_min / year_max: optional release-year filter.

        Returns:
            Ranked list of MatchedSong (best first for blend/single; interleaved
            for mix).
        """
        if intent.mode in (MODE_BLEND, MODE_SINGLE):
            return self._match_blend(intent, limit, year_min, year_max)
        if intent.mode == MODE_MIX:
            return self._match_mix(intent, limit, year_min, year_max)
        raise ValueError(f"Unknown intent mode: {intent.mode!r}")

    # ── blend / single ──
    def _match_blend(
        self, intent: MatchIntent, limit: int,
        year_min: Optional[int], year_max: Optional[int],
    ) -> list[MatchedSong]:
        target = self._blended_target(intent.emotions, intent.weights)
        # Over-fetch so diversity filtering still yields `limit` songs.
        raw = self.store.search(target, k=limit * 5,
                                year_min=year_min, year_max=year_max)
        label = "+".join(intent.emotions)
        return self._apply_diversity(raw, limit, source_emotion=label)

    # ── mix ──
    def _match_mix(
        self, intent: MatchIntent, limit: int,
        year_min: Optional[int], year_max: Optional[int],
    ) -> list[MatchedSong]:
        """Search each emotion separately, then interleave by weight.

        Each emotion contributes a share of `limit` proportional to its weight
        (at least 1 each). Results are interleaved round-robin so the playlist
        alternates moods, and de-duplicated globally so a song matching two
        emotions appears once (under its stronger source).
        """
        k = len(intent.emotions)
        # Per-emotion quota from weights (at least 1 each, summing to `limit`).
        quotas = _allocate(intent.weights, limit)

        # Retrieve a diversified pool per emotion (over-fetch for interleave).
        pools: list[list[MatchedSong]] = []
        for emo, quota in zip(intent.emotions, quotas):
            target = self._emotion_target(emo)
            raw = self.store.search(target, k=max(quota * 5, 20),
                                    year_min=year_min, year_max=year_max)
            pools.append(self._apply_diversity(raw, quota * 3, source_emotion=emo))

        # Round-robin interleave, global dedup, respect quotas + total limit.
        out: list[MatchedSong] = []
        seen: set[str] = set()
        taken = [0] * k
        idxs = [0] * k
        while len(out) < limit and any(idxs[i] < len(pools[i]) for i in range(k)):
            for i in range(k):
                if len(out) >= limit:
                    break
                if taken[i] >= quotas[i]:
                    continue
                # advance to next unseen song in this pool
                while idxs[i] < len(pools[i]) and pools[i][idxs[i]].track_id in seen:
                    idxs[i] += 1
                if idxs[i] >= len(pools[i]):
                    continue
                song = pools[i][idxs[i]]
                idxs[i] += 1
                seen.add(song.track_id)
                out.append(song)
                taken[i] += 1
        return out


def _allocate(weights: list[float], total: int) -> list[int]:
    """Split `total` into per-emotion quotas proportional to weights (≥1 each)."""
    k = len(weights)
    if k == 0:
        return []
    if k == 1:
        return [total]
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-8)
    raw = w * total
    quotas = np.maximum(1, np.floor(raw)).astype(int)
    # Fix rounding so the quotas sum to `total`.
    diff = total - int(quotas.sum())
    order = np.argsort(-(raw - np.floor(raw)))  # largest fractional part first
    i = 0
    while diff != 0 and k > 0:
        j = int(order[i % k])
        if diff > 0:
            quotas[j] += 1; diff -= 1
        elif quotas[j] > 1:
            quotas[j] -= 1; diff += 1
        i += 1
        if i > 4 * total:  # safety
            break
    return [int(x) for x in quotas]


# ─────────────────────────────────────────────────────────────
# Self-test  (uses a tiny in-memory store; no CSV needed)
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Matcher self-test")
    print("-" * 55)

    # Build a tiny synthetic store spanning clear emotional corners.
    _p2 = str(_P2)
    from schema import Song  # from phase_2 path

    def mk(tid, name, artist, feats, year=2000):
        base = {f: 0.5 for f in
                ["danceability", "energy", "key", "loudness", "mode",
                 "speechiness", "acousticness", "instrumentalness",
                 "liveness", "valence", "tempo"]}
        base.update(feats)
        return Song(tid, name, [artist], year, f"{year}-01-01", base)

    songs = [
        # happy-ish (high energy/valence/tempo)
        mk("h1", "Sunshine", "ArtistA", {"tempo": 125, "energy": 0.85, "valence": 0.9, "danceability": 0.8, "acousticness": 0.1, "instrumentalness": 0.02}),
        mk("h2", "Bright Day", "ArtistA", {"tempo": 122, "energy": 0.82, "valence": 0.88, "danceability": 0.78, "acousticness": 0.12, "instrumentalness": 0.03}),
        mk("h3", "Good Vibes", "ArtistB", {"tempo": 128, "energy": 0.88, "valence": 0.92, "danceability": 0.82, "acousticness": 0.08, "instrumentalness": 0.01}),
        # sad-ish (low energy/valence/tempo, high acousticness)
        mk("s1", "Rainy Night", "ArtistC", {"tempo": 68, "energy": 0.2, "valence": 0.15, "danceability": 0.3, "acousticness": 0.7, "instrumentalness": 0.2}),
        mk("s2", "Lonely Road", "ArtistC", {"tempo": 70, "energy": 0.22, "valence": 0.18, "danceability": 0.32, "acousticness": 0.68, "instrumentalness": 0.25}),
        mk("s3", "Grey Skies", "ArtistD", {"tempo": 65, "energy": 0.18, "valence": 0.12, "danceability": 0.28, "acousticness": 0.72, "instrumentalness": 0.22}),
    ]
    store = FeatureStore().build_from_songs(songs)
    m = Matcher(store, max_per_artist=2)

    from intent import parse_intent
    import numpy as np
    from config import AETHER_EMOTIONS

    def dist(**kw):
        d = np.zeros(len(AETHER_EMOTIONS))
        for name, p in kw.items():
            d[AETHER_EMOTIONS.index(name)] = p
        return d

    # 1. Single happy → happy songs on top.
    it = parse_intent(dist(happy=0.9), "happy")
    res = m.match(it, limit=3)
    assert all(r.track_id.startswith("h") for r in res), [r.track_id for r in res]
    print(f"  single happy   → {[r.name for r in res]}")

    # 2. Artist diversity — max 2 per artist (ArtistA has 2 happy songs).
    assert sum(1 for r in res if r.artist == "ArtistA") <= 2
    print(f"  artist cap(2)  → ArtistA count = "
          f"{sum(1 for r in res if r.artist=='ArtistA')}")

    # 3. Mix happy + sad → both emotions represented, no dup.
    it = parse_intent(dist(happy=0.5, sad=0.45), "some happy and some sad")
    assert it.mode == "mix"
    res = m.match(it, limit=4)
    emos = {r.source_emotion for r in res}
    ids = [r.track_id for r in res]
    assert "happy" in emos and "sad" in emos, emos
    assert len(ids) == len(set(ids)), "duplicate track in mix"
    print(f"  mix happy/sad  → {[(r.name, r.source_emotion) for r in res]}")

    # 4. Blend happy + sad → single averaged target (one search).
    it = parse_intent(dist(happy=0.5, sad=0.45), "happy but sad")
    assert it.mode == "blend"
    res = m.match(it, limit=3)
    assert len(res) == len(set(r.track_id for r in res))
    print(f"  blend happy/sad→ {[r.name for r in res]} (source={res[0].source_emotion})")

    # 5. _allocate splits correctly.
    assert _allocate([0.5, 0.5], 4) == [2, 2]
    assert sum(_allocate([0.7, 0.3], 10)) == 10
    print(f"  allocate 0.7/0.3 of 10 → {_allocate([0.7,0.3],10)}")

    print("-" * 55)
    print("✅ All matcher self-tests passed.")


if __name__ == "__main__":
    _selftest()
