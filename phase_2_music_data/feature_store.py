"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 2: Music Feature Store
═══════════════════════════════════════════════════════════════════

Turns cleaned Song objects into a searchable store where every song is
represented by a 6-dim MATCH vector in the SAME space as the emotion targets
in config.EMOTION_MUSIC_TARGETS. This is the bridge to Phase 3: once songs and
emotions live in one space, emotion→song matching is a nearest-neighbor query.

The 6 match features (config order):
    [tempo, energy, valence, danceability, acousticness, instrumentalness]

Normalization
-------------
Five of the six are already 0–1 in the source data. `tempo` (BPM) is min-max
normalized to 0–1 using a fixed, documented range (TEMPO_MIN..TEMPO_MAX) so
the store is reproducible and new songs normalize consistently with the built
store. The SAME transform is applied to emotion targets (via
normalize_emotion_target) so both sides match.

Persistence
-----------
The built store saves to a compressed .npz:
    match_vectors  (N, 6) float32   — normalized match space
    track_ids      (N,)   str
    names          (N,)   str
    artists        (N,)   str   (primary artist)
    years          (N,)   int32
plus a sidecar meta.json (feature order, tempo range, counts).

This keeps the store compact (~tens of MB for 1.2M songs) and fast to load —
no need to re-parse the 1.2 GB CSV after the one-time build.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from schema import Song, MATCH_FEATURES

try:
    from config import EMOTION_MUSIC_TARGETS  # type: ignore
except Exception:  # pragma: no cover
    EMOTION_MUSIC_TARGETS = {}

# Fixed tempo range for reproducible min-max normalization (BPM).
# Chosen to cover essentially all real music; clamped outside this range.
TEMPO_MIN = 0.0
TEMPO_MAX = 250.0

_EPS = 1e-8


# ─────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────
def _norm_tempo(bpm: float) -> float:
    """Min-max normalize tempo (BPM) to 0–1, clamped to [TEMPO_MIN, TEMPO_MAX]."""
    v = (float(bpm) - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN + _EPS)
    return float(np.clip(v, 0.0, 1.0))


def song_match_vector(song: Song) -> np.ndarray:
    """Build a song's normalized 6-dim match vector (MATCH_FEATURES order)."""
    vec = np.empty(len(MATCH_FEATURES), dtype=np.float32)
    for i, feat in enumerate(MATCH_FEATURES):
        raw = float(song.raw_features.get(feat, 0.0))
        vec[i] = _norm_tempo(raw) if feat == "tempo" else float(np.clip(raw, 0.0, 1.0))
    return vec


def normalize_emotion_target(target: dict) -> np.ndarray:
    """Normalize an EMOTION_MUSIC_TARGETS entry into the same 6-dim space.

    Applies the identical tempo normalization so emotion targets and song
    vectors are directly comparable in Phase 3.

    Args:
        target: dict like config.EMOTION_MUSIC_TARGETS["happy"].

    Returns:
        (6,) float32 vector in MATCH_FEATURES order.
    """
    vec = np.empty(len(MATCH_FEATURES), dtype=np.float32)
    for i, feat in enumerate(MATCH_FEATURES):
        raw = float(target.get(feat, 0.0))
        vec[i] = _norm_tempo(raw) if feat == "tempo" else float(np.clip(raw, 0.0, 1.0))
    return vec


# ─────────────────────────────────────────────────────────────
# Feature store
# ─────────────────────────────────────────────────────────────
@dataclass
class SearchResult:
    """One matched song from a query."""
    track_id: str
    name: str
    artist: str
    year: Optional[int]
    score: float          # similarity in [0, 1] (higher = closer)

    def as_dict(self) -> dict:
        return {"track_id": self.track_id, "name": self.name,
                "artist": self.artist, "year": self.year,
                "score": round(self.score, 4)}


class FeatureStore:
    """In-memory searchable store of song match-vectors.

    Build it from Song objects (build_from_songs) or load a prebuilt .npz
    (load). Query with search(target_vector, k) for nearest songs.
    """

    def __init__(self):
        self.match_vectors: np.ndarray = np.empty((0, len(MATCH_FEATURES)), dtype=np.float32)
        self.track_ids: np.ndarray = np.empty((0,), dtype=object)
        self.names: np.ndarray = np.empty((0,), dtype=object)
        self.artists: np.ndarray = np.empty((0,), dtype=object)
        self.years: np.ndarray = np.empty((0,), dtype=np.int32)

    # ── build ──
    def build_from_songs(self, songs: Iterable[Song]) -> "FeatureStore":
        """Populate the store from an iterable of Song objects."""
        vecs, ids, names, arts, yrs = [], [], [], [], []
        for s in songs:
            vecs.append(song_match_vector(s))
            ids.append(s.track_id)
            names.append(s.name)
            arts.append(s.primary_artist())
            yrs.append(s.year if s.year is not None else -1)
        if not vecs:
            raise ValueError("No songs provided to build the feature store.")
        self.match_vectors = np.stack(vecs).astype(np.float32)
        self.track_ids = np.array(ids, dtype=object)
        self.names = np.array(names, dtype=object)
        self.artists = np.array(arts, dtype=object)
        self.years = np.array(yrs, dtype=np.int32)
        return self

    def __len__(self) -> int:
        return int(self.match_vectors.shape[0])

    # ── search ──
    def search(
        self,
        target_vector: np.ndarray,
        k: int = 20,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> list[SearchResult]:
        """Return the k songs closest to `target_vector` in match space.

        Similarity = 1 / (1 + euclidean_distance), so score ∈ (0, 1],
        higher = closer. Euclidean is appropriate here because both sides are
        normalized to the same 0–1 per-feature scale.

        Args:
            target_vector: (6,) normalized emotion target.
            k: number of songs to return.
            year_min / year_max: optional release-year filter.

        Returns:
            List of SearchResult, best first.
        """
        if len(self) == 0:
            return []
        tv = np.asarray(target_vector, dtype=np.float32).reshape(1, -1)
        if tv.shape[1] != len(MATCH_FEATURES):
            raise ValueError(
                f"target_vector must be length {len(MATCH_FEATURES)}, got {tv.shape[1]}."
            )

        # Optional year mask.
        mask = np.ones(len(self), dtype=bool)
        if year_min is not None:
            mask &= self.years >= year_min
        if year_max is not None:
            mask &= self.years <= year_max
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            return []

        # Euclidean distance in the 6-dim normalized space.
        diff = self.match_vectors[idxs] - tv
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        scores = 1.0 / (1.0 + dist)

        top = idxs[np.argsort(-scores)[:k]]
        results = []
        for i in top:
            score = float(1.0 / (1.0 + np.sqrt(np.sum(
                (self.match_vectors[i] - tv[0]) ** 2))))
            yr = int(self.years[i])
            results.append(SearchResult(
                track_id=str(self.track_ids[i]),
                name=str(self.names[i]),
                artist=str(self.artists[i]),
                year=yr if yr >= 0 else None,
                score=score,
            ))
        return results

    def search_emotion(self, emotion: str, k: int = 20, **kwargs) -> list[SearchResult]:
        """Convenience: search by an Aether emotion name using config targets."""
        if emotion not in EMOTION_MUSIC_TARGETS:
            raise KeyError(
                f"Unknown emotion {emotion!r}. Must be one of "
                f"{sorted(EMOTION_MUSIC_TARGETS)}."
            )
        tv = normalize_emotion_target(EMOTION_MUSIC_TARGETS[emotion])
        return self.search(tv, k=k, **kwargs)

    # ── persistence ──
    def save(self, path: str) -> None:
        """Save the store to a compressed .npz + meta.json sidecar."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            match_vectors=self.match_vectors,
            track_ids=self.track_ids.astype(str),
            names=self.names.astype(str),
            artists=self.artists.astype(str),
            years=self.years,
        )
        meta = {
            "num_songs": len(self),
            "match_features": MATCH_FEATURES,
            "tempo_min": TEMPO_MIN, "tempo_max": TEMPO_MAX,
        }
        with open(p.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureStore":
        """Load a prebuilt store from .npz."""
        data = np.load(path, allow_pickle=True)
        store = cls()
        store.match_vectors = data["match_vectors"].astype(np.float32)
        store.track_ids = data["track_ids"].astype(object)
        store.names = data["names"].astype(object)
        store.artists = data["artists"].astype(object)
        store.years = data["years"].astype(np.int32)
        return store


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Feature store self-test")
    print("-" * 50)

    # Fake songs at known feature points.
    def mk(tid, name, feats):
        base = {f: 0.5 for f in
                ["danceability", "energy", "key", "loudness", "mode",
                 "speechiness", "acousticness", "instrumentalness",
                 "liveness", "valence", "tempo"]}
        base.update(feats)
        return Song(tid, name, ["Artist " + tid], 2000, "2000-01-01", base)

    # A "happy-ish" song (high energy/valence/tempo) and a "sad-ish" one.
    happy_song = mk("h", "Happy Tune",
                    {"tempo": 120, "energy": 0.8, "valence": 0.85,
                     "danceability": 0.75, "acousticness": 0.2, "instrumentalness": 0.05})
    sad_song = mk("s", "Sad Tune",
                  {"tempo": 70, "energy": 0.2, "valence": 0.15,
                   "danceability": 0.25, "acousticness": 0.65, "instrumentalness": 0.20})

    store = FeatureStore().build_from_songs([happy_song, sad_song])
    assert len(store) == 2

    # tempo 120 → 120/250 = 0.48 normalized
    hv = song_match_vector(happy_song)
    assert abs(hv[0] - 120 / 250) < 1e-4, hv[0]
    assert abs(hv[2] - 0.85) < 1e-4  # valence passes through
    print(f"  match vector (happy): {np.round(hv, 3)}")

    # Search with a happy target → happy song should rank first.
    happy_target = {"tempo": 120, "energy": 0.80, "valence": 0.85,
                    "danceability": 0.75, "acousticness": 0.20, "instrumentalness": 0.05}
    tv = normalize_emotion_target(happy_target)
    res = store.search(tv, k=2)
    assert res[0].track_id == "h", "happy target should match happy song first"
    print(f"  search(happy) → {res[0].name} (score {res[0].score:.3f}), "
          f"{res[1].name} (score {res[1].score:.3f})")

    # Year filter.
    res_y = store.search(tv, k=2, year_min=2010)
    assert res_y == [], "year filter should exclude 2000 songs"
    print("  year filter (>=2010) → correctly empty")

    # Save / load roundtrip.
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "fs_test.npz")
    store.save(tmp)
    loaded = FeatureStore.load(tmp)
    assert len(loaded) == 2
    r2 = loaded.search(tv, k=1)
    assert r2[0].track_id == "h"
    print("  save/load roundtrip ✓")

    print("-" * 50)
    print("✅ All feature-store self-tests passed.")


if __name__ == "__main__":
    _selftest()
