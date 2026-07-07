"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 4 · Recommendation Service (offline core)
═══════════════════════════════════════════════════════════════════
The user-facing orchestration layer. One call takes a full request
(fused 15-dim emotion distribution + raw text) all the way to a finished,
sequenced, explained recommendation:

    distribution + raw_text
      → PlaylistGenerator.generate()      (Phase 3: intent → intensity → match)
      → re-hydrate energy/valence/tempo    (from the Phase 2 FeatureStore by id)
      → sequence()                         (Phase 4: energy arc + artist variety)
      → provider.enrich()                  (delivery seam; no-op offline)
      → Recommendation                     (ordered Tracks + full 'why')

Design decisions (interview-defensible)
---------------------------------------
1. Dependency injection: the recommender is given a `PlaylistGenerator` (and
   optionally a `MusicProvider`), so it's unit-testable with a tiny in-memory
   store — no CSV, no network. `from_store_path()` wires the real store.
2. Anti-corruption boundary: Phase 3's `MatchedSong` is normalized into
   Phase 4's `Track` immediately, so Phase 4 never depends on Phase 3 internals.
3. Feature re-hydration: `MatchedSong` is lean (no audio features). We recover
   the normalized energy/valence/tempo from the store by `track_id`, keeping a
   single source of truth for features and Phase 3's output small.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Path bootstrap: make Aether's sibling phases + root config importable ──
_ROOT = Path(__file__).resolve().parent.parent          # …/Aether
_P2 = _ROOT / "phase_2_music_data"
_P3 = _ROOT / "phase_3_emotion_music_mapping"
for _p in (_ROOT, _P2, _P3):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Phase 2 (feature store + feature order)
from feature_store import FeatureStore                  # noqa: E402
from schema import MATCH_FEATURES                        # noqa: E402
# Phase 3 (public API + result types)
from playlist import PlaylistGenerator, PlaylistResult, load_generator  # noqa: E402
from matcher import MatchedSong                          # noqa: E402
# Root config
try:
    from config import AETHER_EMOTIONS                   # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    AETHER_EMOTIONS = []

# Phase 4 (own modules — flat imports, matching the repo convention)
from models import (                                      # noqa: E402
    Track, Recommendation, InvalidRequestError, Phase3ContractError,
)
from sequencer import sequence, default_arc_for          # noqa: E402
from provider import MusicProvider, NullProvider         # noqa: E402

# Indices of the features we re-hydrate, resolved from MATCH_FEATURES so we
# never hardcode positions (robust if the feature order ever changes).
_ENERGY_I = MATCH_FEATURES.index("energy")
_VALENCE_I = MATCH_FEATURES.index("valence")
_TEMPO_I = MATCH_FEATURES.index("tempo")

# Human phrasing for each arc shape, appended to the Phase 3 reason string.
_ARC_PHRASE = {
    "arc": "arranged as a rise-and-settle energy arc",
    "ascending": "arranged as a steady energy build",
    "descending": "arranged as a gentle wind-down",
    "steady": "kept in relevance order",
}


class AetherRecommender:
    """High-level Phase 4 API: full request → finished, sequenced recommendation."""

    def __init__(
        self,
        generator: PlaylistGenerator,
        provider: Optional[MusicProvider] = None,
    ):
        """
        Args:
            generator: a ready Phase 3 PlaylistGenerator (holds the FeatureStore).
            provider:  delivery provider for enrichment/export. Defaults to
                       NullProvider (pure offline — no network).
        """
        self.generator = generator
        self.store: FeatureStore = generator.store
        self.provider: MusicProvider = provider or NullProvider()
        # track_id → normalized 6-dim match vector, for feature re-hydration.
        self._feature_index = self._build_feature_index(self.store)

    # ── convenience constructor ──
    @classmethod
    def from_store_path(
        cls,
        store_path: Optional[str] = None,
        max_per_artist: int = 2,
        provider: Optional[MusicProvider] = None,
    ) -> "AetherRecommender":
        """Load the real Phase 2 store + Phase 3 generator, return a recommender."""
        generator = load_generator(store_path=store_path, max_per_artist=max_per_artist)
        return cls(generator, provider=provider)

    # ── feature re-hydration ──
    @staticmethod
    def _build_feature_index(store: FeatureStore) -> dict[str, np.ndarray]:
        """Map track_id → its normalized match vector (for energy/valence/tempo)."""
        index: dict[str, np.ndarray] = {}
        tids = store.track_ids
        vecs = store.match_vectors
        for i in range(len(store)):
            index[str(tids[i])] = vecs[i]
        return index

    def _to_track(self, ms: MatchedSong) -> Track:
        """Normalize a Phase 3 MatchedSong into a Phase 4 Track (+ re-hydrate features)."""
        vec = self._feature_index.get(str(ms.track_id))
        energy = valence = tempo = None
        if vec is not None:
            energy = float(vec[_ENERGY_I])
            valence = float(vec[_VALENCE_I])
            tempo = float(vec[_TEMPO_I])
        return Track(
            title=ms.name,
            artist=ms.artist,
            track_id=ms.track_id,
            year=ms.year,
            energy=energy,
            valence=valence,
            tempo=tempo,
            match_score=ms.score,
            source_emotion=ms.source_emotion,
        )

    # ── main entry point ──
    def recommend(
        self,
        distribution: np.ndarray,
        raw_text: str = "",
        length: int = 20,
        mode_override: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        arc: Optional[str] = None,
        space_artists: bool = True,
    ) -> Recommendation:
        """
        Produce a finished recommendation from a fused emotion distribution.

        Args:
            distribution: (15,) fused emotion distribution from Phase 1C.
            raw_text: the user's original request (EN/HI free text or transcript).
            length: number of tracks to return.
            mode_override: force 'single' | 'blend' | 'mix' (skip detection).
            year_min / year_max: optional release-year filter.
            arc: force an arc shape ("arc"|"ascending"|"descending"|"steady").
                 If None, chosen automatically from the dominant emotion +
                 intensity via sequencer.default_arc_for().
            space_artists: avoid consecutive same-artist tracks.

        Returns:
            A Recommendation (ordered Tracks + full provenance).

        Raises:
            InvalidRequestError: on a malformed distribution / raw_text.
            Phase3ContractError: if the Phase 3 generate() call fails.
        """
        # 1. Validate the request.
        dist = np.asarray(distribution, dtype=np.float64).flatten()
        if AETHER_EMOTIONS and dist.size != len(AETHER_EMOTIONS):
            raise InvalidRequestError(
                f"distribution must have {len(AETHER_EMOTIONS)} dims, got {dist.size}."
            )
        if not isinstance(raw_text, str):
            raise InvalidRequestError("raw_text must be a string.")
        if length <= 0:
            raise InvalidRequestError(f"length must be positive, got {length}.")

        # 2. Phase 3: intent → intensity → matched songs.
        try:
            pr: PlaylistResult = self.generator.generate(
                dist, raw_text, length=length, mode_override=mode_override,
                year_min=year_min, year_max=year_max,
            )
        except Exception as exc:  # normalize any Phase 3 failure at the boundary
            raise Phase3ContractError(f"Phase 3 generate() failed: {exc}") from exc

        # 3. Normalize + re-hydrate features → Phase 4 Tracks.
        tracks = [self._to_track(ms) for ms in pr.songs]

        # 4. Choose an arc shape and sequence for flow + variety.
        dominant = pr.emotions[0] if pr.emotions else "calm"
        shape = arc or default_arc_for(dominant, pr.intensity)
        tracks = sequence(tracks, shape=shape, space_artists=space_artists)

        # 5. Delivery seam (no-op offline; Deezer/Spotify attach playable data).
        tracks = self.provider.enrich(tracks)

        # 6. Assemble the explained result.
        reason = self._compose_reason(pr, shape)
        dominant_emotions = [
            (e, float(w)) for e, w in zip(pr.emotions, pr.weights)
        ]
        return Recommendation(
            tracks=tracks,
            request_text=raw_text,
            intent_mode=pr.mode,
            intensity_level=pr.intensity,
            intensity_label=pr.intensity_label,
            dominant_emotions=dominant_emotions,
            arc_shape=shape,
            reason=reason,
        )

    @staticmethod
    def _compose_reason(pr: PlaylistResult, shape: str) -> str:
        """Extend Phase 3's reason with the Phase 4 sequencing decision."""
        arc_phrase = _ARC_PHRASE.get(shape, "sequenced")
        return f"{pr.reason} Tracks {arc_phrase}."


# ─────────────────────────────────────────────────────────────
# Self-test — full pipeline on a tiny in-memory store (no CSV, no network)
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Recommender self-test")
    print("-" * 55)

    from schema import Song
    from config import AETHER_EMOTIONS as EMO

    def mk(tid, name, artist, feats, year=2000):
        base = {f: 0.5 for f in
                ["danceability", "energy", "key", "loudness", "mode",
                 "speechiness", "acousticness", "instrumentalness",
                 "liveness", "valence", "tempo"]}
        base.update(feats)
        return Song(tid, name, [artist], year, f"{year}-01-01", base)

    songs = [
        mk("h1", "Sunshine", "A", {"tempo": 125, "energy": 0.85, "valence": 0.9, "danceability": 0.8, "acousticness": 0.1, "instrumentalness": 0.02}),
        mk("h2", "Bright", "B", {"tempo": 122, "energy": 0.70, "valence": 0.88, "danceability": 0.78, "acousticness": 0.12, "instrumentalness": 0.03}),
        mk("h3", "Vibes", "C", {"tempo": 128, "energy": 0.95, "valence": 0.92, "danceability": 0.82, "acousticness": 0.08, "instrumentalness": 0.01}),
        mk("s1", "Rain", "D", {"tempo": 68, "energy": 0.2, "valence": 0.15, "danceability": 0.3, "acousticness": 0.7, "instrumentalness": 0.2}),
        mk("s2", "Road", "E", {"tempo": 70, "energy": 0.22, "valence": 0.18, "danceability": 0.32, "acousticness": 0.68, "instrumentalness": 0.25}),
        mk("s3", "Grey", "F", {"tempo": 65, "energy": 0.18, "valence": 0.12, "danceability": 0.28, "acousticness": 0.72, "instrumentalness": 0.22}),
    ]
    store = FeatureStore().build_from_songs(songs)
    generator = PlaylistGenerator(store, max_per_artist=2)
    rec = AetherRecommender(generator)

    def dist(**kw):
        d = np.zeros(len(EMO))
        for name, p in kw.items():
            d[EMO.index(name)] = p
        return d

    # 1. Single happy → 3 happy tracks, features re-hydrated, ranks set.
    r = rec.recommend(dist(happy=0.8), "so happy", length=3)
    assert r.intent_mode == "single", r.intent_mode
    assert r.size == 3, r.size
    assert all(t.energy is not None for t in r.tracks), "features not re-hydrated"
    assert [t.rank for t in r.tracks] == [1, 2, 3], [t.rank for t in r.tracks]
    assert all(t.track_id.startswith("h") for t in r.tracks), [t.track_id for t in r.tracks]
    print(f"  single happy → {[(t.title, round(t.energy, 2)) for t in r.tracks]} "
          f"[{r.intensity_label}, arc={r.arc_shape}]")

    # 2. High peak → intense intensity carried through from Phase 3.
    assert r.intensity_level == 3, r.intensity_level

    # 3. Sad → auto 'descending' arc (energies non-increasing).
    r_sad = rec.recommend(dist(sad=0.9), "very sad", length=3)
    assert r_sad.arc_shape == "descending", r_sad.arc_shape
    energies = [t.energy for t in r_sad.tracks]
    assert energies == sorted(energies, reverse=True), energies
    print(f"  sad → arc={r_sad.arc_shape}, energies={[round(e, 2) for e in energies]}")

    # 4. Mix → distinct emotions represented, no duplicate track.
    r_mix = rec.recommend(dist(happy=0.5, sad=0.45), "some happy and some sad", length=4)
    assert r_mix.intent_mode == "mix", r_mix.intent_mode
    srcs = {t.source_emotion for t in r_mix.tracks}
    ids = [t.track_id for t in r_mix.tracks]
    assert "happy" in srcs and "sad" in srcs, srcs
    assert len(ids) == len(set(ids)), "duplicate track in mix"
    print(f"  mix → {[(t.title, t.source_emotion) for t in r_mix.tracks]}")

    # 5. Forced arc override is honored.
    r_asc = rec.recommend(dist(happy=0.8), "happy", length=3, arc="ascending")
    en = [t.energy for t in r_asc.tracks]
    assert r_asc.arc_shape == "ascending" and en == sorted(en), en
    print(f"  forced ascending → energies={[round(e, 2) for e in en]}")

    # 6. as_dict() is API-ready.
    d = r.as_dict()
    assert {"intent_mode", "intensity_label", "arc_shape", "reason", "tracks",
            "dominant_emotions"} <= set(d), set(d)
    assert d["tracks"][0]["rank"] == 1
    print(f"  as_dict() → API-ready ✓   reason=\"{d['reason']}\"")

    # 7. Bad distribution length → InvalidRequestError.
    try:
        rec.recommend(np.zeros(3), "x")
        raise AssertionError("expected InvalidRequestError")
    except InvalidRequestError:
        pass

    print("-" * 55)
    print("✅ All recommender self-tests passed.")


if __name__ == "__main__":
    _selftest()
