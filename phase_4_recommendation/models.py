"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 4 · Data Models
═══════════════════════════════════════════════════════════════════
Phase-4-owned types. Phase 4 deliberately does NOT reuse Phase 3's
`MatchedSong` internally: it normalizes each match into these stable,
provider-agnostic structures at the boundary (an anti-corruption layer).

Why this matters:
  • The recommendation service stays decoupled from Phase 3 internals — if
    `MatchedSong` gains/loses a field, we change ONE normalizer, not the
    whole module.
  • The live delivery layer (Deezer/Spotify) and the future RAG/agent layers
    all consume ONE stable shape (`Track` / `Recommendation`).

Note on audio features
----------------------
Phase 3's `MatchedSong` is intentionally lean (id, name, artist, year, score,
source_emotion) and does NOT carry audio features. Phase 4 re-hydrates
`energy` / `valence` / `tempo` from the Phase 2 `FeatureStore` by `track_id`
(see recommender.py). These are the store's NORMALIZED 0–1 values — exactly
what the energy-arc sequencer needs. They may be ``None`` if a track_id isn't
found; downstream logic degrades gracefully.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ──────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────
class AetherRecommenderError(Exception):
    """Base class for all Phase 4 recommendation errors."""


class InvalidRequestError(AetherRecommenderError):
    """Raised when the incoming request (distribution/text) is malformed."""


class Phase3ContractError(AetherRecommenderError):
    """Raised when the Phase 3 generate() call fails or returns an unusable result."""


# ──────────────────────────────────────────────
# Track — the normalized unit of a recommendation
# ──────────────────────────────────────────────
@dataclass
class Track:
    """
    A single, provider-agnostic track in a recommendation.

    ``energy`` / ``valence`` / ``tempo`` are the Phase 2 store's normalized
    0–1 features, re-hydrated by the recommender. They may be ``None``.

    ``provider_ref`` is empty at the core stage; the live delivery layer fills
    it with playable data (preview URL, streaming URI, external link, cover…).
    """

    title: str                                # ← Phase 3 MatchedSong.name
    artist: str                               # ← MatchedSong.artist (primary)
    track_id: Optional[str] = None            # ← MatchedSong.track_id
    year: Optional[int] = None                # ← MatchedSong.year

    # Audio features (normalized 0–1), re-hydrated from the feature store.
    energy: Optional[float] = None
    valence: Optional[float] = None
    tempo: Optional[float] = None

    # Provenance / scoring
    match_score: Optional[float] = None       # ← MatchedSong.score (cosine-like)
    source_emotion: Optional[str] = None       # ← MatchedSong.source_emotion

    # Final ordered position in the playlist (set by the sequencer, 1-based).
    rank: Optional[int] = None

    # Filled later by the live delivery provider (preview_url, uri, link…).
    provider_ref: dict[str, Any] = field(default_factory=dict)

    # Carry-through for any additional fields we don't model explicitly.
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (API-ready)."""
        return asdict(self)


# ──────────────────────────────────────────────
# Recommendation — the full Phase 4 output
# ──────────────────────────────────────────────
@dataclass
class Recommendation:
    """
    The complete result of one recommendation request: an ordered list of
    tracks plus the full 'why' (intent, intensity, dominant emotions, arc, and
    a human-readable reason the Phase 5 RAG layer can later expand on).
    """

    tracks: list[Track]
    request_text: str
    intent_mode: str                            # "single" | "blend" | "mix"
    intensity_level: int                        # 0–3 (config.INTENSITY_LEVELS)
    intensity_label: str                        # neutral|mild|moderate|intense
    dominant_emotions: list[tuple[str, float]]   # [(emotion, weight), …]
    arc_shape: str                              # arc|ascending|descending|steady
    reason: str                                 # human-readable explanation

    @property
    def size(self) -> int:
        return len(self.tracks)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (API-ready)."""
        return {
            "request_text": self.request_text,
            "intent_mode": self.intent_mode,
            "intensity_level": self.intensity_level,
            "intensity_label": self.intensity_label,
            "dominant_emotions": [
                {"emotion": e, "weight": round(float(w), 4)}
                for e, w in self.dominant_emotions
            ],
            "arc_shape": self.arc_shape,
            "reason": self.reason,
            "size": self.size,
            "tracks": [t.as_dict() for t in self.tracks],
        }
