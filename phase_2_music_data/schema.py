"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 2: Music Data Schema
═══════════════════════════════════════════════════════════════════

Defines the canonical `Song` representation and the audio-feature space the
whole music pipeline operates in.

Two feature sets matter:

  1. RAW_FEATURES — the 11 numeric audio features present in the source
     dataset (rodolfofigueroa/spotify-12m-songs). Kept for completeness,
     analytics, and future use.

  2. MATCH_FEATURES — the 6 features that emotion→music matching actually
     uses. These are EXACTLY the keys in config.EMOTION_MUSIC_TARGETS, so a
     song's match-vector and an emotion's target-vector live in the same
     6-dim space and can be compared directly (cosine / distance) in Phase 3.

        MATCH_FEATURES = [tempo, energy, valence, danceability,
                          acousticness, instrumentalness]

Normalization note
------------------
Five of the six match-features are already 0–1 in the source data. `tempo` is
in BPM (~0–250). The feature store (feature_store.py) min-max normalizes tempo
into 0–1 so all six dimensions are comparable; raw values are preserved too.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────────────────────
# All 11 numeric audio features available in the source CSV.
RAW_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

# The 6 features used for emotion matching — MUST match the keys of
# config.EMOTION_MUSIC_TARGETS exactly, in this order.
MATCH_FEATURES = [
    "tempo", "energy", "valence", "danceability",
    "acousticness", "instrumentalness",
]

# Features already on a 0–1 scale in the source data (no normalization needed).
_ALREADY_UNIT_SCALE = {"energy", "valence", "danceability",
                       "acousticness", "instrumentalness"}
# Features that need min-max normalization to 0–1 for matching.
_NEEDS_NORM = {"tempo"}

# Metadata columns we keep from the source CSV.
META_COLUMNS = ["id", "name", "artists", "year", "release_date"]

# All source columns we read (metadata + raw features). Others are dropped.
SOURCE_COLUMNS = META_COLUMNS + RAW_FEATURES


# ─────────────────────────────────────────────────────────────
# Song container
# ─────────────────────────────────────────────────────────────
@dataclass
class Song:
    """One track with its metadata and audio features.

    Attributes:
        track_id: Spotify track id (source `id` column).
        name: Track title.
        artists: List of artist names (parsed from the source's stringified list).
        year: Release year (int), or None if missing.
        release_date: Raw release-date string (e.g. "1999-11-02"), or None.
        raw_features: dict of the 11 raw audio features (source values).
        match_vector: 6-dim list in MATCH_FEATURES order, normalized to 0–1
            (populated by the feature store). Empty until normalized.
    """
    track_id: str
    name: str
    artists: list[str]
    year: Optional[int]
    release_date: Optional[str]
    raw_features: dict[str, float]
    match_vector: list[float] = field(default_factory=list)

    def primary_artist(self) -> str:
        """First (primary) artist name, or 'Unknown'."""
        return self.artists[0] if self.artists else "Unknown"

    def as_dict(self) -> dict:
        """JSON-serializable summary (for the API / debugging)."""
        return {
            "track_id": self.track_id,
            "name": self.name,
            "artists": self.artists,
            "year": self.year,
            "release_date": self.release_date,
            "raw_features": {k: round(float(v), 4) for k, v in self.raw_features.items()},
            "match_vector": [round(float(x), 4) for x in self.match_vector],
        }


# ─────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────
def parse_artists(raw: object) -> list[str]:
    """Parse the source `artists` field into a clean list of names.

    The source stores artists as a stringified Python list, e.g.
    "['Rage Against The Machine']" or "['A', 'B']". This safely parses that
    into ["Rage Against The Machine"] etc., and degrades gracefully on
    malformed input.

    Args:
        raw: the raw `artists` cell value (usually a str).

    Returns:
        A list of artist-name strings (possibly empty).
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip()]
    s = str(raw).strip()
    if not s:
        return []
    # Try to literal-eval the stringified list.
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(a).strip() for a in val if str(a).strip()]
        return [str(val).strip()]
    except (ValueError, SyntaxError):
        # Fallback: strip brackets/quotes and split on commas.
        cleaned = s.strip("[]").replace("'", "").replace('"', "")
        return [part.strip() for part in cleaned.split(",") if part.strip()]


def safe_year(raw: object) -> Optional[int]:
    """Coerce a year value to int, or None if unparseable/out of range."""
    try:
        y = int(float(raw))
        return y if 1900 <= y <= 2100 else None
    except (TypeError, ValueError):
        return None
