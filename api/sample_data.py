"""
═══════════════════════════════════════════════════════════════════
AETHER — API · Sample Library
═══════════════════════════════════════════════════════════════════
A small in-memory catalog (~26 tracks) spanning the emotional range and the
Camelot wheel, so the API returns sensible playlists and the live player has
real harmonic choices — with zero data files.

This is dev scaffolding. Swapping to the real 1.2M store is a one-line change in
the service (`AetherService.from_store_path(...)`); nothing here is load-bearing
for production.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_P2 = _ROOT / "phase_2_music_data"
for _p in (_ROOT, _P2):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schema import Song                                       # Phase 2  # noqa: E402

# Per-emotion feature templates (match features + a mood-typical acoustic feel).
_GROUPS: dict[str, dict[str, float]] = {
    "sad":       {"energy": 0.20, "valence": 0.15, "danceability": 0.30, "acousticness": 0.72, "instrumentalness": 0.25},
    "calm":      {"energy": 0.25, "valence": 0.52, "danceability": 0.30, "acousticness": 0.80, "instrumentalness": 0.45},
    "focused":   {"energy": 0.35, "valence": 0.42, "danceability": 0.20, "acousticness": 0.45, "instrumentalness": 0.75},
    "nostalgic": {"energy": 0.42, "valence": 0.50, "danceability": 0.40, "acousticness": 0.65, "instrumentalness": 0.20},
    "hopeful":   {"energy": 0.52, "valence": 0.62, "danceability": 0.52, "acousticness": 0.45, "instrumentalness": 0.15},
    "happy":     {"energy": 0.66, "valence": 0.82, "danceability": 0.66, "acousticness": 0.30, "instrumentalness": 0.10},
    "energetic": {"energy": 0.86, "valence": 0.76, "danceability": 0.70, "acousticness": 0.18, "instrumentalness": 0.12},
    "romantic":  {"energy": 0.42, "valence": 0.62, "danceability": 0.50, "acousticness": 0.60, "instrumentalness": 0.18},
    "angry":     {"energy": 0.82, "valence": 0.24, "danceability": 0.48, "acousticness": 0.20, "instrumentalness": 0.22},
    "anxious":   {"energy": 0.60, "valence": 0.22, "danceability": 0.40, "acousticness": 0.30, "instrumentalness": 0.25},
}

# (id, title, artist, emotion, key, mode, tempo_bpm)
_ROWS = [
    ("sad-1", "Paper Boats", "Lyra Vance", "sad", 9, 0, 70),        # 8A
    ("sad-2", "Grey Window", "Elias Moor", "sad", 2, 0, 68),        # 7A
    ("sad-3", "Undertow", "Nils Hartt", "sad", 4, 0, 72),           # 9A
    ("calm-1", "Slow Tide", "Mira Sol", "calm", 0, 1, 82),          # 8B
    ("calm-2", "Cedar Air", "Ren Okabe", "calm", 7, 1, 80),         # 9B
    ("calm-3", "Still Harbor", "Ana Belle", "calm", 11, 1, 78),     # 1B
    ("focus-1", "Deep Current", "Kavi Rao", "focused", 5, 1, 96),   # 7B
    ("focus-2", "Signal Path", "Otto Lind", "focused", 0, 1, 92),   # 8B
    ("focus-3", "Quiet Engine", "Vera Cho", "focused", 7, 1, 98),   # 9B
    ("nost-1", "Old Cinema", "Sam Ives", "nostalgic", 2, 1, 90),    # 10B
    ("nost-2", "Super 8", "Dana Wren", "nostalgic", 9, 1, 88),      # 11B
    ("hope-1", "First Light", "Isla Faye", "hopeful", 7, 1, 104),   # 9B
    ("hope-2", "Open Road", "Milo Grant", "hopeful", 2, 1, 108),    # 10B
    ("happy-1", "Sunday Bright", "Coco Lane", "happy", 0, 1, 116),  # 8B
    ("happy-2", "Confetti", "Bex Powell", "happy", 7, 1, 118),      # 9B
    ("en-1", "Voltage", "Nova Kane", "energetic", 7, 1, 128),       # 9B
    ("en-2", "Afterburner", "Rhys Kelo", "energetic", 0, 1, 126),   # 8B
    ("en-3", "Ignition", "Tess Vane", "energetic", 5, 1, 132),      # 7B
    ("en-4", "Redline", "Jax Moro", "energetic", 2, 1, 130),        # 10B
    ("rom-1", "Slow Dance", "Lea Marín", "romantic", 4, 1, 96),     # 12B
    ("rom-2", "Candlelight", "Theo Ray", "romantic", 9, 1, 92),     # 11B
    ("ang-1", "Break Glass", "Ivo Stark", "angry", 5, 0, 134),      # 4A
    ("ang-2", "Fault Line", "Mara Vex", "angry", 0, 0, 128),        # 5A
    ("anx-1", "Racing Mind", "Piper Aldo", "anxious", 9, 0, 112),   # 8A
    ("anx-2", "Held Breath", "Cy Nolan", "anxious", 2, 0, 108),     # 7A
    ("anx-3", "Static", "Wren Bell", "anxious", 4, 0, 114),         # 9A
]


def _song(tid: str, title: str, artist: str, emotion: str,
          key: int, mode: int, tempo: float, year: int = 2015) -> Song:
    base = _GROUPS[emotion]
    rf = {
        "danceability": base["danceability"], "energy": base["energy"],
        "key": key, "loudness": 0.5, "mode": mode, "speechiness": 0.08,
        "acousticness": base["acousticness"],
        "instrumentalness": base["instrumentalness"], "liveness": 0.12,
        "valence": base["valence"], "tempo": tempo,
    }
    return Song(tid, title, [artist], year, f"{year}-01-01", rf)


def build_sample_songs() -> list[Song]:
    """The full sample catalog as Song objects."""
    return [_song(tid, title, artist, emo, key, mode, tempo)
            for (tid, title, artist, emo, key, mode, tempo) in _ROWS]


if __name__ == "__main__":
    songs = build_sample_songs()
    print(f"sample catalog: {len(songs)} songs")
    print("  first:", songs[0].name, "-", songs[0].primary_artist(),
          "| tempo", songs[0].raw_features["tempo"])
