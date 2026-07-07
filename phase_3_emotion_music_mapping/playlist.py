"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 3: Playlist Generator  (public interface)
═══════════════════════════════════════════════════════════════════

The single entry point downstream phases (4: recommendation, 8: website) call
to turn a Phase 1C fusion result + the user's raw request into a finished,
ready-to-serve playlist.

Pipeline:

    fused 15-dim distribution + raw_text
        → intent.parse_intent()      (blend | mix | single; EN + HI)
        → intensity scaling          (config.INTENSITY_MODIFIERS)
        → Matcher.match()            (retrieval over the 1.2M feature store)
        → PlaylistResult             (ordered songs + full provenance)

Intensity
---------
The fused distribution's peak probability implies how STRONG the emotion is.
We map that to an intensity level (0–3) and scale each emotion's raw music
target via config.INTENSITY_MODIFIERS *before* normalization — so "intensely
happy" pulls higher-energy/tempo songs than "mildly happy". This is applied by
temporarily adjusting the target the matcher searches with.

The result carries full provenance (mode, emotions, weights, intensity, why)
so the Phase 5 RAG layer can later explain "why these songs" and the API can
surface it.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from intent import parse_intent, MatchIntent
from matcher import Matcher, MatchedSong

# Phase 2 modules (feature store + normalization).
_P2 = Path(__file__).resolve().parent.parent / "phase_2_music_data"
if _P2.exists() and str(_P2) not in sys.path:
    sys.path.insert(0, str(_P2))
from feature_store import FeatureStore  # noqa: E402

try:
    from config import (  # type: ignore
        EMOTION_MUSIC_TARGETS, INTENSITY_MODIFIERS, INTENSITY_LEVELS,
        AETHER_EMOTIONS,
    )
except Exception:  # pragma: no cover
    EMOTION_MUSIC_TARGETS, INTENSITY_MODIFIERS, INTENSITY_LEVELS = {}, {}, {}
    AETHER_EMOTIONS = []

_EPS = 1e-8

# Peak-probability thresholds → intensity level (0–3).
# Below MILD → still 'mild' (we always have *some* signal by this point).
_INTENSITY_THRESHOLDS = [
    (0.65, 3),   # peak >= 0.65 → intense
    (0.40, 2),   # >= 0.40 → moderate
    (0.00, 1),   # else → mild
]


@dataclass
class PlaylistResult:
    """A finished playlist plus the reasoning behind it."""
    songs: list[MatchedSong]
    mode: str                       # blend | mix | single
    emotions: list[str]
    weights: list[float]
    intensity: int                  # 0–3
    intensity_label: str            # neutral | mild | moderate | intense
    reason: str                     # human-readable "why this playlist"
    raw_text: str = ""

    def __len__(self) -> int:
        return len(self.songs)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "emotions": self.emotions,
            "weights": [round(float(w), 4) for w in self.weights],
            "intensity": self.intensity,
            "intensity_label": self.intensity_label,
            "reason": self.reason,
            "num_songs": len(self.songs),
            "songs": [s.as_dict() for s in self.songs],
        }


# ─────────────────────────────────────────────────────────────
# Intensity
# ─────────────────────────────────────────────────────────────
def _intensity_from_distribution(distribution: np.ndarray) -> int:
    """Map the distribution's peak probability to an intensity level (1–3)."""
    dist = np.asarray(distribution, dtype=np.float64).flatten()
    if dist.sum() <= _EPS:
        return 1
    peak = float(dist.max() / dist.sum())
    for thresh, level in _INTENSITY_THRESHOLDS:
        if peak >= thresh:
            return level
    return 1


def _scaled_targets(intensity: int) -> dict:
    """Return a copy of EMOTION_MUSIC_TARGETS scaled by the intensity modifier.

    INTENSITY_MODIFIERS scales tempo/energy/valence multiplicatively on the raw
    (pre-normalization) targets. Features not in the modifier pass through
    unchanged. Level 0 (neutral) is treated as level 1 here — by this stage we
    always have a real emotion to act on.
    """
    mod = INTENSITY_MODIFIERS.get(intensity) or INTENSITY_MODIFIERS.get(2, {})
    scaled = {}
    for emo, target in EMOTION_MUSIC_TARGETS.items():
        new_t = dict(target)
        for feat, factor in mod.items():
            if feat in new_t and factor > 0:
                new_t[feat] = new_t[feat] * factor
        scaled[emo] = new_t
    return scaled


# ─────────────────────────────────────────────────────────────
# Reason string
# ─────────────────────────────────────────────────────────────
def _build_reason(intent: MatchIntent, intensity_label: str) -> str:
    """Human-readable explanation of the playlist choice."""
    emos = intent.emotions
    if intent.mode == "single":
        return f"An {intensity_label} '{emos[0]}' playlist."
    if intent.mode == "blend":
        return (f"A blended '{' + '.join(emos)}' mood "
                f"({intensity_label}) — songs sitting between these emotions.")
    # mix
    parts = ", ".join(f"{w:.0%} {e}" for e, w in zip(emos, intent.weights))
    return (f"A mix of distinct songs: {parts} ({intensity_label}).")


# ─────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────
class PlaylistGenerator:
    """High-level Phase 3 API: distribution + text → PlaylistResult."""

    def __init__(
        self,
        store: FeatureStore,
        max_per_artist: int = 2,
    ):
        self.store = store
        self.max_per_artist = max_per_artist

    def generate(
        self,
        distribution: np.ndarray,
        raw_text: str = "",
        length: int = 20,
        mode_override: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> PlaylistResult:
        """Generate a playlist from a fused emotion distribution + request text.

        Args:
            distribution: (15,) fused emotion distribution from Phase 1C.
            raw_text: user's original request (EN/HI free text or transcript).
            length: number of songs in the playlist.
            mode_override: force 'single' | 'blend' | 'mix' (skip detection).
            year_min / year_max: optional release-year filter (e.g. decade).

        Returns:
            PlaylistResult with ordered songs and full provenance.
        """
        # 1. Parse intent (which emotions + blend/mix).
        intent = parse_intent(distribution, raw_text, mode_override=mode_override)

        # 2. Intensity from the distribution's peak.
        intensity = _intensity_from_distribution(distribution)
        intensity_label = INTENSITY_LEVELS.get(intensity, "moderate")

        # 3. Build a matcher over intensity-scaled targets.
        #    We temporarily swap the module-level targets the matcher reads so
        #    the same emotion pulls stronger/softer songs by intensity.
        scaled = _scaled_targets(intensity)
        matcher = Matcher(self.store, max_per_artist=self.max_per_artist)
        songs = self._match_with_targets(matcher, intent, scaled, length,
                                          year_min, year_max)

        reason = _build_reason(intent, intensity_label)
        return PlaylistResult(
            songs=songs, mode=intent.mode, emotions=intent.emotions,
            weights=intent.weights, intensity=intensity,
            intensity_label=intensity_label, reason=reason, raw_text=raw_text,
        )

    @staticmethod
    def _match_with_targets(
        matcher: Matcher, intent: MatchIntent, scaled_targets: dict,
        length: int, year_min: Optional[int], year_max: Optional[int],
    ) -> list[MatchedSong]:
        """Run the matcher using intensity-scaled targets.

        The matcher normally reads config.EMOTION_MUSIC_TARGETS. To apply
        intensity without mutating global config, we monkey-patch the matcher's
        target lookup for the duration of this call via a small override.
        """
        # Override the matcher's per-emotion target source with scaled targets.
        import matcher as matcher_mod
        original = matcher_mod.EMOTION_MUSIC_TARGETS
        try:
            matcher_mod.EMOTION_MUSIC_TARGETS = scaled_targets
            return matcher.match(intent, limit=length,
                                 year_min=year_min, year_max=year_max)
        finally:
            matcher_mod.EMOTION_MUSIC_TARGETS = original


# ─────────────────────────────────────────────────────────────
# Convenience one-shot
# ─────────────────────────────────────────────────────────────
def load_generator(
    store_path: str = None,
    max_per_artist: int = 2,
) -> PlaylistGenerator:
    """Load the feature store and return a ready PlaylistGenerator.

    Args:
        store_path: path to the Phase 2 .npz store. Defaults to the standard
            location under phase_2_music_data/store/.
    """
    if store_path is None:
        store_path = str(_P2 / "store" / "music_store.npz")
    store = FeatureStore.load(store_path)
    return PlaylistGenerator(store, max_per_artist=max_per_artist)


# ─────────────────────────────────────────────────────────────
# Self-test (tiny in-memory store; no CSV needed)
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Playlist generator self-test")
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
        mk("h2", "Bright", "B", {"tempo": 122, "energy": 0.82, "valence": 0.88, "danceability": 0.78, "acousticness": 0.12, "instrumentalness": 0.03}),
        mk("h3", "Vibes", "C", {"tempo": 128, "energy": 0.88, "valence": 0.92, "danceability": 0.82, "acousticness": 0.08, "instrumentalness": 0.01}, year=2020),
        mk("s1", "Rain", "D", {"tempo": 68, "energy": 0.2, "valence": 0.15, "danceability": 0.3, "acousticness": 0.7, "instrumentalness": 0.2}),
        mk("s2", "Road", "E", {"tempo": 70, "energy": 0.22, "valence": 0.18, "danceability": 0.32, "acousticness": 0.68, "instrumentalness": 0.25}, year=2020),
        mk("s3", "Grey", "F", {"tempo": 65, "energy": 0.18, "valence": 0.12, "danceability": 0.28, "acousticness": 0.72, "instrumentalness": 0.22}),
    ]
    store = FeatureStore().build_from_songs(songs)
    gen = PlaylistGenerator(store, max_per_artist=2)

    def dist(peak=0.5, **kw):
        d = np.zeros(len(EMO))
        for name, p in kw.items():
            d[EMO.index(name)] = p
        return d

    # 1. Single happy, moderate intensity.
    pr = gen.generate(dist(happy=0.45), "happy songs", length=3)
    assert pr.mode == "single" and pr.emotions == ["happy"]
    assert all(s.track_id.startswith("h") for s in pr.songs), [s.track_id for s in pr.songs]
    print(f"  single happy → {[s.name for s in pr.songs]} "
          f"[{pr.intensity_label}]")

    # 2. Intensity: high peak → intense.
    pr_hi = gen.generate(dist(happy=0.8), "so happy", length=3)
    assert pr_hi.intensity == 3, pr_hi.intensity
    print(f"  peak 0.8 → intensity '{pr_hi.intensity_label}' (level {pr_hi.intensity})")

    # 3. Mix.
    pr = gen.generate(dist(happy=0.5, sad=0.45), "some happy and some sad", length=4)
    assert pr.mode == "mix"
    srcs = {s.source_emotion for s in pr.songs}
    assert "happy" in srcs and "sad" in srcs
    print(f"  mix → {[(s.name, s.source_emotion) for s in pr.songs]}")

    # 4. Blend.
    pr = gen.generate(dist(happy=0.5, sad=0.45), "happy but sad", length=3)
    assert pr.mode == "blend"
    print(f"  blend → {[s.name for s in pr.songs]} :: reason=\"{pr.reason}\"")

    # 5. Year filter (only 2020 songs).
    pr = gen.generate(dist(happy=0.5, sad=0.45), "some happy and some sad",
                      length=4, year_min=2020)
    assert all(s.year == 2020 for s in pr.songs), [(s.name, s.year) for s in pr.songs]
    print(f"  year>=2020 → {[(s.name, s.year) for s in pr.songs]}")

    # 6. as_dict is API-ready.
    d = gen.generate(dist(happy=0.5), "happy").as_dict()
    assert {"mode", "emotions", "intensity_label", "reason", "songs"} <= set(d)
    print("  as_dict() → API-ready ✓")

    print("-" * 55)
    print("✅ All playlist-generator self-tests passed.")


if __name__ == "__main__":
    _selftest()
