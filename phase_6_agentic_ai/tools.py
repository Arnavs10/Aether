"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 6 · Agent Tools
═══════════════════════════════════════════════════════════════════
The concrete actions the agent can take. Each wraps an existing, tested phase
so the agent never re-implements matching or explanation — it orchestrates:

  • recommend_segment(emotion, length, arc)  → Phase 4 AetherRecommender
  • assemble_journey(segments, plan, …)       → stitch per-waypoint segments into
                                                ONE arc-traversing Recommendation
  • explain(recommendation) / annotate(…)     → Phase 5 AetherExplainer (RAG)

Keeping these as a thin tool layer means the agent's graph nodes stay about
DECISIONS (perceive/plan/reflect), while the actual work routes through code
that already has its own green self-tests.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ── Path bootstrap: root + Phase 4 + Phase 5 + local ──
_ROOT = Path(__file__).resolve().parent.parent
_P4 = _ROOT / "phase_4_recommendation"
_P5 = _ROOT / "phase_5_rag"
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _P4, _P5, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import AETHER_EMOTIONS, INTENSITY_LEVELS         # noqa: E402
from models import Track, Recommendation                     # Phase 4  # noqa: E402
from recommender import AetherRecommender                     # Phase 4  # noqa: E402
from rag import AetherExplainer                               # Phase 5  # noqa: E402

# Local
from arc import ArcPlan                                       # noqa: E402


class AetherTools:
    """The agent's action surface over Phases 4 (recommend) and 5 (explain)."""

    def __init__(self, recommender: AetherRecommender,
                 explainer: Optional[AetherExplainer] = None):
        self.recommender = recommender
        self.explainer = explainer or AetherExplainer.default(prefer_chroma=False)

    # ── convenience constructor (real store + RAG) ──
    @classmethod
    def default(
        cls,
        store_path: Optional[str] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        prefer_chroma: bool = True,
    ) -> "AetherTools":
        """Wire the real Phase 4 recommender + Phase 5 explainer."""
        recommender = AetherRecommender.from_store_path(store_path)
        explainer = AetherExplainer.default(prefer_chroma=prefer_chroma, llm_fn=llm_fn)
        return cls(recommender, explainer)

    # ── distribution helper ──
    @staticmethod
    def distribution_for(emotion: str, peak: float = 0.9) -> np.ndarray:
        """A 15-dim distribution peaked on one emotion (drives 'single' intent)."""
        d = np.zeros(len(AETHER_EMOTIONS), dtype=np.float64)
        if emotion in AETHER_EMOTIONS:
            d[AETHER_EMOTIONS.index(emotion)] = peak
        return d

    # ── tool: recommend one waypoint's worth of tracks ──
    def recommend_segment(
        self, emotion: str, length: int, arc: Optional[str] = None,
    ) -> Recommendation:
        """Recommend a segment of `length` tracks for a single waypoint emotion."""
        dist = self.distribution_for(emotion)
        return self.recommender.recommend(
            dist, raw_text=emotion, length=max(1, length), arc=arc,
        )

    # ── tool: explain the final playlist (Phase 5 RAG) ──
    def explain(self, recommendation: Recommendation):
        return self.explainer.explain(recommendation)

    def annotate(self, recommendation: Recommendation) -> Recommendation:
        return self.explainer.annotate(recommendation)

    # ── assembly: stitch per-waypoint segments into one journey ──
    @staticmethod
    def assemble_journey(
        segments: list[Recommendation],
        plan: ArcPlan,
        request_text: str,
        length: int,
    ) -> Recommendation:
        """
        Concatenate waypoint segments (in arc order) into one Recommendation that
        traverses the emotional arc, re-ranking tracks 1..N and recording the
        journey as provenance.
        """
        tracks: list[Track] = []
        for seg in segments:
            tracks.extend(seg.tracks)
        tracks = tracks[:length]
        for i, tr in enumerate(tracks, start=1):
            tr.rank = i

        # Journey provenance: waypoint emotions weighted by how many tracks each
        # contributed.
        counts: dict[str, int] = {}
        for tr in tracks:
            counts[tr.source_emotion or "?"] = counts.get(tr.source_emotion or "?", 0) + 1
        total = max(1, sum(counts.values()))
        # Preserve arc order in the dominant_emotions listing.
        seen: set[str] = set()
        dominant: list[tuple[str, float]] = []
        for w in plan.waypoints:
            if w in counts and w not in seen:
                dominant.append((w, counts[w] / total))
                seen.add(w)
        if not dominant:
            dominant = [(plan.waypoints[0], 1.0)]

        # Carry the strongest intensity seen across segments.
        level = max((s.intensity_level for s in segments), default=2)
        label = INTENSITY_LEVELS.get(level, "moderate")

        shape = plan.direction if plan.is_journey else (
            segments[0].arc_shape if segments else "steady")
        mode = "journey" if plan.is_journey else (
            segments[0].intent_mode if segments else "single")

        reason = (
            f"A journey through {plan.describe()} "
            f"({label} intensity, energy {plan.direction})."
            if plan.is_journey
            else f"A {label} '{plan.waypoints[0]}' set."
        )

        return Recommendation(
            tracks=tracks, request_text=request_text, intent_mode=mode,
            intensity_level=level, intensity_label=label,
            dominant_emotions=dominant, arc_shape=shape, reason=reason,
        )


# ─────────────────────────────────────────────────────────────
# Self-test — tiny in-memory store (no CSV/network)
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Agent tools self-test")
    print("-" * 55)

    from schema import Song                       # Phase 2
    from feature_store import FeatureStore        # Phase 2
    from playlist import PlaylistGenerator        # Phase 3
    from arc import plan_arc

    def mk(tid, name, artist, feats, year=2000):
        base = {f: 0.5 for f in
                ["danceability", "energy", "key", "loudness", "mode",
                 "speechiness", "acousticness", "instrumentalness",
                 "liveness", "valence", "tempo"]}
        base.update(feats)
        return Song(tid, name, [artist], year, f"{year}-01-01", base)

    # Store spanning several emotional corners so segments differ.
    songs = [
        mk("a1", "Edge", "Ana", {"tempo": 105, "energy": 0.62, "valence": 0.22, "danceability": 0.38, "acousticness": 0.30, "instrumentalness": 0.25}),
        mk("a2", "Nerve", "Ana", {"tempo": 108, "energy": 0.60, "valence": 0.24, "danceability": 0.40, "acousticness": 0.28, "instrumentalness": 0.22}),
        mk("f1", "Flow", "Ben", {"tempo": 92, "energy": 0.35, "valence": 0.42, "danceability": 0.18, "acousticness": 0.45, "instrumentalness": 0.70}),
        mk("f2", "Deep Work", "Cy", {"tempo": 90, "energy": 0.34, "valence": 0.40, "danceability": 0.20, "acousticness": 0.48, "instrumentalness": 0.72}),
        mk("c1", "Still", "Dee", {"tempo": 80, "energy": 0.22, "valence": 0.50, "danceability": 0.28, "acousticness": 0.80, "instrumentalness": 0.40}),
        mk("c2", "Breath", "Eli", {"tempo": 78, "energy": 0.20, "valence": 0.52, "danceability": 0.26, "acousticness": 0.82, "instrumentalness": 0.42}),
        mk("n1", "Old Film", "Fi", {"tempo": 88, "energy": 0.38, "valence": 0.48, "danceability": 0.32, "acousticness": 0.72, "instrumentalness": 0.15}),
    ]
    store = FeatureStore().build_from_songs(songs)
    tools = AetherTools(
        AetherRecommender(PlaylistGenerator(store, max_per_artist=2)),
        AetherExplainer.default(prefer_chroma=False),
    )

    # 1. distribution_for peaks the right index.
    d = tools.distribution_for("calm")
    assert d[AETHER_EMOTIONS.index("calm")] == 0.9 and d.sum() == 0.9
    print("  distribution_for('calm') ✓")

    # 2. recommend_segment returns that emotion's tracks.
    seg = tools.recommend_segment("focused", length=2)
    assert seg.size == 2 and all(t.source_emotion == "focused" for t in seg.tracks)
    print(f"  recommend_segment('focused',2) → {[t.title for t in seg.tracks]}")

    # 3. assemble_journey stitches segments into one arc-traversing playlist.
    plan = plan_arc("anxious", "calm", max_waypoints=3)   # anxious→…→calm
    segs = [tools.recommend_segment(w, length=2, arc=plan.direction)
            for w in plan.waypoints]
    journey = AetherTools.assemble_journey(segs, plan, "from anxious to calm", length=6)
    assert journey.intent_mode == "journey", journey.intent_mode
    assert [t.rank for t in journey.tracks] == list(range(1, journey.size + 1))
    srcs_in_order = [t.source_emotion for t in journey.tracks]
    # First track's emotion is the start waypoint; last is the target.
    assert srcs_in_order[0] == plan.waypoints[0], srcs_in_order
    assert srcs_in_order[-1] == plan.waypoints[-1], srcs_in_order
    print(f"  assemble_journey → {plan.describe()} : "
          f"{[(t.title, t.source_emotion) for t in journey.tracks]}")

    # 4. Energy actually trends the intended direction across the journey.
    first_e = journey.tracks[0].energy
    last_e = journey.tracks[-1].energy
    assert last_e < first_e, (first_e, last_e)   # anxious→calm = energy down
    print(f"  energy trend → {first_e:.2f} → {last_e:.2f} (descending) ✓")

    # 5. Phase 5 explains the assembled journey; every track grounded.
    expl = tools.explain(journey)
    assert all(t.why and t.citations for t in expl.tracks)
    print(f"  explain(journey) summary → {expl.summary[:70]}…")

    print("-" * 55)
    print("✅ All agent-tools self-tests passed.")


if __name__ == "__main__":
    _selftest()
