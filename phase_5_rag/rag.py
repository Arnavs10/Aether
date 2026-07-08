"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 5 · RAG Facade (public API)
═══════════════════════════════════════════════════════════════════
The one class the rest of Aether (the future REST API and the Phase 6 agent)
calls to get explainable recommendations:

    knowledge base  →  retriever  →  explainer  →  Explanation

    AetherExplainer.default()           # wire the whole stack in one line
        .explain(recommendation)        # → Explanation (summary + per-track why)
        .annotate(recommendation)       # → same Recommendation, enriched in place

Design decisions (interview-defensible)
---------------------------------------
1. Composition root. This module is the ONLY place that wires the three seams
   together, so swapping the retriever (TF-IDF ↔ Chroma) or the generator
   (template ↔ LLM) is a one-argument change and nothing downstream moves.
2. Non-invasive enrichment. `annotate()` feeds the "why" back into the EXISTING
   Phase 4 shapes — it upgrades `Recommendation.reason` to the grounded summary
   and writes each track's rationale to `Track.extra["why"]` (+ its citations).
   Phase 4's `as_dict()` therefore serves explanations for free; no schema change.
3. Fail-open. Explanation is an enhancement layer, never a gate. If retrieval or
   generation fails, `annotate()` leaves the recommendation untouched and usable.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional

# ── Path bootstrap: root + all sibling phases + local modules ──
_ROOT = Path(__file__).resolve().parent.parent
_P2 = _ROOT / "phase_2_music_data"
_P3 = _ROOT / "phase_3_emotion_music_mapping"
_P4 = _ROOT / "phase_4_recommendation"
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _P2, _P3, _P4, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from knowledge import KnowledgeDoc, build_knowledge_base   # noqa: E402
from retriever import Retriever, build_retriever            # noqa: E402
from explainer import (                                     # noqa: E402
    Explainer, GroundedTemplateExplainer, LLMExplainer, Explanation,
)


class AetherExplainer:
    """High-level Phase 5 API: a wired knowledge→retriever→explainer stack."""

    def __init__(
        self,
        explainer: Optional[Explainer] = None,
        retriever: Optional[Retriever] = None,
        docs: Optional[list[KnowledgeDoc]] = None,
    ):
        """
        Args:
            explainer: the generator to use. Defaults to GroundedTemplateExplainer.
            retriever: an INDEXED retriever. Built (TF-IDF/Chroma auto) if None.
            docs: corpus to index when building a retriever (defaults to the
                curated knowledge base).
        """
        self.docs = docs if docs is not None else build_knowledge_base()
        self.retriever = retriever or build_retriever(self.docs)
        self.explainer = explainer or GroundedTemplateExplainer(self.retriever)

    # ── convenience constructor ──
    @classmethod
    def default(
        cls,
        prefer_chroma: bool = True,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> "AetherExplainer":
        """
        Wire the full stack. Uses the dense Chroma backend if available (else
        TF-IDF), and the LLM generator iff `llm_fn` is supplied (else the
        offline grounded-template generator).
        """
        docs = build_knowledge_base()
        retriever = build_retriever(docs, prefer_chroma=prefer_chroma)
        explainer: Explainer = (
            LLMExplainer(retriever, llm_fn=llm_fn) if llm_fn is not None
            else GroundedTemplateExplainer(retriever)
        )
        return cls(explainer=explainer, retriever=retriever, docs=docs)

    # ── main entry points ──
    def explain(self, recommendation: Any) -> Explanation:
        """Produce a full Explanation (summary + per-track why) for a recommendation."""
        return self.explainer.explain(recommendation)

    def annotate(self, recommendation: Any) -> Any:
        """
        Enrich a Recommendation in place with grounded explanations and return
        it: `reason` becomes the grounded summary, and every track gains
        ``extra["why"]`` and ``extra["why_citations"]``.

        Fail-open: on any error the recommendation is returned unchanged.
        """
        try:
            expl = self.explainer.explain(recommendation)
        except Exception as exc:  # never let explanation break delivery
            print(f"  [rag] explanation skipped ({exc}); recommendation unchanged.")
            return recommendation

        recommendation.reason = expl.summary
        # Explanations are in track order (explain iterates rec.tracks in order).
        for track, te in zip(recommendation.tracks, expl.tracks):
            track.extra["why"] = te.why
            track.extra["why_citations"] = te.citations
            if te.feature_notes:
                track.extra["why_features"] = te.feature_notes
        return recommendation


# ─────────────────────────────────────────────────────────────
# Self-test — FULL pipeline: Phase 2 store → Phase 4 recommend → Phase 5 explain
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("RAG facade self-test (end-to-end)")
    print("-" * 55)

    import numpy as np
    from schema import Song                       # Phase 2
    from feature_store import FeatureStore        # Phase 2
    from playlist import PlaylistGenerator        # Phase 3
    from recommender import AetherRecommender     # Phase 4
    from config import AETHER_EMOTIONS as EMO

    # Build a tiny in-memory store spanning happy & sad corners (no CSV/network).
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
    recommender = AetherRecommender(PlaylistGenerator(store, max_per_artist=2))

    def dist(**kw):
        d = np.zeros(len(EMO))
        for name, p in kw.items():
            d[EMO.index(name)] = p
        return d

    # Real Phase 4 recommendation.
    rec = recommender.recommend(dist(sad=0.9), "i feel really low today", length=3)
    assert rec.arc_shape == "descending"
    print(f"  recommended → {[t.title for t in rec.tracks]} "
          f"[{rec.intensity_label}, arc={rec.arc_shape}]")

    rag = AetherExplainer.default(prefer_chroma=False)

    # 1. explain() → grounded Explanation with per-track why + citations.
    expl = rag.explain(rec)
    assert len(expl.tracks) == rec.size
    assert all(t.why and t.citations for t in expl.tracks)
    assert "emotion:sad" in {c for t in expl.tracks for c in t.citations}
    print(f"  explain() summary → {expl.summary}")
    print(f"  explain() why[0]  → {expl.tracks[0].why}")

    # 2. annotate() flows the why back into the Phase 4 shapes.
    original_reason = rec.reason
    rag.annotate(rec)
    assert rec.reason != original_reason and rec.reason == expl.summary
    assert all("why" in t.extra and t.extra["why"] for t in rec.tracks)
    assert all("why_citations" in t.extra for t in rec.tracks)
    print(f"  annotate() → reason upgraded, every track has extra['why'] ✓")

    # 3. Phase 4 as_dict() now serves explanations for free (no schema change).
    d = rec.as_dict()
    assert d["tracks"][0]["extra"]["why"], d["tracks"][0]["extra"]
    assert d["reason"] == expl.summary
    print(f"  as_dict() carries why ✓  e.g. \"{d['tracks'][0]['extra']['why'][:60]}…\"")

    # 4. Happy path routes to happy docs (different mood → different grounding).
    rec_h = recommender.recommend(dist(happy=0.85), "so happy today", length=3)
    eh = rag.explain(rec_h)
    assert "emotion:happy" in {c for t in eh.tracks for c in t.citations}
    print(f"  happy path grounding → emotion:happy cited ✓")

    # 5. Fail-open: a broken explainer leaves the recommendation usable.
    class _Broken(GroundedTemplateExplainer):
        def explain(self, recommendation):
            raise RuntimeError("boom")
    safe = AetherExplainer(explainer=_Broken(rag.retriever), retriever=rag.retriever)
    before = rec_h.reason
    out = safe.annotate(rec_h)
    assert out is rec_h and rec_h.reason == before   # unchanged, still usable
    print("  fail-open on explainer error ✓")

    print("-" * 55)
    print("✅ All RAG facade self-tests passed.")


if __name__ == "__main__":
    _selftest()
