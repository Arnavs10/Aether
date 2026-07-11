"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 5 · Explanation Seam (the "G" in RAG)
═══════════════════════════════════════════════════════════════════
Turns a finished Phase 4 Recommendation into a grounded "why these songs"
explanation — at the playlist level and per track — using ONLY facts retrieved
from the knowledge base plus each track's own measured audio features.

Two interchangeable generators behind one interface (same pattern as the rest
of Aether):

  • GroundedTemplateExplainer — the default. Deterministic, offline, no API key.
      For each track it (1) retrieves the most relevant knowledge docs, (2)
      compares the track's normalized energy/valence/tempo against the SAME
      normalized emotion target the matcher scored against, and (3) composes a
      faithful sentence that cites the retrieved knowledge. Because it only ever
      restates retrieved facts and measured numbers, it cannot hallucinate.

  • LLMExplainer — optional. Given an injected `llm_fn(prompt) -> str` (wrap
      Anthropic / OpenAI / a local model), it hands the SAME retrieved context
      to an LLM for more fluent prose. It is strictly grounding-constrained by
      the prompt, and falls back to the template generator on any error or when
      no llm_fn is supplied — so the pipeline is never broken by a missing key.

Faithfulness note
-----------------
Track features are re-hydrated by Phase 4 as the store's NORMALIZED 0–1 values.
We normalize the emotion target with feature_store.normalize_emotion_target —
the exact function the matcher uses — so "close to target" here means the same
thing it meant at match time. Blend labels ("happy+sad") average their targets,
mirroring the matcher's blended target.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# ── Path bootstrap: root config + Phase 2 (normalizer/schema) + local modules ──
_ROOT = Path(__file__).resolve().parent.parent
_P2 = _ROOT / "phase_2_music_data"
_P4 = _ROOT / "phase_4_recommendation"
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _P2, _P4, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from knowledge import KnowledgeDoc                          # noqa: E402
from retriever import Retriever, RetrievedDoc               # noqa: E402

try:
    from config import EMOTION_MUSIC_TARGETS, RAG_TOP_K     # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    EMOTION_MUSIC_TARGETS, RAG_TOP_K = {}, 10

# Reuse the matcher's normalization so "closeness" is measured identically.
try:
    from feature_store import normalize_emotion_target      # type: ignore  # noqa: E402
    from schema import MATCH_FEATURES                        # type: ignore  # noqa: E402
except Exception:  # pragma: no cover — degrade to feature-agnostic explanations
    normalize_emotion_target = None                          # type: ignore
    MATCH_FEATURES = ["tempo", "energy", "valence", "danceability",
                      "acousticness", "instrumentalness"]

_TEMPO_I = MATCH_FEATURES.index("tempo")
_ENERGY_I = MATCH_FEATURES.index("energy")
_VALENCE_I = MATCH_FEATURES.index("valence")


# ──────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────
@dataclass
class TrackExplanation:
    """Why one track was chosen."""
    title: str
    artist: str
    emotion: str
    why: str                                               # plain-language, no jargon
    why_technical: str = ""                                # feature-grounded, on-demand
    citations: list[str] = field(default_factory=list)     # doc_ids grounding it
    feature_notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "artist": self.artist, "emotion": self.emotion,
            "why": self.why, "why_technical": self.why_technical,
            "citations": self.citations, "feature_notes": self.feature_notes,
        }


@dataclass
class Explanation:
    """The full 'why' for a recommendation: a summary plus per-track reasons."""
    summary: str
    tracks: list[TrackExplanation]
    citations: list[dict[str, str]] = field(default_factory=list)  # [{id,title}]

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "tracks": [t.as_dict() for t in self.tracks],
            "citations": self.citations,
        }


# ──────────────────────────────────────────────
# Feature-vs-target helpers (faithful to the matcher)
# ──────────────────────────────────────────────
def _qual_norm(value: float) -> str:
    """Qualitative band for a NORMALIZED 0–1 value."""
    if value < 0.20: return "very low"
    if value < 0.40: return "low"
    if value < 0.60: return "moderate"
    if value < 0.80: return "high"
    return "very high"


def _effective_target(emotion_label: str) -> Optional[np.ndarray]:
    """
    Normalized 6-dim target for an emotion label, averaging components of a
    blend label ("happy+sad"). Returns None if unknown or normalizer missing.
    """
    if normalize_emotion_target is None or not EMOTION_MUSIC_TARGETS:
        return None
    parts = [e.strip() for e in str(emotion_label).split("+") if e.strip()]
    vecs = []
    for e in parts:
        tgt = EMOTION_MUSIC_TARGETS.get(e)
        if tgt is not None:
            vecs.append(normalize_emotion_target(tgt))
    if not vecs:
        return None
    return np.mean(np.stack(vecs), axis=0)


def _signature_feature(target_norm: np.ndarray) -> str:
    """The feature that most defines this target (farthest from neutral 0.5)."""
    idx = int(np.argmax(np.abs(target_norm - 0.5)))
    return MATCH_FEATURES[idx]


def _first_sentence(text: str) -> str:
    """First sentence of a doc body (for compact grounding)."""
    for sep in (". ", "; "):
        if sep in text:
            return text.split(sep)[0].strip().rstrip(".") + "."
    return text.strip()


# ──────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────
class Explainer(ABC):
    """Abstract 'why these songs' generator over a retriever + knowledge base."""

    name: str = "abstract"

    def __init__(self, retriever: Retriever, k: int = RAG_TOP_K):
        self.retriever = retriever
        self.k = k

    @abstractmethod
    def explain(self, recommendation: Any) -> Explanation:
        """Produce a full Explanation for a Phase 4 Recommendation."""
        raise NotImplementedError

    # ── shared retrieval + feature analysis (used by both concrete explainers) ──
    def _analyze_track(self, track: Any) -> dict[str, Any]:
        """
        Retrieve grounding docs for a track and measure it against its emotion
        target. Returns a dict both generators consume.
        """
        emotion = track.source_emotion or "calm"
        target = _effective_target(emotion)

        # Qualitative bands for the track's re-hydrated features (may be None).
        bands: dict[str, str] = {}
        for feat, val in (("energy", track.energy), ("valence", track.valence),
                          ("tempo", track.tempo)):
            if val is not None:
                bands[feat] = _qual_norm(float(val))

        # Signature feature drives the retrieval query (what defines this mood).
        sig = _signature_feature(target) if target is not None else "energy"
        band_terms = " ".join(f"{k} {v}" for k, v in bands.items())
        query = (f"{emotion.replace('+', ' ')} mood {sig} "
                 f"{band_terms} why this song fits").strip()
        hits = self.retriever.query(query, k=self.k)

        # Per-feature closeness to target (only for features we have).
        deltas: dict[str, dict[str, float]] = {}
        if target is not None:
            for feat, idx, val in (("energy", _ENERGY_I, track.energy),
                                   ("valence", _VALENCE_I, track.valence),
                                   ("tempo", _TEMPO_I, track.tempo)):
                if val is not None:
                    tv = float(target[idx])
                    deltas[feat] = {
                        "track": round(float(val), 3),
                        "target": round(tv, 3),
                        "closeness": round(1.0 - abs(float(val) - tv), 3),
                    }

        return {
            "emotion": emotion, "signature": sig, "bands": bands,
            "hits": hits, "deltas": deltas,
            "is_fresh": bool(str(track.track_id or "").startswith(("fake:", "deezer:", "itunes:")))
                        or (track.energy is None and bool(track.provider_ref)),
        }

    def _feature_doc(self, feature: str, hits: list[RetrievedDoc]):
        """
        Return the knowledge doc for `feature`, so the mechanism we quote always
        matches the feature we quantified. Prefers a doc already retrieved (it's
        contextually relevant); otherwise does a cheap targeted lookup; None if
        the corpus has no such doc.
        """
        for h in hits:
            if h.doc.feature == feature:
                return h.doc
        for h in self.retriever.query(f"what {feature} means emotionally", k=3):
            if h.doc.feature == feature:
                return h.doc
        # Last resort: any feature doc among the hits (keeps output grounded).
        return next((h.doc for h in hits if h.doc.kind == "feature"), None)


# ──────────────────────────────────────────────
# Default generator — grounded template (offline, deterministic)
# ──────────────────────────────────────────────
class GroundedTemplateExplainer(Explainer):
    """Composes faithful explanations by restating retrieved facts + measured features."""

    name = "grounded_template"

    def explain(self, recommendation: Any) -> Explanation:
        tracks = list(recommendation.tracks)
        used: dict[str, str] = {}                 # doc_id → title (dedup citations)

        track_expls: list[TrackExplanation] = []
        for tr in tracks:
            info = self._analyze_track(tr)
            expl = self._compose_track(tr, info)
            track_expls.append(expl)
            for h in info["hits"][:2]:
                used[h.doc.doc_id] = h.doc.title

        summary = self._compose_summary(recommendation, used)
        citations = [{"id": did, "title": title} for did, title in used.items()]
        return Explanation(summary=summary, tracks=track_expls, citations=citations)

    # ── per-track prose ──
    def _compose_track(self, track: Any, info: dict[str, Any]) -> TrackExplanation:
        emotion = info["emotion"]
        hits: list[RetrievedDoc] = info["hits"]
        emo_doc = next((h.doc for h in hits if h.doc.kind == "emotion"), None)

        cites: list[str] = []
        pieces: list[str] = []

        pretty_emo = emotion.replace("+", " + ")
        pieces.append(f"'{track.title}' by {track.artist} was picked for your "
                      f"{pretty_emo} mood.")

        # Ground the mood in the emotion doc.
        if emo_doc is not None:
            pieces.append(_first_sentence(emo_doc.text))
            cites.append(emo_doc.doc_id)

        # Quantified feature match (the strongest evidence). Ground the mechanism
        # in the feature doc for the SAME feature we quantify, so the number and
        # the explanation always agree.
        deltas: dict[str, dict[str, float]] = info["deltas"]
        if deltas:
            feat, d = max(deltas.items(), key=lambda kv: kv[1]["closeness"])
            band = info["bands"].get(feat, _qual_norm(d["track"]))
            pieces.append(
                f"Its {band} {feat} ({d['track']:.2f}) sits close to the "
                f"{pretty_emo} target ({d['target']:.2f})."
            )
            feat_doc = self._feature_doc(feat, hits)
            if feat_doc is not None:
                pieces.append(_first_sentence(feat_doc.text))
                cites.append(feat_doc.doc_id)
        elif info["is_fresh"]:
            pieces.append("It's a fresh, current-catalog pick surfaced for this "
                          "mood rather than a feature-matched library track.")
            sig_doc = self._feature_doc(info["signature"], hits)
            if sig_doc is not None:
                cites.append(sig_doc.doc_id)

        if track.match_score is not None:
            pieces.append(f"Match score: {track.match_score:.2f}.")

        return TrackExplanation(
            title=track.title, artist=track.artist, emotion=emotion,
            why=" ".join(pieces), citations=list(dict.fromkeys(cites)),
            feature_notes=deltas,
        )

    # ── playlist-level prose ──
    def _compose_summary(self, rec: Any, used: dict[str, str]) -> str:
        hits = self.retriever.query(
            "why these songs match my mood how are they ordered energy arc", k=self.k
        )
        concept = next((h.doc for h in hits if h.doc.kind == "concept"), None)

        dom = rec.dominant_emotions[0][0] if rec.dominant_emotions else "your"
        arc_phrase = {
            "arc": "shaped as a rise-and-settle energy arc",
            "ascending": "ordered as a steady energy build",
            "descending": "ordered as a gentle wind-down",
            "steady": "kept in relevance order",
        }.get(rec.arc_shape, "sequenced")

        parts = [
            f"These {rec.size} tracks match your {rec.intensity_label} "
            f"'{dom}' mood.",
        ]
        if concept is not None:
            parts.append(_first_sentence(concept.text))
            used[concept.doc_id] = concept.title
        parts.append(f"They're {arc_phrase} so the playlist has emotional shape.")
        return " ".join(parts)


# ──────────────────────────────────────────────
# Optional generator — LLM (injected, grounding-constrained, graceful fallback)
# ──────────────────────────────────────────────
class LLMExplainer(Explainer):
    """
    Fluent explanations via an injected `llm_fn(prompt) -> str`. Strictly
    grounded by the prompt (uses only retrieved context + measured features) and
    falls back to GroundedTemplateExplainer whenever llm_fn is absent or errors.
    """

    name = "llm"

    def __init__(self, retriever: Retriever, llm_fn: Optional[Callable[[str], str]] = None,
                 k: int = RAG_TOP_K):
        super().__init__(retriever, k=k)
        self.llm_fn = llm_fn
        self._fallback = GroundedTemplateExplainer(retriever, k=k)

    def explain(self, recommendation: Any) -> Explanation:
        if self.llm_fn is None:
            return self._fallback.explain(recommendation)

        used: dict[str, str] = {}
        track_expls: list[TrackExplanation] = []
        for tr in recommendation.tracks:
            info = self._analyze_track(tr)
            for h in info["hits"][:2]:
                used[h.doc.doc_id] = h.doc.title
            try:
                why_simple, why_tech = self._llm_track(tr, info)
                cites = [h.doc.doc_id for h in info["hits"][:2]]
                track_expls.append(TrackExplanation(
                    title=tr.title, artist=tr.artist, emotion=info["emotion"],
                    why=why_simple.strip(), why_technical=why_tech.strip(),
                    citations=cites, feature_notes=info["deltas"],
                ))
            except Exception:
                # Any LLM failure → deterministic grounded fallback for this track.
                track_expls.append(self._fallback._compose_track(tr, info))

        try:
            summary = self._llm_summary(recommendation)
        except Exception:
            summary = self._fallback._compose_summary(recommendation, used)

        citations = [{"id": did, "title": title} for did, title in used.items()]
        return Explanation(summary=summary, tracks=track_expls, citations=citations)

    # ── prompt construction (grounding-constrained) ──
    def _llm_track(self, track: Any, info: dict[str, Any]) -> tuple[str, str]:
        """Return (simple, technical) explanations from a single LLM call."""
        context = "\n".join(f"- {h.doc.title}: {h.doc.text}" for h in info["hits"][:3])
        deltas = "; ".join(
            f"{f}: track {d['track']:.2f} vs {info['emotion']} target {d['target']:.2f}"
            for f, d in info["deltas"].items()
        ) or "no feature data"
        prompt = (
            "You explain why a song fits a listener's mood. Use ONLY the context "
            "and measurements below. Do not invent facts about the song.\n\n"
            "Write TWO explanations:\n"
            "SIMPLE: one warm, plain-language sentence anyone can understand. "
            "Describe how the song FEELS (gentle, upbeat, slow, tense, bright, "
            "mellow…) and why that suits the mood. Do NOT use numbers or the words "
            "'valence', 'tempo', 'energy', or 'arousal'.\n"
            "TECHNICAL: one or two sentences grounded in the measurements, naming "
            "the audio features and how they match the target.\n\n"
            f"MOOD: {info['emotion']}\n"
            f"SONG: '{track.title}' by {track.artist}\n"
            f"MEASUREMENTS: {deltas}\n"
            f"CONTEXT:\n{context}\n\n"
            "Respond in EXACTLY this format:\n"
            "SIMPLE: <one sentence>\n"
            "TECHNICAL: <one or two sentences>"
        )
        raw = self.llm_fn(prompt)  # type: ignore[misc]
        return self._split_two(raw)

    @staticmethod
    def _split_two(text: str) -> tuple[str, str]:
        """Parse 'SIMPLE: … TECHNICAL: …' → (simple, technical), robust to drift."""
        simple, technical = "", ""
        if "TECHNICAL:" in text:
            pre, tech = text.split("TECHNICAL:", 1)
            technical = tech.strip()
            simple = (pre.split("SIMPLE:", 1)[1] if "SIMPLE:" in pre else pre).strip()
        else:
            # No marker → treat the whole reply as the simple explanation.
            simple = text.replace("SIMPLE:", "").strip()
        if not simple:
            simple = technical            # never leave the visible line empty
        return simple, technical

    def _llm_summary(self, rec: Any) -> str:
        hits = self.retriever.query("why these songs mood energy arc order", k=self.k)
        context = "\n".join(f"- {h.doc.title}: {h.doc.text}"
                            for h in hits if h.doc.kind == "concept")
        dom = rec.dominant_emotions[0][0] if rec.dominant_emotions else "the"
        prompt = (
            "Summarize why this playlist fits the listener, in 1–2 sentences, "
            "using ONLY the context. Be concrete.\n\n"
            f"MOOD: {dom} ({rec.intensity_label}); ARC: {rec.arc_shape}; "
            f"{rec.size} tracks.\nCONTEXT:\n{context}\n\nSummary:"
        )
        return self.llm_fn(prompt).strip()  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────
# Self-test — builds real Track/Recommendation objects, offline TF-IDF retriever
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Explainer self-test")
    print("-" * 55)

    from retriever import build_retriever
    from models import Track, Recommendation  # Phase 4 real types

    retriever = build_retriever(prefer_chroma=False)
    explainer = GroundedTemplateExplainer(retriever, k=4)

    # A small, realistic 'sad' recommendation (normalized features ~ sad target).
    tracks = [
        Track(title="Rain", artist="D", track_id="s1", year=2001,
              energy=0.19, valence=0.14, tempo=0.12, match_score=0.94,
              source_emotion="sad", rank=1),
        Track(title="Grey Skies", artist="F", track_id="s3", year=1999,
              energy=0.17, valence=0.11, tempo=0.10, match_score=0.91,
              source_emotion="sad", rank=2),
    ]
    rec = Recommendation(
        tracks=tracks, request_text="i feel so down",
        intent_mode="single", intensity_level=3, intensity_label="intense",
        dominant_emotions=[("sad", 1.0)], arc_shape="descending",
        reason="An intense 'sad' playlist.",
    )

    expl = explainer.explain(rec)

    # 1. One explanation per track, all non-empty.
    assert len(expl.tracks) == 2, len(expl.tracks)
    assert all(t.why for t in expl.tracks)
    print(f"  summary → {expl.summary}")
    print(f"  track[0] → {expl.tracks[0].why}")

    # 2. Explanations are GROUNDED — every track cites ≥1 knowledge doc.
    assert all(t.citations for t in expl.tracks), \
        [t.citations for t in expl.tracks]
    print(f"  track[0] citations → {expl.tracks[0].citations}")

    # 3. Faithfulness — the sad emotion doc is among citations (routed correctly).
    all_cites = {c for t in expl.tracks for c in t.citations}
    assert "emotion:sad" in all_cites, all_cites

    # 4. Feature notes quantify closeness to the SAME normalized target.
    fn = expl.tracks[0].feature_notes
    assert {"energy", "valence", "tempo"} <= set(fn), fn
    assert all(0.0 <= v["closeness"] <= 1.0 for v in fn.values()), fn
    assert fn["valence"]["closeness"] > 0.8, fn["valence"]  # tight sad match
    print(f"  feature_notes[valence] → {fn['valence']}")

    # 5. Playlist-level citations collected.
    assert expl.citations, expl.citations
    print(f"  {len(expl.citations)} citations gathered")

    # 6. as_dict() is API-ready.
    d = expl.as_dict()
    assert {"summary", "tracks", "citations"} <= set(d)
    assert d["tracks"][0]["title"] == "Rain"

    # 7. Fresh pick (no features) still explained, no crash.
    fresh = Track(title="New Song", artist="NewArtist", track_id="deezer:sad:0",
                  source_emotion="sad", provider_ref={"preview_url": "x"}, rank=3)
    rec2 = Recommendation(
        tracks=[fresh], request_text="down", intent_mode="single",
        intensity_level=2, intensity_label="moderate",
        dominant_emotions=[("sad", 1.0)], arc_shape="descending", reason="x",
    )
    e2 = explainer.explain(rec2)
    assert e2.tracks[0].why and e2.tracks[0].citations
    print(f"  fresh pick → {e2.tracks[0].why[:70]}…")

    # 8. LLMExplainer with no llm_fn falls back cleanly to template output.
    llm = LLMExplainer(retriever, llm_fn=None, k=4)
    e3 = llm.explain(rec)
    assert e3.tracks and e3.tracks[0].why
    print("  LLMExplainer(no key) → fell back to grounded template ✓")

    # 9. LLMExplainer with a fake llm_fn uses it (and stays grounded via prompt).
    def fake_llm(prompt: str) -> str:
        assert "Use ONLY the context" in prompt or "ONLY the context" in prompt
        return "A tender, low-energy fit for the mood."
    llm2 = LLMExplainer(retriever, llm_fn=fake_llm, k=4)
    e4 = llm2.explain(rec)
    assert e4.tracks[0].why == "A tender, low-energy fit for the mood."
    print(f"  LLMExplainer(fake fn) → \"{e4.tracks[0].why}\"")

    print("-" * 55)
    print("✅ All explainer self-tests passed.")


if __name__ == "__main__":
    _selftest()
