"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 5 · Knowledge Base (grounding corpus for RAG)
═══════════════════════════════════════════════════════════════════
The corpus the explainer RETRIEVES over. Every "why these songs" sentence is
grounded in a document here, so explanations are faithful and non-hallucinated.

Design decisions (interview-defensible)
---------------------------------------
1. Grounded in AUDIO FEATURES, not lyrics. Aether matches songs by their audio
   feature vector (tempo/energy/valence/danceability/acousticness/
   instrumentalness) against a per-emotion target — see config.EMOTION_MUSIC_
   TARGETS. A truthful explanation must therefore reason about *those* features,
   not song lyrics the system never looked at. This also keeps the corpus
   copyright-clean (no reproduced lyrics).
2. Derived from the single source of truth. The per-emotion profile docs read
   their numbers straight from config.EMOTION_MUSIC_TARGETS, so the knowledge
   base can never drift from the vectors the matcher actually scores against.
   Change a target in config → the explanation's facts change with it.
3. Two document families:
     • emotion  → one doc per Aether emotion (its musical fingerprint + the
                  psychology of why that fingerprint reads as that feeling).
     • feature  → one doc per match-feature (what the dimension means
                  emotionally, grounded in the valence–arousal circumplex model
                  of affect — Russell, 1980).
   Both are needed: the feature docs explain the *mechanism*, the emotion docs
   explain the *target*, and retrieval surfaces whichever is most relevant to
   the track being explained.

The whole module is pure Python, deterministic, and offline — no data files,
no network, no heavy deps. Building the base is effectively free.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Path bootstrap: make Aether's root config importable from the phase dir ──
_ROOT = Path(__file__).resolve().parent.parent          # …/Aether
if _ROOT.exists() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from config import (                                  # type: ignore
        AETHER_EMOTIONS, EMOTION_MUSIC_TARGETS,
    )
except Exception:  # pragma: no cover — allow standalone import in odd layouts
    AETHER_EMOTIONS = []
    EMOTION_MUSIC_TARGETS = {}


# ──────────────────────────────────────────────
# Document type
# ──────────────────────────────────────────────
@dataclass
class KnowledgeDoc:
    """
    One retrievable unit of grounding knowledge.

    Attributes:
        doc_id:  Stable unique id (e.g. "emotion:sad", "feature:energy").
        title:   Short human title (shown in citations).
        text:    The body the retriever embeds/searches AND the explainer cites.
        kind:    "emotion" | "feature" | "concept".
        emotion: The Aether emotion this doc is about (emotion docs only).
        feature: The match-feature this doc is about (feature docs only).
        tags:    Extra keywords to strengthen lexical retrieval.
    """

    doc_id: str
    title: str
    text: str
    kind: str
    emotion: Optional[str] = None
    feature: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def searchable_text(self) -> str:
        """Text used for retrieval — body plus title and tags for recall."""
        return f"{self.title}. {self.text} {' '.join(self.tags)}".strip()


# ──────────────────────────────────────────────
# Curated psychology (original prose — no copyrighted text)
# ──────────────────────────────────────────────
# The emotional *meaning* of each match-feature, grounded in the valence–arousal
# circumplex model of affect (Russell, 1980): most feelings sit somewhere on two
# axes — how pleasant (valence) and how activated (arousal) they are. Audio
# features are proxies for those axes.
_FEATURE_PSYCHOLOGY: dict[str, str] = {
    "tempo": (
        "Tempo (beats per minute) is the clearest proxy for arousal. Fast "
        "tempos raise heart rate and perceived urgency and read as high-energy "
        "or agitated states; slow tempos lower arousal and read as calm, sad, "
        "or reflective. It sets the physical pace a listener's body entrains to."
    ),
    "energy": (
        "Energy captures a track's overall intensity and activity — loud, "
        "dense, fast-attacking music scores high, sparse and gentle music "
        "scores low. It is the primary arousal dimension: high energy suits "
        "excited, angry, or confident states; low energy suits calm, sad, or "
        "lonely ones."
    ),
    "valence": (
        "Valence measures musical positivity — how happy, cheerful, or bright a "
        "track sounds versus sad, dark, or negative. It is the pleasantness "
        "axis of affect: high valence pairs with joy, hope, and love; low "
        "valence pairs with grief, anger, and melancholy, largely via major "
        "versus minor tonality and bright versus dark timbre."
    ),
    "danceability": (
        "Danceability reflects how steady and groove-driven the rhythm is — "
        "strong, regular beats invite movement. High danceability reinforces "
        "upbeat, social, energetic moods; low danceability suits introspective "
        "or somber listening where the body stays still."
    ),
    "acousticness": (
        "Acousticness is the confidence that a track is acoustic rather than "
        "electronic. Acoustic, organic timbres feel intimate, warm, and human — "
        "fitting calm, nostalgic, romantic, or lonely states — while electronic "
        "production feels more energetic, modern, or intense."
    ),
    "instrumentalness": (
        "Instrumentalness predicts the absence of vocals. Instrumental music "
        "leaves cognitive room for concentration and reverie, so it suits "
        "focused, dreamy, and calm states; vocal-forward tracks foreground "
        "narrative and direct emotional address."
    ),
}

# A short, evocative sonic fingerprint per emotion (original phrasing). Keeps the
# emotion docs vivid without restating config numbers as bare digits.
_EMOTION_SONICS: dict[str, str] = {
    "happy":       "bright, upbeat, major-key pop that feels celebratory",
    "sad":         "slow, sparse ballads in minor keys with heavy space",
    "angry":       "fast, distorted, aggressive music with hard-hitting drums",
    "calm":        "gentle, acoustic, ambient textures that lower the pulse",
    "anxious":     "tense, restless, mid-tempo music that never quite settles",
    "energetic":   "high-BPM dance and EDM built for movement and adrenaline",
    "focused":     "minimal, steady, largely instrumental beats that stay in the background",
    "nostalgic":   "warm, acoustic, retro-tinged songs that look backward fondly",
    "romantic":    "tender, intimate slow jams and love songs",
    "melancholic": "dark, slow, deeply minor music with existential weight",
    "confident":   "bold, bass-heavy anthems that stride forward",
    "hopeful":     "uplifting, building, major-key songs that reach upward",
    "frustrated":  "hard, dissonant, driving rock that grinds against a wall",
    "lonely":      "minimal, echo-heavy, sparse music full of empty space",
    "dreamy":      "ethereal, atmospheric synth textures that float and drift",
}


# ──────────────────────────────────────────────
# Fact extraction from the single source of truth
# ──────────────────────────────────────────────
def _qual(feature: str, value: float) -> str:
    """Qualitative band ('very low'…'very high') for a feature value.

    Tempo is judged in BPM; every other match-feature is on a 0–1 scale.
    """
    if feature == "tempo":
        # BPM bands chosen to match how the emotion targets cluster.
        if value < 75:   return "very slow"
        if value < 95:   return "slow"
        if value < 115:  return "moderate"
        if value < 130:  return "fast"
        return "very fast"
    # 0–1 features
    if value < 0.20: return "very low"
    if value < 0.40: return "low"
    if value < 0.60: return "moderate"
    if value < 0.80: return "high"
    return "very high"


def _emotion_profile_sentence(emotion: str) -> str:
    """One faithful sentence describing an emotion's target feature profile."""
    t = EMOTION_MUSIC_TARGETS.get(emotion, {})
    if not t:
        return f"{emotion.capitalize()} maps to a distinct musical profile."
    parts = []
    for feat in ("tempo", "energy", "valence", "danceability",
                 "acousticness", "instrumentalness"):
        if feat in t:
            v = t[feat]
            shown = f"{v:.0f} BPM" if feat == "tempo" else f"{v:.2f}"
            parts.append(f"{_qual(feat, v)} {feat} ({shown})")
    return "Its target profile is " + ", ".join(parts) + "."


# ──────────────────────────────────────────────
# Corpus builder
# ──────────────────────────────────────────────
def build_knowledge_base() -> list[KnowledgeDoc]:
    """
    Assemble the full grounding corpus: one doc per emotion, one per feature,
    plus a couple of cross-cutting concept docs.

    Returns:
        A deterministic list of KnowledgeDoc (order stable across runs).
    """
    docs: list[KnowledgeDoc] = []

    # ── Concept docs (the mechanism the whole system rests on) ──
    docs.append(KnowledgeDoc(
        doc_id="concept:circumplex",
        title="The valence–arousal model of musical emotion",
        text=(
            "Aether locates every emotion on two axes: valence (how pleasant "
            "the feeling is) and arousal (how activated it is). Audio features "
            "are proxies for these axes — energy and tempo track arousal, while "
            "valence and tonality track pleasantness. Matching a song to a mood "
            "means finding a track whose feature vector sits near the mood's "
            "target point on this map."
        ),
        kind="concept",
        tags=["arousal", "valence", "affect", "mood", "why", "match"],
    ))
    docs.append(KnowledgeDoc(
        doc_id="concept:cosine-match",
        title="How Aether picks each song",
        text=(
            "Each Aether emotion has a target feature vector. Every song in the "
            "catalog is stored as a comparable feature vector. Aether ranks "
            "songs by nearest-neighbor similarity to the target, so the "
            "top picks are the tracks whose measured tempo, energy, valence, "
            "danceability, acousticness, and instrumentalness most closely "
            "resemble the emotion's ideal. A higher match score means a tighter "
            "fit to that target."
        ),
        kind="concept",
        tags=["cosine", "similarity", "match score", "nearest neighbor", "why"],
    ))
    docs.append(KnowledgeDoc(
        doc_id="concept:energy-arc",
        title="Why the playlist is ordered the way it is",
        text=(
            "Beyond picking the right songs, Aether sequences them into an "
            "energy arc. High-arousal moods often rise then settle; low moods "
            "wind down gently; steady moods hold a level. Ordering by energy "
            "gives the playlist emotional shape rather than a random jumble, so "
            "the listen has a beginning, a middle, and a resolution."
        ),
        kind="concept",
        tags=["arc", "sequence", "order", "energy", "flow", "why"],
    ))

    # ── Feature docs (the emotional meaning of each dimension) ──
    for feat, text in _FEATURE_PSYCHOLOGY.items():
        docs.append(KnowledgeDoc(
            doc_id=f"feature:{feat}",
            title=f"What {feat} means emotionally",
            text=text,
            kind="feature",
            feature=feat,
            tags=[feat, "audio feature", "why"],
        ))

    # ── Emotion docs (target profile + psychology, one per Aether emotion) ──
    emotions = AETHER_EMOTIONS or list(EMOTION_MUSIC_TARGETS.keys())
    for emo in emotions:
        sonic = _EMOTION_SONICS.get(emo, "a distinct musical character")
        profile = _emotion_profile_sentence(emo)
        article = "An" if emo[:1].lower() in "aeiou" else "A"
        docs.append(KnowledgeDoc(
            doc_id=f"emotion:{emo}",
            title=f"The sound of {emo}",
            text=(
                f"{article} {emo} mood is served by {sonic}. {profile} "
                f"Songs are chosen when their audio features land close to this "
                f"target, which is what makes them read as {emo}."
            ),
            kind="emotion",
            emotion=emo,
            tags=[emo, "emotion", "mood", "why", sonic],
        ))

    return docs


# Convenience: index by id for O(1) citation lookup.
def index_by_id(docs: list[KnowledgeDoc]) -> dict[str, KnowledgeDoc]:
    """Map doc_id → KnowledgeDoc."""
    return {d.doc_id: d for d in docs}


# ─────────────────────────────────────────────────────────────
# Self-test — pure, offline
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Knowledge base self-test")
    print("-" * 55)

    docs = build_knowledge_base()
    by_id = index_by_id(docs)

    # 1. One doc per emotion (all 15) and per feature (all 6).
    emo_docs = [d for d in docs if d.kind == "emotion"]
    feat_docs = [d for d in docs if d.kind == "feature"]
    n_emos = len(AETHER_EMOTIONS) if AETHER_EMOTIONS else 15
    assert len(emo_docs) == n_emos, (len(emo_docs), n_emos)
    assert len(feat_docs) == 6, len(feat_docs)
    print(f"  {len(emo_docs)} emotion docs + {len(feat_docs)} feature docs "
          f"+ {len(docs) - len(emo_docs) - len(feat_docs)} concept docs "
          f"= {len(docs)} total")

    # 2. Ids are unique and stable.
    assert len(by_id) == len(docs), "duplicate doc_id"

    # 3. Emotion docs are FAITHFUL — the exact target numbers appear in the text.
    if EMOTION_MUSIC_TARGETS:
        sad = by_id["emotion:sad"].text
        assert "70 BPM" in sad, sad          # config sad tempo == 70
        assert "0.15" in sad, sad            # config sad valence == 0.15
        assert "low energy" in sad, sad       # 0.20 → 'low' band
        assert "very low valence" in sad, sad  # 0.15 → 'very low' band
        print(f"  faithfulness ✓  emotion:sad → \"{sad[:72]}…\"")

    # 4. Feature docs cover every match-feature.
    covered = {d.feature for d in feat_docs}
    assert covered == set(_FEATURE_PSYCHOLOGY), covered
    print(f"  feature coverage ✓  {sorted(covered)}")

    # 5. searchable_text is non-empty for every doc.
    assert all(d.searchable_text() for d in docs)

    print("-" * 55)
    print("✅ All knowledge-base self-tests passed.")


if __name__ == "__main__":
    _selftest()
