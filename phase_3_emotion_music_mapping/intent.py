"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 3: Request Intent Parser  (English + Hindi)
═══════════════════════════════════════════════════════════════════

Turns a user's free-form request (text — English or Hindi, or a transcript of
their voice) plus the fused 15-emotion distribution from Phase 1C into a
structured `MatchIntent` that tells the matcher HOW to build the playlist.

The core distinction (raised by Arnav and correct):

    "nostalgic BUT hopeful"          → ONE blended mood        → mode = "blend"
    "SOME nostalgic AND SOME hopeful" → TWO distinct song-sets  → mode = "mix"

We detect this from CONNECTIVE words in the raw text, in both languages:

    blend connectives : but, yet, -ish, though, however |
                        लेकिन, पर, मगर, फिर भी
    mix   connectives : and, some…and some, mix of, both, along with, plus |
                        और, कुछ…कुछ, थोड़े…थोड़े, मिक्स, दोनों, साथ में

Which EMOTIONS to use comes from the fused distribution (not from re-parsing
text): we take the top-N emotions above a threshold. This keeps Phase 3
grounded in the actual Phase 1 model output rather than brittle keyword
emotion-spotting — the models already did the hard "what emotion is this"
work; here we only decide blend-vs-mix and the weighting.

Design note (interview-defensible): this is a rule/keyword detector — fast,
transparent, no extra model, and correct for the common phrasings in both
languages. The Phase 5 RAG/LLM layer can later upgrade intent parsing for
rarer or ambiguous phrasings. A `mode` override lets the UI/caller force a
choice, bypassing detection entirely.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from config import AETHER_EMOTIONS  # type: ignore
except Exception:  # pragma: no cover
    AETHER_EMOTIONS = [
        "happy", "sad", "angry", "calm", "anxious",
        "energetic", "focused", "nostalgic", "romantic",
        "melancholic", "confident", "hopeful", "frustrated",
        "lonely", "dreamy",
    ]

NUM_EMOTIONS = len(AETHER_EMOTIONS)

# Modes.
MODE_BLEND = "blend"   # single averaged target
MODE_MIX = "mix"       # separate targets, interleaved
MODE_SINGLE = "single" # one dominant emotion
VALID_MODES = (MODE_BLEND, MODE_MIX, MODE_SINGLE)

# ─────────────────────────────────────────────────────────────
# Connective lexicons (English + Hindi, incl. common romanized Hindi)
# ─────────────────────────────────────────────────────────────
# "mix" connectives → user wants DISTINCT sets of each emotion.
_MIX_CONNECTIVES = [
    # English
    r"\bsome\b.*\band\b.*\bsome\b",   # "some X and some Y"
    r"\bmix of\b", r"\bboth\b", r"\balong with\b", r"\bas well as\b",
    r"\bplus\b", r"\ba bit of\b.*\band\b",
    r"\band some\b", r"\band a few\b",
    # Hindi (Devanagari)
    r"कुछ.*और.*कुछ",      # "kuch X aur kuch Y"
    r"थोड़े.*थोड़े",        # "thode X thode Y"
    r"मिक्स", r"दोनों", r"साथ में", r"साथ मे",
    # Romanized Hindi
    r"\bkuch\b.*\baur\b.*\bkuch\b",
    r"\bthode\b.*\bthode\b", r"\bmix\b", r"\bdono\b", r"\bsaath\b",
]

# "blend" connectives → user wants ONE combined mood.
_BLEND_CONNECTIVES = [
    # English
    r"\bbut\b", r"\byet\b", r"\bthough\b", r"\bhowever\b",
    r"\bwith a hint of\b", r"\btinged with\b", r"\bbittersweet\b",
    r"-ish\b", r"\bkind of\b", r"\bslightly\b",
    # Hindi (Devanagari)
    r"लेकिन", r"पर\b", r"मगर", r"फिर भी", r"थोड़ा सा",
    # Romanized Hindi
    r"\blekin\b", r"\bmagar\b", r"\bphir bhi\b", r"\bpar\b",
]

# Plain "and" — weak signal. Treated as MIX only if no blend connective is
# present, since "X and Y" usually means "some of each".
_PLAIN_AND = [r"\band\b", r"\bऔर\b", r"\baur\b", r"\b&\b", r",\s*"]


@dataclass
class MatchIntent:
    """Structured result of parsing a request.

    Attributes:
        mode: "single" | "blend" | "mix".
        emotions: ordered list of emotion names to use (most→least weight).
        weights: parallel list of weights (sum to 1.0) for those emotions.
        detected_from: short reason string (which connective/branch fired),
            for transparency / debugging / the API.
        raw_text: the original request text (may be empty).
    """
    mode: str
    emotions: list[str]
    weights: list[float]
    detected_from: str = ""
    raw_text: str = ""

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "emotions": self.emotions,
            "weights": [round(float(w), 4) for w in self.weights],
            "detected_from": self.detected_from,
        }


# ─────────────────────────────────────────────────────────────
# Emotion selection from the fused distribution
# ─────────────────────────────────────────────────────────────
def _top_emotions(
    distribution: np.ndarray,
    max_emotions: int = 2,
    min_prob: float = 0.15,
    dominance_ratio: float = 2.5,
) -> tuple[list[str], list[float]]:
    """Pick the emotion(s) to build the playlist from, using the model output.

    Rules:
      • Always include the top emotion.
      • Include the 2nd emotion only if it is reasonably strong (>= min_prob)
        AND the top emotion is not overwhelmingly dominant
        (top / second < dominance_ratio). This prevents forcing a blend when
        one emotion clearly dominates.

    Args:
        distribution: (15,) probability vector over AETHER_EMOTIONS.
        max_emotions: hard cap on how many emotions to return.
        min_prob: minimum probability for a secondary emotion to count.
        dominance_ratio: if top/second >= this, treat as single-emotion.

    Returns:
        (emotion_names, weights) where weights are re-normalized to sum to 1.
    """
    dist = np.asarray(distribution, dtype=np.float64).flatten()
    if dist.shape[0] != NUM_EMOTIONS:
        raise ValueError(
            f"distribution must be length {NUM_EMOTIONS}, got {dist.shape[0]}."
        )
    dist = np.clip(dist, 0.0, None)
    if dist.sum() <= 1e-8:
        # No signal — fall back to a neutral default ("calm").
        return ["calm"], [1.0]
    dist = dist / dist.sum()

    order = np.argsort(-dist)
    top_idx = int(order[0])
    chosen = [top_idx]

    for j in range(1, min(max_emotions, NUM_EMOTIONS)):
        cand = int(order[j])
        p_top, p_cand = dist[chosen[0]], dist[cand]
        if p_cand < min_prob:
            break
        if p_top / (p_cand + 1e-8) >= dominance_ratio:
            break
        chosen.append(cand)

    names = [AETHER_EMOTIONS[i] for i in chosen]
    weights = np.array([dist[i] for i in chosen], dtype=np.float64)
    weights = weights / weights.sum()
    return names, [float(w) for w in weights]


# ─────────────────────────────────────────────────────────────
# Connective detection
# ─────────────────────────────────────────────────────────────
def _matches_any(text: str, patterns: list[str]) -> Optional[str]:
    """Return the first pattern that matches `text`, or None."""
    for pat in patterns:
        if re.search(pat, text):
            return pat
    return None


def _detect_mode(text: str, num_emotions: int) -> tuple[str, str]:
    """Decide blend vs mix vs single from the raw text.

    Priority:
      1. If only one emotion is in play → "single" (no connective needed).
      2. An explicit MIX connective → "mix".
      3. An explicit BLEND connective → "blend".
      4. A plain "and"/"," with two emotions → "mix" (default reading of
         "X and Y" as "some of each").
      5. Otherwise (two emotions, no clear connective) → "blend" (safer
         default: a nuanced single mood).

    Returns:
        (mode, reason)
    """
    if num_emotions <= 1:
        return MODE_SINGLE, "single dominant emotion"

    t = f" {text.lower().strip()} "

    mix_hit = _matches_any(t, _MIX_CONNECTIVES)
    if mix_hit:
        return MODE_MIX, f"mix connective: {mix_hit}"

    blend_hit = _matches_any(t, _BLEND_CONNECTIVES)
    if blend_hit:
        return MODE_BLEND, f"blend connective: {blend_hit}"

    plain_and = _matches_any(t, _PLAIN_AND)
    if plain_and:
        return MODE_MIX, f"plain conjunction: {plain_and}"

    return MODE_BLEND, "two emotions, no explicit connective → blended mood"


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Explicit user-specified proportions (override model-inferred weights)
# ─────────────────────────────────────────────────────────────
# Fuzzy quantifiers → an approximate share, in both languages. These apply to
# whichever emotion they sit near; the counterpart takes the remainder.
_FUZZY_QUANTIFIERS = {
    # strong majority (~0.8)
    "mostly": 0.8, "mainly": 0.8, "primarily": 0.8, "lots of": 0.8,
    "ज्यादातर": 0.8, "मुख्यतः": 0.8, "zyada": 0.8, "jyada": 0.8,
    # majority (~0.7)
    "more": 0.7, "mostly of": 0.7, "अधिक": 0.7,
    # small amount (~0.25)
    "a bit of": 0.25, "a little": 0.25, "some": 0.35, "few": 0.25,
    "थोड़ा": 0.25, "थोड़े": 0.25, "कुछ": 0.35, "thoda": 0.25, "thode": 0.25,
}


def _extract_explicit_ratios(text: str, emotions: list[str]) -> Optional[list[float]]:
    """Try to read user-stated proportions for the chosen emotions.

    Handles, for two emotions, patterns like:
        "70% happy 30% sad"     → percentages attached to emotion names
        "70-30" / "70:30"        → a bare ratio (applied in emotion order)
        "mostly happy, a bit sad"→ fuzzy quantifiers near each emotion

    Args:
        text: raw request (lowercased inside).
        emotions: the emotions chosen from the distribution (order matters for
            bare ratios).

    Returns:
        A weight list (sums to 1) aligned to `emotions`, or None if no explicit
        proportion was found (caller then keeps model-inferred weights).
    """
    if len(emotions) < 2:
        return None
    t = text.lower()

    # 1. Percentages attached to specific emotion names: "70% happy".
    #    Look for "<num>% <emotion>" or "<emotion> <num>%".
    per_emotion: dict[str, float] = {}
    for emo in emotions:
        # number then emotion:  "70% happy" / "70 percent happy"
        m = re.search(rf"(\d{{1,3}})\s*(?:%|percent|प्रतिशत)?\s*{re.escape(emo)}", t)
        if not m:
            # emotion then number: "happy 70%"
            m = re.search(rf"{re.escape(emo)}\s*(\d{{1,3}})\s*(?:%|percent|प्रतिशत)", t)
        if m:
            per_emotion[emo] = float(m.group(1))
    if len(per_emotion) >= 2:
        vals = np.array([per_emotion.get(e, 0.0) for e in emotions], dtype=np.float64)
        if vals.sum() > 0:
            return list(vals / vals.sum())

    # 2. Bare ratio "70-30" / "70:30" / "70/30" → apply in emotion order.
    m = re.search(r"\b(\d{1,3})\s*[-:/]\s*(\d{1,3})\b", t)
    if m and len(emotions) == 2:
        a, b = float(m.group(1)), float(m.group(2))
        if a + b > 0:
            return [a / (a + b), b / (a + b)]

    # 3. Fuzzy quantifiers near an emotion name.
    fuzzy: dict[str, float] = {}
    for emo in emotions:
        for word, share in _FUZZY_QUANTIFIERS.items():
            # quantifier within a short window before the emotion
            if re.search(rf"{re.escape(word)}\b[\w\s]{{0,15}}{re.escape(emo)}", t):
                fuzzy[emo] = share
                break
    if len(fuzzy) == 1 and len(emotions) == 2:
        # One emotion quantified → the other takes the remainder.
        e_known = next(iter(fuzzy))
        other = [e for e in emotions if e != e_known][0]
        share = fuzzy[e_known]
        w = {e_known: share, other: 1.0 - share}
        return [w[e] for e in emotions]
    if len(fuzzy) >= 2:
        vals = np.array([fuzzy.get(e, 0.5) for e in emotions], dtype=np.float64)
        # If the quantifiers are essentially equal (e.g. "some X and some Y"),
        # that means "balanced" — not an explicit skew — so defer to inferred
        # weights rather than forcing an artificial ratio.
        if float(vals.max() - vals.min()) < 1e-6:
            return None
        return list(vals / vals.sum())

    return None


def parse_intent(
    distribution: np.ndarray,
    raw_text: str = "",
    mode_override: Optional[str] = None,
    max_emotions: int = 2,
) -> MatchIntent:
    """Parse a request into a MatchIntent.

    Args:
        distribution: (15,) fused emotion distribution from Phase 1C.
        raw_text: the user's original request (EN/HI free text or transcript).
            Empty is allowed (then mode falls back to single/blend by count).
        mode_override: force "single" | "blend" | "mix", skipping detection.
        max_emotions: cap on emotions used (default 2).

    Returns:
        MatchIntent with mode, emotions, weights, and a reason.

    Raises:
        ValueError: on a bad distribution or an invalid mode_override.
    """
    emotions, weights = _top_emotions(distribution, max_emotions=max_emotions)

    if mode_override is not None:
        if mode_override not in VALID_MODES:
            raise ValueError(
                f"mode_override must be one of {VALID_MODES}, got {mode_override!r}."
            )
        # If forced to single but multiple emotions were found, keep only top.
        if mode_override == MODE_SINGLE:
            emotions, weights = emotions[:1], [1.0]
        return MatchIntent(
            mode=mode_override, emotions=emotions, weights=weights,
            detected_from="override", raw_text=raw_text,
        )

    mode, reason = _detect_mode(raw_text or "", len(emotions))
    if mode == MODE_SINGLE:
        emotions, weights = emotions[:1], [1.0]
    else:
        # User-stated proportions override model-inferred weights.
        explicit = _extract_explicit_ratios(raw_text or "", emotions)
        if explicit is not None:
            weights = explicit
            reason += " | explicit user ratio applied"

    return MatchIntent(
        mode=mode, emotions=emotions, weights=weights,
        detected_from=reason, raw_text=raw_text,
    )


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Intent parser self-test")
    print("-" * 55)

    def dist(**kw):
        d = np.zeros(NUM_EMOTIONS)
        for name, p in kw.items():
            d[AETHER_EMOTIONS.index(name)] = p
        return d

    # 1. Single dominant emotion → single (regardless of text).
    r = parse_intent(dist(happy=0.9, calm=0.05), "I feel great")
    assert r.mode == "single" and r.emotions == ["happy"], r.as_dict()
    print(f"  single dominant           → {r.mode} {r.emotions}")

    # 2. Two emotions + "but" → blend.
    r = parse_intent(dist(nostalgic=0.5, hopeful=0.4), "nostalgic but hopeful")
    assert r.mode == "blend" and set(r.emotions) == {"nostalgic", "hopeful"}, r.as_dict()
    print(f"  'nostalgic but hopeful'   → {r.mode} {r.emotions}  ({r.detected_from})")

    # 3. Two emotions + "some…and some" → mix.
    r = parse_intent(dist(nostalgic=0.5, hopeful=0.45),
                     "some nostalgic and some hopeful songs")
    assert r.mode == "mix", r.as_dict()
    print(f"  'some X and some Y'       → {r.mode} {r.emotions}  ({r.detected_from})")

    # 4. Hindi blend: "लेकिन".
    r = parse_intent(dist(sad=0.5, hopeful=0.4), "उदास लेकिन उम्मीद भरे गाने")
    assert r.mode == "blend", r.as_dict()
    print(f"  Hindi 'लेकिन' (but)        → {r.mode}  ({r.detected_from})")

    # 5. Hindi mix: "कुछ … और … कुछ".
    r = parse_intent(dist(happy=0.5, calm=0.42),
                     "कुछ खुशी वाले और कुछ शांत गाने")
    assert r.mode == "mix", r.as_dict()
    print(f"  Hindi 'कुछ…और…कुछ' (mix)   → {r.mode}  ({r.detected_from})")

    # 6. Romanized Hindi mix.
    r = parse_intent(dist(happy=0.5, sad=0.42), "kuch happy aur kuch sad")
    assert r.mode == "mix", r.as_dict()
    print(f"  Romanized 'kuch…aur…kuch' → {r.mode}  ({r.detected_from})")

    # 7. Plain "and" defaults to mix.
    r = parse_intent(dist(happy=0.5, energetic=0.42), "happy and energetic")
    assert r.mode == "mix", r.as_dict()
    print(f"  'happy and energetic'     → {r.mode}  ({r.detected_from})")

    # 8. Two emotions, no connective → blend default.
    r = parse_intent(dist(dreamy=0.5, calm=0.42), "dreamy calm vibes")
    assert r.mode == "blend", r.as_dict()
    print(f"  'dreamy calm vibes'       → {r.mode}  ({r.detected_from})")

    # 9. Override wins.
    r = parse_intent(dist(nostalgic=0.5, hopeful=0.45),
                     "nostalgic but hopeful", mode_override="mix")
    assert r.mode == "mix" and r.detected_from == "override"
    print(f"  override=mix              → {r.mode}  ({r.detected_from})")

    # 10. No-signal distribution → calm default.
    r = parse_intent(np.zeros(NUM_EMOTIONS), "")
    assert r.emotions == ["calm"]
    print(f"  empty distribution        → {r.mode} {r.emotions}")

    # 11. Weights normalize.
    r = parse_intent(dist(nostalgic=0.6, hopeful=0.3), "nostalgic but hopeful")
    assert abs(sum(r.weights) - 1.0) < 1e-6
    print(f"  weights sum to 1          → {[round(w,3) for w in r.weights]}")

    # 12. Explicit percentages override inferred weights.
    r = parse_intent(dist(happy=0.5, sad=0.5),
                     "70% happy and 30% sad songs")
    # emotions come out ordered by distribution; find happy's weight
    wmap = dict(zip(r.emotions, r.weights))
    assert abs(wmap["happy"] - 0.7) < 0.02 and abs(wmap["sad"] - 0.3) < 0.02, wmap
    print(f"  '70% happy 30% sad'       → {[(e,round(w,2)) for e,w in wmap.items()]}")

    # 13. Bare ratio "70-30" applied in emotion order.
    r = parse_intent(dist(happy=0.55, sad=0.45), "happy and sad 70-30")
    assert abs(r.weights[0] - 0.7) < 0.02, r.weights
    print(f"  bare ratio '70-30'        → {[round(w,2) for w in r.weights]}")

    # 14. Fuzzy quantifier: "mostly happy, a bit sad".
    r = parse_intent(dist(happy=0.5, sad=0.5), "mostly happy and a bit of sad")
    wmap = dict(zip(r.emotions, r.weights))
    assert wmap["happy"] > wmap["sad"], wmap
    print(f"  'mostly happy, a bit sad' → {[(e,round(w,2)) for e,w in wmap.items()]}")

    # 15. No explicit ratio → falls back to model-inferred weights.
    r = parse_intent(dist(happy=0.6, sad=0.3), "some happy and some sad")
    assert abs(sum(r.weights) - 1.0) < 1e-6
    assert "explicit" not in r.detected_from
    print(f"  no ratio → inferred        → {[round(w,2) for w in r.weights]}")

    print("-" * 55)
    print("✅ All intent-parser self-tests passed.")


if __name__ == "__main__":
    _selftest()
