"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 6 · Perceiver
═══════════════════════════════════════════════════════════════════
The agent's "perceive" step: read a free-text request and decide the emotional
JOURNEY it implies — a start emotion (where the listener is) and a target
emotion (where they want to go) — plus the desired playlist length.

Two paths behind one call:
  • rule  — offline lexicon + patterns ("from X to Y", "lift me up from X",
            "take me to Y", "just X music"). Zero deps, deterministic, always on.
  • llm   — optional: an injected llm_fn returns {start,target} JSON, validated
            and snapped to the 15 Aether emotions; falls back to rule on any
            issue.

When only a current mood is given for a NEGATIVE state, the target defaults via
the music-therapy iso principle (meet-then-guide): e.g. sad → hopeful,
anxious → calm. A single positive/neutral mood stays put (start == target).
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from config import AETHER_EMOTIONS, map_label_to_aether  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    AETHER_EMOTIONS = []
    def map_label_to_aether(x):  # type: ignore
        return None


# Colloquial mood words not in config's fine→core map.
_AGENT_LEXICON: dict[str, str] = {
    "down": "sad", "low": "sad", "blue": "sad", "heavy": "sad", "tearful": "sad",
    "stressed": "anxious", "tense": "anxious", "on edge": "anxious",
    "nervous": "anxious", "overwhelmed": "anxious", "worried": "anxious",
    "pumped": "energetic", "hyped": "energetic", "amped": "energetic",
    "chill": "calm", "relaxed": "calm", "mellow": "calm", "peaceful": "calm",
    "mad": "angry", "pissed": "angry", "furious": "angry",
    "in love": "romantic", "loved up": "romantic",
    "productive": "focused", "study": "focused", "studying": "focused",
    "work": "focused", "working": "focused", "locked in": "focused",
    "focus": "focused", "concentrate": "focused", "concentrating": "focused",
    "relax": "calm", "unwind": "calm", "calm down": "calm",
    "empty": "lonely", "alone": "lonely", "isolated": "lonely",
    "wistful": "nostalgic", "reminiscing": "nostalgic",
    "bold": "confident", "powerful": "confident", "unstoppable": "confident",
}

# Iso-principle regulation targets for negative start states.
_REGULATION_TARGET: dict[str, str] = {
    "sad": "hopeful", "melancholic": "hopeful", "lonely": "hopeful",
    "anxious": "calm", "frustrated": "calm", "angry": "calm",
}

# Words signalling the listener wants OUT of their current (negative) state.
_CHANGE_CUES = re.compile(
    r"\b(from|lift|out of|past|over|escape|shake|beat|better|help|move|turn|"
    r"switch|transition|tired of|sick of|done)\b",
    re.I,
)

# Transport phrasing ("take me to X") → X is the GOAL, start is unstated.
_TRANSPORT_TO = re.compile(r"\b(?:take|get|bring|move|carry)\s+me\s+to\b", re.I)

_LEN_RE = re.compile(r"\b(\d{1,3})\s*(?:songs?|tracks?)\b", re.I)


@dataclass
class Perceived:
    """The parsed request: an emotional journey + length."""
    start: str
    target: str
    length: int
    raw: str
    source: str = "rule"                      # "rule" | "llm"
    notes: list[str] = field(default_factory=list)

    @property
    def is_journey(self) -> bool:
        return self.start != self.target


# ──────────────────────────────────────────────
# Emotion spotting
# ──────────────────────────────────────────────
def _emotion_at(token: str) -> Optional[str]:
    """Map a single token/phrase to an Aether emotion, or None."""
    t = token.lower().strip()
    if t in AETHER_EMOTIONS:
        return t
    mapped = map_label_to_aether(t)
    if mapped:
        return mapped
    return _AGENT_LEXICON.get(t)


def _spot_emotions(text: str) -> list[tuple[int, str]]:
    """Find emotion mentions as (char_position, emotion), in order of appearance."""
    hits: list[tuple[int, str]] = []
    # Multi-word lexicon phrases first (e.g. "on edge", "in love").
    for phrase, emo in _AGENT_LEXICON.items():
        if " " in phrase:
            for m in re.finditer(re.escape(phrase), text, re.I):
                hits.append((m.start(), emo))
    # Single tokens.
    for m in re.finditer(r"[a-zA-Z]+", text):
        emo = _emotion_at(m.group(0))
        if emo:
            hits.append((m.start(), emo))
    # De-dup consecutive same-emotion at same rough spot; keep order.
    hits.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str]] = []
    for pos, emo in hits:
        if deduped and deduped[-1][1] == emo and pos - deduped[-1][0] < 3:
            continue
        deduped.append((pos, emo))
    return deduped


def regulation_target(start: str) -> str:
    """Iso-principle goal emotion for a start state (identity if not negative)."""
    return _REGULATION_TARGET.get(start, start)


# ──────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────
def perceive(
    text: str,
    llm_fn: Optional[Callable[[str], str]] = None,
    default_length: int = 12,
) -> Perceived:
    """
    Parse a request into a Perceived journey (start, target, length).

    Tries the LLM path if `llm_fn` is given; otherwise (or on failure) uses the
    deterministic rule path.
    """
    text = (text or "").strip()
    length = default_length
    m = _LEN_RE.search(text)
    if m:
        length = max(1, min(int(m.group(1)), 100))

    if llm_fn is not None:
        got = _perceive_llm(text, llm_fn, length)
        if got is not None:
            return got

    return _perceive_rule(text, length)


def _perceive_rule(text: str, length: int) -> Perceived:
    notes: list[str] = []

    # Explicit "from X to Y".
    fx = re.search(r"\bfrom\s+([a-z ]+?)\s+to\s+([a-z ]+)", text, re.I)
    if fx:
        a, b = _emotion_at(fx.group(1).split()[-1]), _emotion_at(fx.group(2).split()[0])
        # try the whole captured phrase too (handles multi-word lexicon)
        a = a or _first_emotion(fx.group(1))
        b = b or _first_emotion(fx.group(2))
        if a and b:
            return Perceived(a, b, length, text, "rule",
                             ["explicit 'from X to Y'"])

    spots = _spot_emotions(text)
    emotions = [e for _, e in spots]

    # "to Y" / "take me to Y" goal phrasing → target is the emotion after a cue.
    ty = re.search(r"\b(?:to|toward|towards|into|reach|become)\s+([a-z ]+)", text, re.I)
    target_cue = _first_emotion(ty.group(1)) if ty else None

    wants_change = bool(_CHANGE_CUES.search(text))

    if len(emotions) >= 2:
        # First mention = start, a distinct later mention = target.
        start = emotions[0]
        target = next((e for e in emotions[1:] if e != start), emotions[1])
        if target_cue and target_cue != start:
            target = target_cue
        return Perceived(start, target, length, text, "rule",
                         ["two moods → start then target"])

    if len(emotions) == 1:
        only = emotions[0]
        # Goal phrasing "take me to X" → X is the target, start is unstated.
        if _TRANSPORT_TO.search(text) and only != "calm":
            return Perceived("calm", only, length, text, "rule",
                             ["transport-to goal → neutral start"])
        if target_cue and target_cue != only:
            return Perceived(only, target_cue, length, text, "rule",
                             ["one mood + explicit goal"])
        # Negative mood + a wanting-out cue → iso-principle regulation journey.
        if wants_change and only in _REGULATION_TARGET:
            return Perceived(only, regulation_target(only), length, text, "rule",
                             ["iso-principle regulation target"])
        return Perceived(only, only, length, text, "rule", ["single mood"])

    # No emotion detected → safe neutral single mood.
    return Perceived("calm", "calm", length, text, "rule",
                     ["no emotion detected → default calm"])


def _first_emotion(fragment: str) -> Optional[str]:
    """First Aether emotion found in a text fragment."""
    for _, emo in _spot_emotions(fragment):
        return emo
    return None


def _perceive_llm(text: str, llm_fn: Callable[[str], str], length: int) -> Optional[Perceived]:
    """LLM perception → validated Perceived, or None to trigger rule fallback."""
    valid = ", ".join(AETHER_EMOTIONS)
    prompt = (
        "Read the listener's request and output ONLY compact JSON: "
        '{"start": <emotion>, "target": <emotion>}. Both values MUST be one of '
        f"these exact emotions: [{valid}]. 'start' is how they feel now; "
        "'target' is how they want to feel (equal to start if they just want "
        f"more of the same).\n\nRequest: {text!r}\nJSON:"
    )
    try:
        raw = llm_fn(prompt)
        blob = raw[raw.index("{"): raw.rindex("}") + 1]
        data = json.loads(blob)
        start, target = str(data["start"]).lower(), str(data["target"]).lower()
        if start in AETHER_EMOTIONS and target in AETHER_EMOTIONS:
            return Perceived(start, target, length, text, "llm", ["LLM-perceived"])
    except Exception:
        return None
    return None


# ─────────────────────────────────────────────────────────────
# Self-test — pure, offline (rule path)
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Perceiver self-test")
    print("-" * 55)

    cases = {
        "help me go from anxious to calm":        ("anxious", "calm", True),
        "lift me up out of feeling sad":          ("sad", "hopeful", True),   # iso
        "I just want to focus for a while":       ("focused", "focused", False),
        "take me to energetic":                   ("calm", "energetic", True),  # goal only
        "some happy music please":                ("happy", "happy", False),
        "I'm stressed and need to chill":         ("anxious", "calm", True),
        "give me 20 songs, from lonely to hopeful": ("lonely", "hopeful", True),
        "asdfghjkl nothing here":                 ("calm", "calm", False),
    }
    for text, (es, et, ej) in cases.items():
        p = perceive(text)
        assert p.start == es, (text, p.start, es)
        assert p.target == et, (text, p.target, et)
        assert p.is_journey == ej, (text, p.is_journey, ej)
        tag = f"{p.start}→{p.target}" + (" (journey)" if p.is_journey else "")
        print(f"  {text[:38]:38} → {tag}")

    # length parsing.
    assert perceive("from lonely to hopeful with 20 songs").length == 20
    print("  length parse ('20 songs') → 20 ✓")

    # LLM path used when it returns valid JSON; falls back otherwise.
    def good_llm(_p): return '{"start": "sad", "target": "confident"}'
    pl = perceive("random text", llm_fn=good_llm)
    assert (pl.start, pl.target, pl.source) == ("sad", "confident", "llm"), pl
    print(f"  llm path → {pl.start}→{pl.target} [{pl.source}] ✓")

    def bad_llm(_p): return "not json"
    pf = perceive("some happy music", llm_fn=bad_llm)
    assert pf.source == "rule" and pf.start == "happy", pf
    print("  llm garbage → rule fallback ✓")

    print("-" * 55)
    print("✅ All perceiver self-tests passed.")


if __name__ == "__main__":
    _selftest()
