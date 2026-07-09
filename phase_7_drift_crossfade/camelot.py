"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 7 · Camelot Wheel (harmonic key compatibility)
═══════════════════════════════════════════════════════════════════
DJs don't pick the next track by tempo alone — they mix in *compatible keys* so
the blend doesn't clash. The Camelot wheel is the standard notation for this:
every key becomes a code like ``8B`` (C major) or ``8A`` (A minor), and two
tracks mix harmonically when their codes are the same, one step apart on the
wheel (same letter), or relative major/minor (same number, other letter).

Aether's source data gives each song Spotify's ``key`` (pitch class 0–11, where
0=C … 11=B) and ``mode`` (1=major, 0=minor). This module converts that to a
Camelot code and scores how well two codes mix — the harmonic half of Phase 7's
transition selection.

Pure, deterministic, no dependencies. Interview-defensible: the wheel is just
the circle of fifths (each +1 on the number = up a perfect fifth), so
compatibility = adjacency on that circle plus the relative major/minor pair.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Optional

# ── Pitch class (0=C … 11=B) → Camelot NUMBER, per mode ──
# Major keys use letter "B", minor keys use letter "A". Numbers follow the
# circle of fifths: C major = 8B, and every perfect fifth up adds 1 (wrapping
# 12→1). A minor = 8A is C major's relative minor, and so on.
_MAJOR_NUM = {0: 8, 1: 3, 2: 10, 3: 5, 4: 12, 5: 7,
              6: 2, 7: 9, 8: 4, 9: 11, 10: 6, 11: 1}
_MINOR_NUM = {0: 5, 1: 12, 2: 7, 3: 2, 4: 9, 5: 4,
              6: 11, 7: 6, 8: 1, 9: 8, 10: 3, 11: 10}


def to_camelot(key: int, mode: int) -> Optional[str]:
    """
    Convert a Spotify (key, mode) pair to a Camelot code (e.g. "8B").

    Args:
        key: pitch class 0–11 (Spotify's ``key``); -1 means "unknown".
        mode: 1 for major, 0 for minor.

    Returns:
        Camelot code string, or None if the key is unknown/out of range.
    """
    try:
        k = int(key)
        m = int(mode)
    except (TypeError, ValueError):
        return None
    if not (0 <= k <= 11) or m not in (0, 1):
        return None
    if m == 1:
        return f"{_MAJOR_NUM[k]}B"
    return f"{_MINOR_NUM[k]}A"


def _parse(code: str) -> Optional[tuple[int, str]]:
    """Split "8B" → (8, "B"); returns None if malformed."""
    if not code or len(code) < 2:
        return None
    num_part, letter = code[:-1], code[-1].upper()
    if letter not in ("A", "B"):
        return None
    try:
        num = int(num_part)
    except ValueError:
        return None
    if not (1 <= num <= 12):
        return None
    return num, letter


def _wraps(a: int, b: int) -> bool:
    """True if a and b are adjacent on the 1–12 wheel (12 wraps to 1)."""
    return (a - b) % 12 == 1 or (b - a) % 12 == 1


def compatibility_score(code_a: Optional[str], code_b: Optional[str]) -> float:
    """
    Score how harmonically compatible two Camelot codes are, in [0, 1].

      1.00  same key (perfect blend)
      0.85  adjacent number, same letter  (±1 on the wheel — subtle energy shift)
      0.80  same number, opposite letter  (relative major/minor)
      0.50  adjacent number, opposite letter (diagonal — usable "mood shift")
      0.00  otherwise (clashing keys) — or if either key is unknown

    Unknown keys score 0.0 rather than blocking, so the selector can still fall
    back on tempo/emotion when key data is missing.
    """
    pa, pb = _parse(code_a or ""), _parse(code_b or "")
    if pa is None or pb is None:
        return 0.0
    (na, la), (nb, lb) = pa, pb
    if na == nb and la == lb:
        return 1.0
    if la == lb and _wraps(na, nb):
        return 0.85
    if na == nb and la != lb:
        return 0.80
    if la != lb and _wraps(na, nb):
        return 0.50
    return 0.0


def compatible(code_a: Optional[str], code_b: Optional[str],
               min_score: float = 0.80) -> bool:
    """True if the two codes mix at or above `min_score` (default: harmonic)."""
    return compatibility_score(code_a, code_b) >= min_score


def compatible_codes(code: str) -> set[str]:
    """The set of codes that mix harmonically (score ≥ 0.80) with `code`."""
    p = _parse(code or "")
    if p is None:
        return set()
    num, letter = p
    other = "A" if letter == "B" else "B"
    up = (num % 12) + 1
    down = (num - 2) % 12 + 1
    return {f"{num}{letter}", f"{up}{letter}", f"{down}{letter}", f"{num}{other}"}


# ─────────────────────────────────────────────────────────────
# Self-test — pure, offline
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Camelot wheel self-test")
    print("-" * 55)

    # 1. Known conversions (Spotify key/mode → Camelot).
    cases = {
        (0, 1): "8B",   # C major
        (9, 0): "8A",   # A minor  (relative of C major)
        (7, 1): "9B",   # G major
        (2, 1): "10B",  # D major
        (11, 1): "1B",  # B major (wrap)
        (8, 0): "1A",   # G#/Ab minor
        (5, 1): "7B",   # F major
        (2, 0): "7A",   # D minor
    }
    for (k, m), expect in cases.items():
        got = to_camelot(k, m)
        assert got == expect, f"key={k} mode={m}: got {got}, expected {expect}"
    print(f"  key/mode → Camelot: {len(cases)} conversions ✓")

    # 2. Unknown key degrades to None.
    assert to_camelot(-1, 1) is None and to_camelot(5, 9) is None
    print("  unknown key → None ✓")

    # 3. Compatibility tiers.
    assert compatibility_score("8B", "8B") == 1.00     # same
    assert compatibility_score("8B", "9B") == 0.85     # adjacent same letter
    assert compatibility_score("8B", "7B") == 0.85
    assert compatibility_score("8B", "8A") == 0.80     # relative maj/min
    assert compatibility_score("8B", "9A") == 0.50     # diagonal
    assert compatibility_score("8B", "2B") == 0.00     # clash
    assert compatibility_score("1B", "12B") == 0.85    # wheel wrap
    assert compatibility_score("8B", None) == 0.00     # unknown
    print("  compatibility tiers (same/adjacent/relative/diag/clash/wrap) ✓")

    # 4. compatible() threshold + compatible_codes set.
    assert compatible("8A", "8B") and not compatible("8A", "2B")
    assert compatible_codes("8B") == {"8B", "9B", "7B", "8A"}
    assert compatible_codes("1A") == {"1A", "2A", "12A", "1B"}   # wrap both ways
    print("  compatible() + compatible_codes() ✓")

    print("-" * 55)
    print("✅ All Camelot self-tests passed.")


if __name__ == "__main__":
    _selftest()
