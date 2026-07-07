"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 4 · Playlist Sequencer
═══════════════════════════════════════════════════════════════════
Phase 3 returns tracks ranked by *relevance* (match score). A good playlist
isn't the most-relevant tracks dumped in score order — it has FLOW: an energy
arc, plus variety so one artist doesn't play back-to-back.

This module turns a relevance-ranked list into a *listening arc*:

  • "arc"        low → peak (near the middle) → resolve   (default; curated feel)
  • "ascending"  steady build   (hopeful / energetic / confident)
  • "descending" wind-down      (sad / calm / melancholic / lonely)
  • "steady"     preserve relevance order (minimal reshaping)

Shaping uses each track's normalized ``energy`` (0–1). If energies are missing
(any ``None``), sequencing degrades gracefully to the incoming relevance order
— it never drops or duplicates tracks.

Design note (interview-defensible): sequencing is a *pure function* of the
input list — no global state, no side effects — so it's trivially testable and
reusable by the Phase 7 crossfade layer later.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Optional

from models import Track

# Emotions whose musical intent is a wind-down rather than a build-up.
_DESCENDING_EMOTIONS = {"sad", "calm", "melancholic", "lonely"}
# Emotions whose intent is a build / lift.
_ASCENDING_EMOTIONS = {"hopeful", "energetic", "confident"}

_VALID_SHAPES = {"arc", "ascending", "descending", "steady"}


def default_arc_for(dominant_emotion: str, intensity_level: int) -> str:
    """
    Pick a sensible default arc shape from the dominant emotion + intensity.

      • Neutral intensity (0) → 'steady'  (don't over-shape a flat mood).
      • Wind-down emotions     → 'descending'.
      • Build emotions         → 'ascending'.
      • Everything else        → 'arc'.
    """
    if intensity_level <= 0:
        return "steady"
    if dominant_emotion in _DESCENDING_EMOTIONS:
        return "descending"
    if dominant_emotion in _ASCENDING_EMOTIONS:
        return "ascending"
    return "arc"


def _has_all_energies(tracks: list[Track]) -> bool:
    """True only if every track carries a usable energy value."""
    return all(t.energy is not None for t in tracks)


def _order_by_energy_shape(tracks: list[Track], shape: str) -> list[Track]:
    """
    Reorder tracks by the requested energy shape. Assumes all energies present
    (caller guarantees this). Returns a new list; does not mutate the input.
    """
    if shape == "steady":
        return list(tracks)

    ascending = sorted(tracks, key=lambda t: t.energy)  # type: ignore[arg-type]

    if shape == "ascending":
        return ascending
    if shape == "descending":
        return list(reversed(ascending))

    # shape == "arc": build a clean unimodal "mountain" — low → peak → low.
    # Even indices of the ascending list form the rising slope; odd indices
    # (reversed) form the falling slope. Peak lands near the middle.
    #   e.g. ascending [1,2,3,4,5] → rising [1,3,5] + falling [4,2] = [1,3,5,4,2]
    rising = ascending[0::2]
    falling = ascending[1::2][::-1]
    return rising + falling


def _space_artists(tracks: list[Track]) -> list[Track]:
    """
    Greedy single pass to avoid the same artist playing back-to-back, while
    preserving the overall energy order as much as possible. If a conflict
    can't be resolved (every remaining track is the same artist), the original
    adjacency is kept rather than dropping a track.
    """
    result = list(tracks)
    for i in range(1, len(result)):
        if result[i].artist and result[i].artist == result[i - 1].artist:
            for j in range(i + 1, len(result)):
                if result[j].artist != result[i - 1].artist:
                    result[i], result[j] = result[j], result[i]
                    break
    return result


def sequence(
    tracks: list[Track],
    shape: str = "arc",
    space_artists: bool = True,
) -> list[Track]:
    """
    Turn a relevance-ranked track list into a flowing playlist.

    Args:
        tracks: relevance-ranked tracks (already deduped by Phase 3).
        shape:  one of {"arc", "ascending", "descending", "steady"}.
        space_artists: if True, avoid consecutive same-artist tracks.

    Returns:
        A new list of the same tracks, reordered, with ``rank`` set (1-based).

    Raises:
        ValueError: if `shape` is not recognized.
    """
    if shape not in _VALID_SHAPES:
        raise ValueError(
            f"Unknown arc shape {shape!r}; expected one of {sorted(_VALID_SHAPES)}"
        )

    if not tracks:
        return []

    if shape != "steady" and _has_all_energies(tracks):
        ordered = _order_by_energy_shape(tracks, shape)
    else:
        ordered = list(tracks)

    if space_artists:
        ordered = _space_artists(ordered)

    for idx, track in enumerate(ordered, start=1):
        track.rank = idx

    return ordered


# ──────────────────────────────────────────────
# Self-tests
# ──────────────────────────────────────────────
def _mk(title: str, artist: str, energy: Optional[float]) -> Track:
    return Track(title=title, artist=artist, energy=energy)


def _run_self_tests() -> None:
    ts = [_mk(f"s{i}", f"a{i}", e)
          for i, e in enumerate([0.1, 0.9, 0.5, 0.7, 0.3, 0.8, 0.2])]

    # 1. Arc peaks in the interior and starts low.
    seq = sequence(ts, shape="arc", space_artists=False)
    energies = [t.energy for t in seq]
    peak_idx = energies.index(max(energies))
    assert 0 < peak_idx < len(energies) - 1, f"arc peak at edge: {energies}"
    assert energies[0] == min(energies), f"arc should start low: {energies}"

    # 2. Ascending / descending are monotonic.
    asc = [t.energy for t in sequence(ts, shape="ascending", space_artists=False)]
    assert asc == sorted(asc), f"ascending not sorted: {asc}"
    desc = [t.energy for t in sequence(ts, shape="descending", space_artists=False)]
    assert desc == sorted(desc, reverse=True), f"descending not reversed: {desc}"

    # 3. No track lost or duplicated by any shape.
    for shape in ("arc", "ascending", "descending", "steady"):
        out = sequence(ts, shape=shape, space_artists=True)
        assert len(out) == len(ts), f"{shape}: length changed"
        assert {t.title for t in out} == {t.title for t in ts}, f"{shape}: set changed"

    # 4. Artist spacing removes back-to-back duplicates when possible.
    dup = [_mk("x", "SAME", 0.5), _mk("y", "SAME", 0.4), _mk("z", "OTHER", 0.3)]
    spaced = sequence(dup, shape="steady", space_artists=True)
    artists = [t.artist for t in spaced]
    assert not any(artists[i] == artists[i - 1] for i in range(1, len(artists))), \
        f"artist spacing failed: {artists}"

    # 5. Missing energies → graceful fallback to input order.
    no_energy = [_mk("p", "a", None), _mk("q", "b", None)]
    out = sequence(no_energy, shape="arc")
    assert [t.title for t in out] == ["p", "q"], "should preserve order when energies missing"

    # 6. Ranks are 1-based and contiguous.
    assert [t.rank for t in out] == [1, 2], f"bad ranks: {[t.rank for t in out]}"

    # 7. default_arc_for picks sensible shapes.
    assert default_arc_for("sad", 3) == "descending"
    assert default_arc_for("hopeful", 2) == "ascending"
    assert default_arc_for("nostalgic", 2) == "arc"
    assert default_arc_for("happy", 0) == "steady"

    # 8. Unknown shape raises.
    try:
        sequence(ts, shape="zigzag")
        raise AssertionError("expected ValueError for bad shape")
    except ValueError:
        pass

    print("✅ All sequencer self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
