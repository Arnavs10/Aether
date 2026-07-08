"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 4 · Cross-Platform Links
═══════════════════════════════════════════════════════════════════
Give every recommended track a set of "open the full song on…" links, so the
UI (Phase 7/8) can render Apple Music / Spotify / YouTube buttons next to the
inline 30-second preview.

Design (honest + universal):
  • Free APIs (iTunes/Deezer) only return 30s previews — full playback of a
    complete track requires the listener's OWN paid account + that platform's
    SDK. So the universal, works-for-everyone pattern is:
        inline 30s preview  +  "Open full song on <platform>" deep-links out.
  • When the source provider gives an exact track URL (iTunes does), we use it.
    Otherwise we build a search deep-link that lands the user on the song.
  • No auth, no network — pure URL construction. Fully testable offline.

This module only produces the link DATA. Rendering the buttons, the audio
player, and any Premium-only in-app playback are Phase 7/8 (frontend).
Links are stored on ``track.extra["links"]`` so models.py stays untouched.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import urllib.parse
from typing import Optional

from models import Recommendation, Track


def _q(text: str) -> str:
    """URL-encode a search term (spaces → '+')."""
    return urllib.parse.quote_plus(text.strip())


def _search_query(track: Track) -> str:
    """Human search string for a track: 'Title Artist'."""
    return f"{track.title} {track.artist}".strip()


def build_links(track: Track) -> dict[str, Optional[str]]:
    """
    Build cross-platform links for one track.

    Returns a dict with:
        preview_url  — the 30s preview (universal inline playback), or None
        apple_music  — exact track URL if the source is iTunes, else a search link
        spotify      — Spotify search deep-link to the song
        youtube      — YouTube search deep-link to the song
        source_link  — the native link from whichever provider resolved it

    All links are anonymous/no-auth and open the full song on that platform.
    """
    ref = track.provider_ref or {}
    query = _search_query(track)

    # Apple Music: iTunes provider already returns the canonical track URL.
    if ref.get("source") == "itunes" and ref.get("link"):
        apple = ref["link"]
    else:
        apple = f"https://music.apple.com/search?term={_q(query)}"

    return {
        "preview_url": ref.get("preview_url"),
        "apple_music": apple,
        "spotify": f"https://open.spotify.com/search/{_q(query)}",
        "youtube": f"https://www.youtube.com/results?search_query={_q(query)}",
        "source_link": ref.get("link"),
    }


def attach_links(recommendation: Recommendation) -> Recommendation:
    """
    Populate ``track.extra['links']`` for every track in a Recommendation.

    Call this after recommend() (and after the provider's enrich()) so the
    links reflect resolved preview/source URLs. Returns the same object for
    chaining. Safe to call more than once (idempotent).
    """
    for track in recommendation.tracks:
        track.extra["links"] = build_links(track)
    return recommendation


# ─────────────────────────────────────────────────────────────
# Self-test — no network, no external state
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Links helper self-test")
    print("-" * 55)

    # 1. iTunes-sourced track → exact Apple link, search links elsewhere.
    itunes_track = Track(
        title="Feel Good", artist="Gryffin & ILLENIUM", track_id="itunes:1",
        provider_ref={"source": "itunes", "preview_url": "https://x/p.m4a",
                      "link": "https://music.apple.com/song/1"},
    )
    L = build_links(itunes_track)
    assert L["apple_music"] == "https://music.apple.com/song/1", L["apple_music"]
    assert L["preview_url"] == "https://x/p.m4a"
    assert L["spotify"].startswith("https://open.spotify.com/search/")
    assert "Feel+Good" in L["spotify"] and "ILLENIUM" in L["spotify"]
    assert L["youtube"].startswith("https://www.youtube.com/results?search_query=")
    print(f"  iTunes track → apple(exact) + spotify/youtube(search) ✓")

    # 2. Store track with no provider_ref → all search links, preview None.
    store_track = Track(title="Bright", artist="B", track_id="h2")
    L2 = build_links(store_track)
    assert L2["preview_url"] is None
    assert L2["apple_music"].startswith("https://music.apple.com/search?term=")
    assert "Bright+B" in L2["apple_music"]
    print("  store track (unenriched) → all search links, no preview ✓")

    # 3. Special characters are safely URL-encoded.
    weird = Track(title="Café / Déjà", artist="A&B", track_id="x")
    L3 = build_links(weird)
    assert " " not in L3["spotify"] and "&" not in L3["spotify"].split("search/")[1]
    print("  special chars → URL-encoded safely ✓")

    # 4. attach_links populates every track's extra['links'] and is idempotent.
    rec = Recommendation(
        tracks=[itunes_track, store_track], request_text="x", intent_mode="single",
        intensity_level=2, intensity_label="moderate",
        dominant_emotions=[("happy", 1.0)], arc_shape="arc", reason="r",
    )
    attach_links(rec)
    assert all("links" in t.extra for t in rec.tracks)
    assert rec.tracks[0].extra["links"]["apple_music"] == "https://music.apple.com/song/1"
    attach_links(rec)  # idempotent — no crash, still valid
    assert set(rec.tracks[0].extra["links"]) == {
        "preview_url", "apple_music", "spotify", "youtube", "source_link"}
    print("  attach_links() → every track linked, idempotent ✓")

    # 5. It also flows into as_dict() (API/UI ready).
    d = rec.as_dict()
    assert "links" in d["tracks"][0]["extra"], d["tracks"][0]["extra"].keys()
    print("  links surface in as_dict() for the API/UI ✓")

    print("-" * 55)
    print("✅ All links-helper self-tests passed.")


if __name__ == "__main__":
    _selftest()
