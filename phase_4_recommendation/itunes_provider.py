"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 4 · iTunes Provider  (global delivery + freshness)
═══════════════════════════════════════════════════════════════════
Default concrete `MusicProvider` for regions where Deezer is unavailable
(India/China/Russia). Apple's iTunes Search API is free, needs NO auth, and
works from 66 storefronts worldwide — so the live demo works for everyone,
including Indian interviewers.

Same three-job contract as any provider (the seam pays off — the recommender,
models, sequencer, and provider ABC are untouched):

  • enrich(tracks)          — resolve the store's cosine picks to playable data
                              (30s preview, artwork, store link) via iTunes.
  • discover(emotion, limit) — FRESHNESS layer: mood query → current Apple
                              catalog tracks (how 2026 songs enter the mix).
  • export_playlist(...)     — Apple has no public playlist-write API, so this
                              raises; use exporter.py (.m3u8/.json) instead —
                              which works for every user regardless.

Why swap from Deezer
--------------------
Deezer geo-blocks India: its API returns `total>0` but an empty `data` array
to Indian IPs. iTunes Search has no such restriction. Because Aether isolates
the music service behind `MusicProvider`, switching is a one-file change with
zero impact on the recommendation brain.

Notes
-----
• Endpoint: https://itunes.apple.com/search  (media=music, entity=song)
• No audio features exposed (like Deezer) — matching stays on the local store.
• Rate limit ~20 req/min per IP (HTTP 429 if exceeded); failures degrade to
  empty/no-op so a hiccup never breaks a playlist.
• HTTP is injected for offline testing (default: stdlib urllib).
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from models import Track
from provider import MusicProvider

_API_BASE = "https://itunes.apple.com/search"


# ──────────────────────────────────────────────
# Emotion → iTunes mood query (freshness layer)
# Keywords follow the music-style hints in config.AETHER_EMOTIONS.
# ──────────────────────────────────────────────
EMOTION_MOOD_QUERIES: dict[str, str] = {
    "happy":       "feel good upbeat pop",
    "sad":         "sad slow ballad",
    "angry":       "aggressive heavy rock",
    "calm":        "calm ambient chill",
    "anxious":     "tense dark cinematic",
    "energetic":   "edm dance workout",
    "focused":     "lofi study instrumental",
    "nostalgic":   "throwback retro hits",
    "romantic":    "romantic love songs",
    "melancholic": "melancholic indie",
    "confident":   "empowering hype anthem",
    "hopeful":     "uplifting inspiring",
    "frustrated":  "punk hard rock",
    "lonely":      "lonely sad acoustic",
    "dreamy":      "dream pop ethereal",
}


def _default_http_get(url: str, timeout: float = 10.0) -> dict[str, Any]:
    """GET `url` and parse JSON (stdlib only — no `requests` dependency)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Aether/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class ITunesProvider(MusicProvider):
    """Global, free, no-auth delivery + freshness via Apple's iTunes Search API."""

    name = "itunes"

    def __init__(
        self,
        http_get: Optional[Callable[[str], dict]] = None,
        timeout: float = 10.0,
        country: str = "US",
    ):
        """
        Args:
            http_get: injectable HTTP callable (default: urllib). Tests pass a
                fake returning canned iTunes JSON.
            timeout: request timeout (seconds) for the default callable.
            country: iTunes storefront (e.g. "US" for the broadest catalog,
                "IN" for the India storefront). The API itself is reachable
                from anywhere; this only selects which catalog to search.
        """
        self._http_get = http_get or (lambda u: _default_http_get(u, timeout))
        self.timeout = timeout
        self.country = country

    # ── low-level search ──
    def _search(self, term: str, limit: int) -> list[dict]:
        """Run an iTunes song search; return the raw result dicts (best first)."""
        params = urllib.parse.urlencode({
            "term": term, "media": "music", "entity": "song",
            "limit": max(1, limit), "country": self.country,
        })
        url = f"{_API_BASE}?{params}"
        try:
            data = self._http_get(url)
        except Exception:
            return []
        return data.get("results", []) if isinstance(data, dict) else []

    # ── raw iTunes result → Phase 4 Track ──
    @staticmethod
    def _raw_to_track(raw: dict, source_emotion: Optional[str] = None) -> Track:
        """Map an iTunes song result into a playable Phase 4 Track."""
        # Upscale the 100px artwork to 600px (simple, documented iTunes trick).
        art = raw.get("artworkUrl100")
        if art:
            art = art.replace("100x100", "600x600")
        return Track(
            title=raw.get("trackName", ""),
            artist=raw.get("artistName", "Unknown"),
            track_id=f"itunes:{raw.get('trackId')}",
            year=_year_from_date(raw.get("releaseDate")),
            energy=None,          # Apple doesn't expose audio features …
            valence=None,         # … (matching stays on the local store)
            tempo=None,
            match_score=None,
            source_emotion=source_emotion,
            provider_ref={
                "source": "itunes",
                "itunes_id": raw.get("trackId"),
                "preview_url": raw.get("previewUrl"),
                "link": raw.get("trackViewUrl"),
                "cover": art,
                "album": raw.get("collectionName"),
                "genre": raw.get("primaryGenreName"),
            },
        )

    # ── MusicProvider API ──
    def enrich(self, tracks: list[Track]) -> list[Track]:
        """Resolve each store pick to playable iTunes data (in place, order kept)."""
        for t in tracks:
            if t.provider_ref:            # fresh picks already resolved
                continue
            results = self._search(f"{t.title} {t.artist}", 1)
            if results:
                t.provider_ref = self._raw_to_track(results[0]).provider_ref
        return tracks

    def discover(self, emotion: str, limit: int) -> list[Track]:
        """Fetch fresh current-catalog tracks for an emotion (freshness layer)."""
        if limit <= 0:
            return []
        term = EMOTION_MOOD_QUERIES.get(emotion, emotion)
        raws = self._search(term, limit)
        tracks = [self._raw_to_track(r, source_emotion=emotion) for r in raws]
        return tracks[:limit]

    def export_playlist(self, tracks: list[Track], name: str) -> dict:
        """
        Apple has no public playlist-write API, so streaming-account export
        isn't available here. Use exporter.export() to write .m3u8 + .json,
        which works for every user regardless of provider.
        """
        raise RuntimeError(
            "iTunes has no public playlist-write API. Use exporter.export() "
            "for .m3u8/.json export instead."
        )


def _year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extract a 4-digit year from an iTunes releaseDate ('YYYY-MM-DDT...')."""
    if not date_str or not isinstance(date_str, str):
        return None
    head = date_str[:4]
    return int(head) if head.isdigit() else None


# ─────────────────────────────────────────────────────────────
# Self-test — mocked iTunes HTTP (no network)
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("iTunes provider self-test")
    print("-" * 55)

    def fake_song(i, title, artist):
        return {
            "trackId": 2000 + i, "trackName": title, "artistName": artist,
            "collectionName": f"Album {i}", "primaryGenreName": "Pop",
            "previewUrl": f"https://audio-preview.itunes.apple.com/{2000+i}.m4a",
            "artworkUrl100": f"https://is1.mzstatic.com/image/{i}/100x100bb.jpg",
            "trackViewUrl": f"https://music.apple.com/song/{2000+i}",
            "releaseDate": "2026-02-10T12:00:00Z",
        }

    calls: list[str] = []

    def fake_get(url: str) -> dict:
        calls.append(url)
        if "limit=1" in url:
            return {"resultCount": 1, "results": [fake_song(0, "Resolved", "Res Artist")]}
        return {"resultCount": 3,
                "results": [fake_song(i, f"Fresh {i}", f"Artist {i}") for i in range(3)]}

    prov = ITunesProvider(http_get=fake_get)

    # 1. discover() maps emotion → mood query and returns fresh, playable Tracks.
    fresh = prov.discover("dreamy", 3)
    assert len(fresh) == 3, len(fresh)
    assert all(t.track_id.startswith("itunes:") for t in fresh)
    assert all(t.provider_ref.get("preview_url", "").endswith(".m4a") for t in fresh)
    assert all(t.energy is None for t in fresh)
    assert fresh[0].year == 2026 and fresh[0].source_emotion == "dreamy"
    assert "600x600" in fresh[0].provider_ref["cover"], "artwork not upscaled"
    assert "dream+pop" in calls[-1] or "dream%20pop" in calls[-1], calls[-1]
    print(f"  discover('dreamy',3) → {[t.title for t in fresh]}")

    # 2. enrich() resolves a store pick and preserves its store features.
    store_pick = Track(title="Bright", artist="B", track_id="h2", energy=0.7)
    prov.enrich([store_pick])
    assert store_pick.provider_ref.get("preview_url"), "enrich didn't attach preview"
    assert store_pick.energy == 0.7, "enrich must not overwrite store features"
    assert store_pick.provider_ref["source"] == "itunes"
    print(f"  enrich(store pick) → preview attached, energy preserved ({store_pick.energy})")

    # 3. enrich() skips already-resolved fresh picks.
    before = len(calls)
    prov.enrich(fresh)
    assert len(calls) == before
    print("  enrich() skips already-resolved fresh picks ✓")

    # 4. export raises (no Apple write API) — points to exporter.
    try:
        prov.export_playlist(fresh, "X")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    print("  export_playlist() → correctly unsupported (use exporter.py) ✓")

    # 5. All 15 Aether emotions have a mood query.
    from config import AETHER_EMOTIONS
    missing = [e for e in AETHER_EMOTIONS if e not in EMOTION_MOOD_QUERIES]
    assert not missing, f"missing mood queries: {missing}"
    print(f"  mood-query coverage → all {len(AETHER_EMOTIONS)} emotions ✓")

    print("-" * 55)
    print("✅ All iTunes-provider self-tests passed.")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    _selftest()
