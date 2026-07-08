"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 4 · Deezer Provider  (live delivery + freshness)
═══════════════════════════════════════════════════════════════════
The default concrete `MusicProvider`. Deezer is free, its search + chart
endpoints need no auth, and it exposes 30-second preview MP3s — so the live
demo works for ANY visitor (interviewers included), no account required.

Three jobs (the Hybrid-C delivery layer):

  • enrich(tracks)   — resolve the store's cosine picks to *playable* data
                       (preview URL, cover art, external link, Deezer id) by
                       searching Deezer for "title + artist".
  • discover(emotion, limit)
                     — the FRESHNESS layer: map an Aether emotion to a mood
                       query and pull current-catalog tracks. These are how
                       2026 songs enter the playlist (they carry no audio
                       features, so the recommender weaves them by position).
  • export_playlist(tracks, name)
                     — optional: create a playlist on the user's Deezer account
                       and add the tracks. Requires an OAuth access token.

Why the audio features aren't here
----------------------------------
Deezer exposes only `bpm` (and not even in list responses) — not
energy/valence/danceability. That's exactly why the *matching brain* stays on
the local feature store; Deezer handles delivery + freshness. Cosine-precise
scoring of fresh tracks is the Phase 5+ upgrade (librosa + a learned feature
model trained on the 1.2M store).

Testability
-----------
All network access goes through injectable `http_get` / `http_post` callables
(default: stdlib urllib, no extra dependency). Tests inject fakes returning
canned Deezer JSON, so the whole provider is verified offline. Run it live on
your machine with the real defaults.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from models import Track
from provider import MusicProvider

# Deezer's public API root (search + charts need no authentication).
_API_BASE = "https://api.deezer.com"

# Fixed BPM range for optional tempo normalization (mirrors Phase 2's store).
_TEMPO_MAX = 250.0


# ──────────────────────────────────────────────
# Emotion → Deezer mood query (freshness layer)
# Keywords follow the music-style hints in config.AETHER_EMOTIONS.
# ──────────────────────────────────────────────
EMOTION_MOOD_QUERIES: dict[str, str] = {
    "happy":       "feel good upbeat pop",
    "sad":         "sad slow ballad",
    "angry":       "aggressive heavy rock",
    "calm":        "calm ambient lofi chill",
    "anxious":     "tense dark cinematic",
    "energetic":   "edm dance workout hype",
    "focused":     "lofi study beats instrumental",
    "nostalgic":   "throwback retro acoustic",
    "romantic":    "romantic love songs rnb",
    "melancholic": "melancholic indie sad",
    "confident":   "empowering hype anthem",
    "hopeful":     "uplifting inspiring feel good",
    "frustrated":  "punk hard rock angst",
    "lonely":      "lonely sad minimal",
    "dreamy":      "dream pop ethereal ambient",
}


def _default_http_get(url: str, timeout: float = 10.0) -> dict[str, Any]:
    """GET `url` and parse JSON (stdlib only — no `requests` dependency)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Aether/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _default_http_post(url: str, timeout: float = 10.0) -> dict[str, Any]:
    """POST `url` (params encoded in the query string, Deezer-style) → JSON."""
    req = urllib.request.Request(url, data=b"", headers={"User-Agent": "Aether/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class DeezerProvider(MusicProvider):
    """Live delivery + freshness via Deezer's free public API."""

    name = "deezer"

    def __init__(
        self,
        http_get: Optional[Callable[[str], dict]] = None,
        http_post: Optional[Callable[[str], dict]] = None,
        timeout: float = 10.0,
        access_token: Optional[str] = None,
    ):
        """
        Args:
            http_get / http_post: injectable HTTP callables (default: urllib).
                Tests pass fakes returning canned Deezer JSON.
            timeout: request timeout (seconds) for the default callables.
            access_token: Deezer OAuth token, required ONLY for export_playlist.
        """
        self._http_get = http_get or (lambda u: _default_http_get(u, timeout))
        self._http_post = http_post or (lambda u: _default_http_post(u, timeout))
        self.timeout = timeout
        self.access_token = access_token

    # ── low-level search ──
    def _search(self, query: str, limit: int) -> list[dict]:
        """Run a Deezer catalog search; return the raw track dicts (best first)."""
        q = urllib.parse.quote(query)
        url = f"{_API_BASE}/search?q={q}&limit={max(1, limit)}"
        try:
            data = self._http_get(url)
        except Exception:
            return []
        return data.get("data", []) if isinstance(data, dict) else []

    # ── raw Deezer track dict → Phase 4 Track ──
    @staticmethod
    def _raw_to_track(raw: dict, source_emotion: Optional[str] = None) -> Track:
        """Map a Deezer track object into a playable Phase 4 Track."""
        artist = (raw.get("artist") or {}).get("name", "Unknown")
        album = raw.get("album") or {}
        bpm = raw.get("bpm")
        tempo = (float(bpm) / _TEMPO_MAX) if bpm else None  # optional proxy
        return Track(
            title=raw.get("title", ""),
            artist=artist,
            track_id=f"deezer:{raw.get('id')}",
            year=_year_from_date(album.get("release_date")),
            energy=None,          # Deezer doesn't expose these …
            valence=None,         # … (matching stays on the local store)
            tempo=tempo,
            match_score=None,
            source_emotion=source_emotion,
            provider_ref={
                "source": "deezer",
                "deezer_id": raw.get("id"),
                "preview_url": raw.get("preview"),
                "link": raw.get("link"),
                "cover": album.get("cover_medium") or album.get("cover"),
                "rank": raw.get("rank"),
            },
        )

    # ── MusicProvider API ──
    def enrich(self, tracks: list[Track]) -> list[Track]:
        """Resolve each store pick to playable Deezer data (in place, order kept)."""
        for t in tracks:
            if t.provider_ref:            # fresh picks already resolved
                continue
            # Deezer advanced search: match on both title and artist.
            query = f'track:"{t.title}" artist:"{t.artist}"'
            results = self._search(query, 1)
            if not results:               # fall back to a loose query
                results = self._search(f"{t.title} {t.artist}", 1)
            if results:
                ref = self._raw_to_track(results[0]).provider_ref
                t.provider_ref = ref      # attach; keep the store's features
        return tracks

    def discover(self, emotion: str, limit: int) -> list[Track]:
        """Fetch fresh current-catalog tracks for an emotion (freshness layer)."""
        if limit <= 0:
            return []
        query = EMOTION_MOOD_QUERIES.get(emotion, emotion)
        raws = self._search(query, limit)
        tracks = [self._raw_to_track(r, source_emotion=emotion) for r in raws]
        return tracks[:limit]

    def export_playlist(self, tracks: list[Track], name: str) -> dict:
        """
        Create a Deezer playlist `name` on the user's account and add `tracks`.

        Requires:
          • self.access_token (Deezer OAuth token with `manage_library` perm).
          • tracks previously enrich()'d so each carries a Deezer id.

        Returns a dict: {"playlist_id", "added", "name"}.
        """
        if not self.access_token:
            raise RuntimeError(
                "Deezer export requires an OAuth access_token "
                "(scope: manage_library). None was provided."
            )
        # Collect Deezer ids from enriched tracks.
        ids = [
            str(t.provider_ref.get("deezer_id"))
            for t in tracks
            if t.provider_ref.get("deezer_id")
        ]
        if not ids:
            raise RuntimeError("No Deezer ids on tracks — call enrich() before export.")

        # 1. Create the playlist on the current user.
        title = urllib.parse.quote(name)
        create_url = (
            f"{_API_BASE}/user/me/playlists"
            f"?title={title}&access_token={self.access_token}"
        )
        created = self._http_post(create_url)
        playlist_id = created.get("id") if isinstance(created, dict) else None
        if not playlist_id:
            raise RuntimeError(f"Deezer playlist creation failed: {created!r}")

        # 2. Add the tracks.
        songs = urllib.parse.quote(",".join(ids))
        add_url = (
            f"{_API_BASE}/playlist/{playlist_id}/tracks"
            f"?songs={songs}&access_token={self.access_token}"
        )
        self._http_post(add_url)
        return {"playlist_id": playlist_id, "added": len(ids), "name": name}


def _year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extract a 4-digit year from a Deezer release_date ('YYYY-MM-DD')."""
    if not date_str or not isinstance(date_str, str):
        return None
    head = date_str[:4]
    return int(head) if head.isdigit() else None


# ─────────────────────────────────────────────────────────────
# Self-test — mocked Deezer HTTP (no network)
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Deezer provider self-test")
    print("-" * 55)

    # A tiny canned Deezer search response.
    def fake_track(i, title, artist, bpm=None):
        d = {
            "id": 1000 + i, "title": title, "link": f"https://deezer.com/track/{1000+i}",
            "preview": f"https://cdns-preview.deezer.com/{1000+i}.mp3", "rank": 500000 - i,
            "artist": {"id": i, "name": artist},
            "album": {"id": i, "title": f"Album {i}", "cover_medium": f"https://cover/{i}.jpg",
                      "release_date": "2026-03-15"},
        }
        if bpm is not None:
            d["bpm"] = bpm
        return d

    calls: list[str] = []

    def fake_get(url: str) -> dict:
        calls.append(url)
        # 'discover' style query returns 3 tracks; 'enrich' returns 1 match.
        if "limit=1" in url:
            return {"data": [fake_track(0, "Resolved Song", "Resolved Artist", bpm=120)]}
        return {"data": [fake_track(i, f"Fresh {i}", f"Artist {i}", bpm=100 + i)
                         for i in range(3)]}

    posts: list[str] = []

    def fake_post(url: str) -> dict:
        posts.append(url)
        if "/playlists?" in url:
            return {"id": 987654}
        return {"data": True}

    prov = DeezerProvider(http_get=fake_get, http_post=fake_post, access_token="TOKEN123")

    # 1. discover() maps emotion → mood query and returns fresh, playable Tracks.
    fresh = prov.discover("dreamy", 3)
    assert len(fresh) == 3, len(fresh)
    assert all(t.track_id.startswith("deezer:") for t in fresh)
    assert all(t.provider_ref.get("preview_url") for t in fresh), "fresh not playable"
    assert all(t.energy is None for t in fresh), "Deezer shouldn't invent energy"
    assert fresh[0].year == 2026 and fresh[0].source_emotion == "dreamy"
    assert "dream pop" in urllib.parse.unquote(calls[-1]), calls[-1]  # mood query used
    print(f"  discover('dreamy',3) → {[t.title for t in fresh]}")

    # 2. enrich() resolves a store pick (no provider_ref) to playable data.
    store_pick = Track(title="Bright", artist="B", track_id="h2", energy=0.7)
    prov.enrich([store_pick])
    assert store_pick.provider_ref.get("preview_url"), "enrich didn't attach preview"
    assert store_pick.energy == 0.7, "enrich must not overwrite store features"
    assert store_pick.provider_ref["source"] == "deezer"
    print(f"  enrich(store pick) → preview attached, energy preserved ({store_pick.energy})")

    # 3. enrich() skips tracks that already have provider_ref (fresh picks).
    before = len(calls)
    prov.enrich(fresh)                 # already resolved
    assert len(calls) == before, "enrich should skip already-resolved tracks"
    print("  enrich() skips already-resolved fresh picks ✓")

    # 4. export_playlist() creates + adds (mocked), returns summary.
    res = prov.export_playlist(fresh, "Aether — Dreamy")
    assert res["playlist_id"] == 987654 and res["added"] == 3, res
    assert len(posts) == 2, posts    # one create + one add
    print(f"  export_playlist() → {res}")

    # 5. export without token → clean error.
    try:
        DeezerProvider(http_get=fake_get).export_playlist(fresh, "X")
        raise AssertionError("expected RuntimeError without access_token")
    except RuntimeError:
        pass

    # 6. All 15 Aether emotions have a mood query.
    from config import AETHER_EMOTIONS
    missing = [e for e in AETHER_EMOTIONS if e not in EMOTION_MOOD_QUERIES]
    assert not missing, f"missing mood queries: {missing}"
    print(f"  mood-query coverage → all {len(AETHER_EMOTIONS)} emotions ✓")

    print("-" * 55)
    print("✅ All Deezer-provider self-tests passed.")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Make root config importable when run standalone.
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    _selftest()
