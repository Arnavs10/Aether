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
import random
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from models import Track
from provider import MusicProvider

_API_BASE = "https://itunes.apple.com/search"


# ──────────────────────────────────────────────
# Freshness sampling
# ──────────────────────────────────────────────
# iTunes Search caps a single response at 200 results.
_ITUNES_MAX_LIMIT = 200
# Pull a much wider pool than requested, then sample. iTunes ranking is
# deterministic, so a narrow pool returns the same tracks every run and two
# curates of one mood come back nearly identical. A wide pool + sampling is what
# makes runs differ. Bigger is strictly better for variety here; the only cost
# is a single larger HTTP response.
_POOL_FACTOR = 20

# Uploader/compilation tells. Because iTunes has no popularity sort, royalty-free
# and library-music accounts rank alongside real releases for any genre keyword.
# These substrings in an ARTIST name are near-certain junk and get dropped.
_JUNK_ARTIST_MARKERS = (
    "various artist", "compilation", "karaoke", "tribute", "cover band",
    "workout", "fitness", "gym music", "study music", "lounge", "muzak",
    "academy", "orchestra hits", "meditation", "lullaby", "sleep music",
    "background music", "spa music", "cafe music", "elevator", "ringtone",
    "instrumental version", "made famous", "in the style of", "as made popular",
)


# ──────────────────────────────────────────────
# Emotion → iTunes mood query (freshness layer)
# ──────────────────────────────────────────────
# iTunes Search ranks on keyword relevance to track metadata. There is no
# recency sort and no popularity sort in the API, so THE QUERY IS THE RANKING.
#
# That makes the phrasing load-bearing, in a way worth stating plainly:
#
#   "throwback retro hits"  → Madonna, Marvin Gaye, Eurythmics, Britney Spears.
#   "edm dance workout"     → "EDM Dance Workout" by Dance Fitness.
#
# The difference is who applies the words. "Throwback" and "retro" are curatorial
# words that sit on compilations OF famous songs, so the query lands on real
# music. "Edm dance workout" is a phrase people type into a search box, so
# royalty-free uploaders name tracks exactly that to intercept it, and they bury
# everything real.
#
# So the rule for these terms: use words that describe what the music IS
# (genre, style, era) and that appear in the metadata of real commercial
# releases. Avoid phrases that read like a search query. Where a phrase is
# already proven to land on famous catalogue, keep it exactly as it is.
EMOTION_MOOD_QUERIES: dict[str, str] = {
    "happy":       "pop hits",
    "sad":         "ballads",
    "angry":       "rock",
    "calm":        "ambient",
    "anxious":     "alternative",
    "energetic":   "dance hits",
    "focused":     "instrumental",
    "nostalgic":   "throwback retro hits",   # proven: keep verbatim
    "romantic":    "love songs",
    "melancholic": "indie",
    "confident":   "hip hop",
    "hopeful":     "indie pop",
    "frustrated":  "punk",
    "lonely":      "acoustic",
    "dreamy":      "dream pop",
}


# ──────────────────────────────────────────────
# Language / market support
# ──────────────────────────────────────────────
# A listener asking for Hindi songs is asking for a different CATALOGUE, not a
# different mood. iTunes exposes that as the storefront (`country`), so the
# market rides alongside the emotion rather than replacing it: the store still
# decides how the listener feels, and the market decides which catalogue answers.
#
# Each market carries its own qualifier because a storefront alone is not enough:
# the India storefront serves plenty of English pop, so "bollywood" is what makes
# a Hindi request return Hindi music.
MARKETS: dict[str, dict[str, str]] = {
    "hindi":    {"country": "IN", "qualifier": "bollywood hindi"},
    "punjabi":  {"country": "IN", "qualifier": "punjabi"},
    "tamil":    {"country": "IN", "qualifier": "tamil"},
    "telugu":   {"country": "IN", "qualifier": "telugu"},
    "korean":   {"country": "KR", "qualifier": "k-pop korean"},
    "japanese": {"country": "JP", "qualifier": "j-pop japanese"},
    "spanish":  {"country": "ES", "qualifier": "latin spanish"},
    "french":   {"country": "FR", "qualifier": "french"},
    "english":  {"country": "US", "qualifier": ""},
}

# What a listener might type → market key. Checked against the raw request text.
MARKET_HINTS: dict[str, str] = {
    "hindi": "hindi", "bollywood": "hindi", "desi": "hindi",
    "punjabi": "punjabi", "bhangra": "punjabi",
    "tamil": "tamil", "kollywood": "tamil",
    "telugu": "telugu", "tollywood": "telugu",
    "korean": "korean", "k-pop": "korean", "kpop": "korean",
    "japanese": "japanese", "j-pop": "japanese", "jpop": "japanese", "anime": "japanese",
    "spanish": "spanish", "latino": "spanish", "reggaeton": "spanish",
    "french": "french",
    "english": "english",
}


def detect_market(text: Optional[str]) -> Optional[str]:
    """Find a language/market request in a listener's own words.

    Returns a key of MARKETS, or None when no language was asked for (in which
    case the provider's configured storefront applies, and nothing changes).

    Deliberately a keyword match rather than language detection: the request is
    "give me Hindi songs", typically written IN English, so detecting the
    language of the sentence would answer the wrong question.

    Args:
        text: the listener's raw request, or None.

    Returns:
        A market key such as "hindi", or None.
    """
    if not text:
        return None
    low = f" {text.lower()} "
    for hint, market in MARKET_HINTS.items():
        if hint in low:
            return market
    return None


def _is_search_bait(raw: dict, term_words: set[str]) -> bool:
    """True if a result is a keyword-farmed upload rather than a real release.

    Because iTunes ranks on metadata relevance with no popularity sort, two kinds
    of junk rank alongside real music for any genre keyword:

      • Title bait: a track literally titled "EDM Dance Workout" wins the query
        "edm dance workout". The tell is the title echoing the query back — real
        songs rarely carry two or more of a mood query's words in their title.
      • Junk artists: royalty-free, karaoke, workout, lullaby and library-music
        accounts. The tell is a known marker substring in the artist name.

    Either tell drops the result. The caller falls back to the unfiltered pool if
    filtering leaves too few, so this can never empty a playlist.

    Args:
        raw: one iTunes search result.
        term_words: the query's significant words, lowercased.

    Returns:
        True if the result should be dropped.
    """
    artist = (raw.get("artistName") or "").lower()
    if any(marker in artist for marker in _JUNK_ARTIST_MARKERS):
        return True

    title = (raw.get("trackName") or "").lower()
    if not title or len(term_words) < 2:
        return False
    return sum(1 for w in term_words if w in title) >= 2


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
    def _search(self, term: str, limit: int,
                country: Optional[str] = None) -> list[dict]:
        """Run an iTunes song search; return the raw result dicts (best first).

        Args:
            term: the search term.
            limit: how many results to ask for (iTunes caps at 200).
            country: storefront override, e.g. "IN". Defaults to the provider's.
        """
        params = urllib.parse.urlencode({
            "term": term, "media": "music", "entity": "song",
            "limit": max(1, limit), "country": country or self.country,
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

    def discover(self, emotion: str, limit: int,
                 market: Optional[str] = None,
                 exclude_titles: Optional[set] = None) -> list[Track]:
        """Fetch fresh current-catalog tracks for an emotion (freshness layer).

        Four things happen here, each fixing a way iTunes Search misbehaves when
        used as a discovery engine:

        1. Market. If the listener asked for a language, search that storefront
           with that language's qualifier. The emotion is unchanged: the store
           still decided how they feel, this only decides which catalogue answers.
        2. Over-fetch. iTunes ranking is deterministic, so asking for exactly
           `limit` returns the same tracks on every identical call, and two
           curates of one mood come back twins. Pull a wide pool and sample it.
        3. Junk filter. Drop keyword-farmed uploads and library-music artists
           (see _is_search_bait), so "dance" returns music, not "Dance Workout
           Hits" by a compilation account.
        4. De-duplication. Skip anything whose "title — artist" is already in
           `exclude_titles`, so a track never appears twice in one playlist,
           whether it collided with the store half or with another emotion in
           the same blend.

        Args:
            emotion: one of the 15 Aether emotions.
            limit: how many tracks to return.
            market: optional key of MARKETS ("hindi", "korean", …).
            exclude_titles: lowercased "title — artist" keys already used. This
                set is MUTATED as picks are chosen, so a caller can thread one set
                through several discover() calls to keep the whole playlist unique.

        Returns:
            Up to `limit` playable Tracks. Empty on any provider failure —
            freshness is best-effort and never breaks the core recommendation.
        """
        if limit <= 0:
            return []
        seen_keys = exclude_titles if exclude_titles is not None else set()

        term = EMOTION_MOOD_QUERIES.get(emotion, emotion)
        country = self.country
        if market and market in MARKETS:
            cfg = MARKETS[market]
            country = cfg["country"]
            if cfg["qualifier"]:
                term = f"{cfg['qualifier']} {term}"

        pool_size = min(max(limit * _POOL_FACTOR, limit), _ITUNES_MAX_LIMIT)
        raws = self._search(term, pool_size, country=country)
        if not raws:
            print(f"[freshness] iTunes returned nothing for {emotion!r} "
                  f"({term!r}, {country}).")
            return []

        term_words = {w for w in term.lower().split() if len(w) > 2}
        clean = [r for r in raws if not _is_search_bait(r, term_words)]
        if len(clean) < limit:
            clean = raws          # filter too aggressive here; relevance wins

        # Shuffle, then take the first `limit` that aren't already used. Shuffling
        # (not sampling a fixed slice) is what makes repeated runs differ.
        random.shuffle(clean)
        out: list[Track] = []
        for raw in clean:
            key = f"{(raw.get('trackName') or '').lower()} — " \
                  f"{(raw.get('artistName') or '').lower()}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(self._raw_to_track(raw, source_emotion=emotion))
            if len(out) >= limit:
                break
        return out

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
