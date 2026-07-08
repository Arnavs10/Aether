"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 4 · Delivery Provider Seam
═══════════════════════════════════════════════════════════════════
The recommendation *brain* (Phase 1–3 + recommender.py) is fully offline and
provider-agnostic. Actually *delivering* tracks — fetching current catalog
matches, preview URLs, and exporting playlists — is a separate concern that
talks to an external service.

This module defines the seam. Concrete providers are added in the live-adapter
step (next in Phase 4):

    • DeezerProvider  — default; free, no-auth search + 30s previews (works for
                        any visitor, incl. interviewers without an account).
    • SpotifyProvider — optional "Export to Spotify" for authorized users.

Keeping this as an ABC means the core never imports a specific music service:
we inject whichever provider we want (or NullProvider for pure-offline runs),
which is exactly how you'd isolate a third-party dependency in production.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from models import Track


class MusicProvider(ABC):
    """Abstract delivery provider. Concrete impls wrap Deezer / Spotify / etc."""

    #: Short identifier for logging / provenance (e.g. "deezer", "spotify").
    name: str = "abstract"

    @abstractmethod
    def enrich(self, tracks: list[Track]) -> list[Track]:
        """
        Attach playable/delivery data to each track's ``provider_ref`` (e.g.
        preview_url, streaming uri, external link, cover art). Implementations
        resolve tracks by (title, artist) against the live service.

        Must return the tracks (order preserved). Unresolvable tracks should be
        returned unchanged (empty ``provider_ref``) rather than dropped, so the
        recommendation's sequencing/ranking stays intact.
        """
        raise NotImplementedError

    @abstractmethod
    def discover(self, emotion: str, limit: int) -> list[Track]:
        """
        Source *fresh, current-catalog* tracks for an Aether `emotion` (the
        freshness layer of Hybrid C). Implementations map the emotion to a
        mood query / genre against the live service and return up to `limit`
        ready-to-play Tracks (``provider_ref`` filled, ``energy``/``valence``
        typically ``None`` since the live service doesn't expose them).

        Returns [] if the provider can't discover (e.g. offline).
        """
        raise NotImplementedError

    @abstractmethod
    def export_playlist(self, tracks: list[Track], name: str) -> dict:
        """
        Create a playlist named `name` on the user's account and add `tracks`.
        Returns a small dict describing the created playlist (id/url/…).

        Requires an authenticated user; raises if the provider can't export.
        """
        raise NotImplementedError


class NullProvider(MusicProvider):
    """
    Default no-op provider for the offline core. Leaves tracks untouched and
    refuses export (there's no live service wired). Lets the whole Phase 4
    pipeline run and be tested without any network dependency.
    """

    name = "null"

    def enrich(self, tracks: list[Track]) -> list[Track]:
        return tracks

    def discover(self, emotion: str, limit: int) -> list[Track]:
        return []

    def export_playlist(self, tracks: list[Track], name: str) -> dict:
        raise RuntimeError(
            "No live provider configured — export is unavailable in offline "
            "mode. Inject a DeezerProvider/SpotifyProvider to enable export."
        )


# ──────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────
def _run_self_tests() -> None:
    p = NullProvider()
    ts = [Track(title="x", artist="a"), Track(title="y", artist="b")]

    # enrich is a no-op passthrough (order + identity preserved).
    out = p.enrich(ts)
    assert out is ts and [t.title for t in out] == ["x", "y"]

    # discover returns nothing offline.
    assert p.discover("happy", 5) == []

    # export refuses cleanly in offline mode.
    try:
        p.export_playlist(ts, "My Playlist")
        raise AssertionError("expected RuntimeError from NullProvider.export_playlist")
    except RuntimeError:
        pass

    # MusicProvider is abstract and cannot be instantiated directly.
    try:
        MusicProvider()  # type: ignore[abstract]
        raise AssertionError("MusicProvider should be abstract")
    except TypeError:
        pass

    print("✅ All provider self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
