"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 4 · Open-Format Playlist Exporter
═══════════════════════════════════════════════════════════════════
Provider-independent export: turn a `Recommendation` into files any player /
service can read — no OAuth, no account, works for every user.

  • .m3u8 — Extended M3U (with #EXTINF metadata + the 30s preview URLs from
            the provider). Opens in VLC, most players, and streams the previews.
  • .json — the full recommendation (ordered tracks + reasoning + provenance),
            for the API, the website, or re-import.

This complements — doesn't replace — the streaming-account export on the
provider (Deezer/Spotify `export_playlist`). Open formats are the universal
fallback that always works.

Note: run the provider's enrich() before exporting so tracks carry preview
URLs; the .m3u still lists every track either way (URL omitted if unresolved).
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from models import Recommendation, Track


def _track_url(track: Track) -> Optional[str]:
    """Best playable/reference URL for a track: preview first, then link."""
    ref = track.provider_ref or {}
    return ref.get("preview_url") or ref.get("link")


def to_m3u(recommendation: Recommendation) -> str:
    """
    Render the recommendation as an Extended M3U (UTF-8 / .m3u8) string.

    Each track becomes:
        #EXTINF:<seconds>,<Artist> - <Title>
        <preview-or-link URL>
    Tracks with no resolved URL still appear as a commented #EXTINF so the
    playlist stays complete and self-documenting.
    """
    lines: list[str] = ["#EXTM3U", f"#PLAYLIST:Aether — {recommendation.intent_mode} "
                         f"({recommendation.intensity_label})"]
    for t in recommendation.tracks:
        duration = int((t.provider_ref or {}).get("duration", -1) or -1)
        label = f"{t.artist} - {t.title}"
        url = _track_url(t)
        if url:
            lines.append(f"#EXTINF:{duration},{label}")
            lines.append(url)
        else:
            # keep the entry, but mark it unresolved (no playable resource)
            lines.append(f"#EXTINF:{duration},{label}")
            lines.append(f"# unresolved — run provider.enrich() to attach a preview")
    return "\n".join(lines) + "\n"


def to_json(recommendation: Recommendation, indent: int = 2) -> str:
    """Render the full recommendation (tracks + reasoning) as a JSON string."""
    return json.dumps(recommendation.as_dict(), indent=indent, ensure_ascii=False)


def _safe_filename(name: str) -> str:
    """Filesystem-safe base filename (no extension)."""
    cleaned = re.sub(r"[^\w\- ]+", "", name).strip().replace(" ", "_")
    return cleaned or "aether_playlist"


def write_m3u(recommendation: Recommendation, path: str) -> str:
    """Write the .m3u8 to `path`; returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_m3u(recommendation), encoding="utf-8")
    return str(p)


def write_json(recommendation: Recommendation, path: str) -> str:
    """Write the .json to `path`; returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json(recommendation), encoding="utf-8")
    return str(p)


def export(
    recommendation: Recommendation,
    out_dir: str = ".",
    name: str = "aether_playlist",
) -> dict[str, str]:
    """
    Write both formats into `out_dir` using a safe base name.

    Returns:
        {"m3u": <path>, "json": <path>}
    """
    base = _safe_filename(name)
    out = Path(out_dir)
    m3u_path = write_m3u(recommendation, str(out / f"{base}.m3u8"))
    json_path = write_json(recommendation, str(out / f"{base}.json"))
    return {"m3u": m3u_path, "json": json_path}


# ─────────────────────────────────────────────────────────────
# Self-test — no network, no external state
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Exporter self-test")
    print("-" * 55)

    tracks = [
        Track(title="Sunshine", artist="A", track_id="h1", rank=1, energy=0.85,
              provider_ref={"source": "deezer", "preview_url": "https://x/1.mp3",
                            "duration": 210, "link": "https://deezer.com/track/1"}),
        Track(title="Vibes", artist="B", track_id="h2", rank=2, energy=0.95,
              provider_ref={"source": "deezer", "preview_url": "https://x/2.mp3",
                            "duration": 195}),
        # a not-yet-enriched track (no provider_ref) — must still appear
        Track(title="Bright", artist="C", track_id="h3", rank=3, energy=0.70),
    ]
    rec = Recommendation(
        tracks=tracks, request_text="happy songs", intent_mode="single",
        intensity_level=3, intensity_label="intense",
        dominant_emotions=[("happy", 1.0)], arc_shape="arc",
        reason="An intense 'happy' playlist. Tracks arranged as an arc.",
    )

    # 1. M3U structure: header + one #EXTINF per track.
    m3u = to_m3u(rec)
    assert m3u.startswith("#EXTM3U"), m3u[:20]
    assert m3u.count("#EXTINF:") == 3, m3u.count("#EXTINF:")
    assert "https://x/1.mp3" in m3u and "https://x/2.mp3" in m3u
    assert "A - Sunshine" in m3u
    assert "unresolved" in m3u  # the un-enriched track is flagged, not dropped
    print(f"  to_m3u() → header + {m3u.count('#EXTINF:')} entries, unresolved flagged ✓")

    # 2. JSON round-trips and preserves order + reasoning.
    parsed = json.loads(to_json(rec))
    assert parsed["size"] == 3 and parsed["intent_mode"] == "single"
    assert [t["title"] for t in parsed["tracks"]] == ["Sunshine", "Vibes", "Bright"]
    assert "reason" in parsed and parsed["tracks"][0]["rank"] == 1
    print("  to_json() → order + reasoning preserved, valid JSON ✓")

    # 3. write both to a temp dir with a sanitized name.
    import tempfile, os
    tmp = tempfile.mkdtemp()
    paths = export(rec, out_dir=tmp, name="Aether — Happy Mix!")
    assert os.path.exists(paths["m3u"]) and os.path.exists(paths["json"])
    assert paths["m3u"].endswith("Aether__Happy_Mix.m3u8"), paths["m3u"]
    print(f"  export() → {os.path.basename(paths['m3u'])}, {os.path.basename(paths['json'])} ✓")

    # 4. Empty recommendation doesn't crash.
    empty = Recommendation(tracks=[], request_text="", intent_mode="single",
                           intensity_level=0, intensity_label="neutral",
                           dominant_emotions=[], arc_shape="steady", reason="")
    assert to_m3u(empty).startswith("#EXTM3U")
    assert json.loads(to_json(empty))["size"] == 0
    print("  empty recommendation → handled ✓")

    print("-" * 55)
    print("✅ All exporter self-tests passed.")


if __name__ == "__main__":
    _selftest()
