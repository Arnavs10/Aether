/**
 * AETHER — client-side playlist export (§6: universal, no login).
 * Mirrors the backend's `exporter.py` byte-for-byte in structure:
 *
 *   .m3u8 →  #EXTM3U
 *            #PLAYLIST:Aether — <label>
 *            #EXTINF:<seconds>,<Artist> - <Title>
 *            <preview-or-link URL | "# unresolved …" comment>
 *
 *   .json →  the full response object (tracks + reasoning + provenance),
 *            with `provider_ref` enriched client-side from iTunes.
 */

import type { Track } from "./types";
import { peekResolved } from "./itunes";

/**
 * Attach client-side provider_ref to each track from the iTunes cache (§13).
 * The API never returns delivery data; this is where the frontend's resolved
 * previews, links and artwork get folded in before export.
 */
export function withProviderRefs(tracks: Track[]): Track[] {
  return tracks.map((t) => {
    const r = peekResolved(t.title, t.artist);
    if (!r) return t;
    return {
      ...t,
      provider_ref: {
        source: "itunes",
        preview_url: r.previewUrl ?? undefined,
        link: r.appleUrl ?? undefined,
        artwork: r.artworkUrl ?? undefined,
        duration: r.durationMs ? Math.round(r.durationMs / 1000) : undefined,
      },
    };
  });
}

/** Best playable/reference URL — preview first, then link (exporter._track_url). */
function trackUrl(t: Track): string | null {
  const ref = (t.provider_ref ?? {}) as Record<string, unknown>;
  const preview = ref["preview_url"];
  const link = ref["link"];
  if (typeof preview === "string" && preview) return preview;
  if (typeof link === "string" && link) return link;
  return null;
}

/** Filesystem-safe base name (exporter._safe_filename). */
export function safeFilename(name: string): string {
  const cleaned = name
    .replace(/[^\w\- ]+/g, "")
    .trim()
    .replace(/ /g, "_");
  return cleaned || "aether_playlist";
}

/** Render Extended M3U — same line structure as exporter.to_m3u(). */
export function toM3U8(tracks: Track[], headerLabel: string): string {
  const lines: string[] = ["#EXTM3U", `#PLAYLIST:Aether — ${headerLabel}`];
  for (const t of tracks) {
    const ref = (t.provider_ref ?? {}) as Record<string, unknown>;
    const rawDuration = ref["duration"];
    const duration =
      typeof rawDuration === "number" && Number.isFinite(rawDuration)
        ? Math.trunc(rawDuration)
        : -1;
    lines.push(`#EXTINF:${duration},${t.artist} - ${t.title}`);
    const url = trackUrl(t);
    lines.push(url ?? "# unresolved — no preview resolved for this track");
  }
  return lines.join("\n") + "\n";
}

/** Full-fidelity JSON, matching exporter.to_json()'s ensure_ascii=False intent. */
export function toJson(payload: unknown): string {
  return JSON.stringify(payload, null, 2);
}

/** Trigger a browser download of a text file. */
export function downloadText(
  filename: string,
  text: string,
  mime = "text/plain",
): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Give the click a beat before revoking (Safari quirk).
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

/**
 * Export both formats for a result set.
 * @param tracks       ordered tracks (with provider_ref enriched from iTunes)
 * @param headerLabel  e.g. `"blend (moderate)"` for Curate,
 *                     `"journey (up)"` for Journey
 * @param fullPayload  the complete API response to serialize as .json
 * @param name         base filename, sanitized like the backend
 */
export function exportPlaylist(
  tracks: Track[],
  headerLabel: string,
  fullPayload: unknown,
  name = "aether_playlist",
): void {
  const base = safeFilename(name);
  downloadText(`${base}.m3u8`, toM3U8(tracks, headerLabel), "audio/x-mpegurl");
  downloadText(`${base}.json`, toJson(fullPayload), "application/json");
}
