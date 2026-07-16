/**
 * AETHER — playlist download (§13): .m3u8 and .json, works for everyone,
 * no login. provider_ref is folded in from everything the iTunes resolver
 * has learned so far, so previews resolved on screen ride along.
 */

import type { Track } from "../../lib/types";
import {
  downloadText,
  safeFilename,
  toJson,
  toM3U8,
  withProviderRefs,
} from "../../lib/exportPlaylist";

interface Props {
  tracks: Track[];
  headerLabel: string;
  payload: unknown;
  name: string;
}

export function DownloadActions({ tracks, headerLabel, payload, name }: Props) {
  const base = safeFilename(name);

  const dlM3u8 = () => {
    downloadText(
      `${base}.m3u8`,
      toM3U8(withProviderRefs(tracks), headerLabel),
      "audio/x-mpegurl",
    );
  };
  const dlJson = () => {
    const enriched = { ...(payload as Record<string, unknown>), tracks: withProviderRefs(tracks) };
    downloadText(`${base}.json`, toJson(enriched), "application/json");
  };

  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="mono-meta text-paper/40">(DOWNLOAD)</span>
      <button
        type="button"
        onClick={dlM3u8}
        className="mono-meta border hairline px-3 py-2 text-paper/70 transition-colors hover:border-blue hover:text-paper"
      >
        .m3u8
      </button>
      <button
        type="button"
        onClick={dlJson}
        className="mono-meta border hairline px-3 py-2 text-paper/70 transition-colors hover:border-blue hover:text-paper"
      >
        .json
      </button>
    </div>
  );
}
