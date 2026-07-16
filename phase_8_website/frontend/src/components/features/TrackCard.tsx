/**
 * AETHER — the shared track card (§7, reused by §8 and §9).
 * Rank, title, artist, match. `why` always visible. The reasoning reveal is
 * built from the fields that are ALWAYS present (source emotion, match,
 * energy / valence / tempo); `why_technical` folds in only when non-empty,
 * so the panel never renders hollow (§2.4 note).
 * Delivery: artwork + 30s preview + Apple link resolved through iTunes at
 * render time; Spotify and YouTube as search deep-links (§13).
 */

import { useEffect, useState } from "react";
import type { Track } from "../../lib/types";
import { deepLinks, resolveTrack, type ItunesResolved } from "../../lib/itunes";
import { subscribePlayback, togglePreview } from "../../lib/audio";
import { BarsLoader } from "../ui/BarsLoader";

function Meter({ label, value }: { label: string; value: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="mono-meta text-paper/45">{label}</span>
        <span className="mono-meta text-paper/60">{pct}</span>
      </div>
      <div className="mt-1.5 h-px w-full bg-paper/10">
        <div className="h-px bg-blue" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

interface Props {
  track: Track;
  onLiveMix?: (track: Track) => void;
}

export function TrackCard({ track, onLiveMix }: Props) {
  const [resolved, setResolved] = useState<ItunesResolved | null | "pending">("pending");
  const [openWhy, setOpenWhy] = useState(false);
  const [playingUrl, setPlayingUrl] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    resolveTrack(track.title, track.artist).then((r) => {
      if (alive) setResolved(r);
    });
    return () => {
      alive = false;
    };
  }, [track.title, track.artist]);

  useEffect(() => subscribePlayback(setPlayingUrl), []);

  const preview = resolved !== "pending" ? resolved?.previewUrl ?? null : null;
  const isPlaying = preview !== null && playingUrl === preview;
  const links = deepLinks(track.title, track.artist);
  const tech = (track.why_technical ?? "").trim();

  return (
    <article className="glass rounded-sm p-5 md:p-6">
      <div className="flex items-center gap-4 md:gap-5">
        <span className="mono-meta w-7 shrink-0 text-paper/40">
          {String(track.rank).padStart(2, "0")}
        </span>

        {/* artwork / placeholder */}
        <div className="relative h-14 w-14 shrink-0 overflow-hidden border hairline bg-ink">
          {resolved !== "pending" && resolved?.artworkUrl ? (
            <img
              src={resolved.artworkUrl}
              alt=""
              loading="lazy"
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="flex h-full w-full items-end justify-center gap-[3px] pb-2 opacity-40">
              <span className="h-3 w-[3px] bg-silver" />
              <span className="h-5 w-[3px] bg-blue" />
              <span className="h-2 w-[3px] bg-silver" />
            </div>
          )}
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="truncate text-base font-medium text-paper md:text-lg">
            {track.title}
          </h3>
          <p className="truncate text-sm text-mist">{track.artist}</p>
        </div>

        <div className="hidden shrink-0 flex-col items-end gap-1 sm:flex">
          <span className="mono-meta text-blue">
            {Math.round(track.match_score * 100)}% MATCH
          </span>
          <span className="mono-meta text-paper/35">({track.source_emotion})</span>
        </div>

        {/* preview */}
        <button
          type="button"
          onClick={() => preview && togglePreview(preview)}
          disabled={!preview}
          aria-label={isPlaying ? "Pause preview" : "Play 30 second preview"}
          title={
            resolved === "pending"
              ? "finding the song…"
              : preview
                ? "30 second preview"
                : "no preview found, links below still work"
          }
          className={`flex h-11 w-11 shrink-0 items-center justify-center border transition-all duration-200 ${
            isPlaying
              ? "border-blue [box-shadow:0_0_16px_rgba(46,107,255,0.4)]"
              : "hairline hover:border-paper/35"
          } disabled:opacity-35`}
        >
          {resolved === "pending" ? (
            <BarsLoader />
          ) : isPlaying ? (
            <span className="flex gap-[3px]">
              <span className="h-3.5 w-[3px] bg-paper" />
              <span className="h-3.5 w-[3px] bg-paper" />
            </span>
          ) : (
            <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
              <path d="M7 4 L19 12 L7 20 Z" fill="var(--color-paper)" opacity="0.8" />
            </svg>
          )}
        </button>
      </div>

      {/* why, always visible when present */}
      {track.why && (
        <p className="mt-4 max-w-3xl text-sm leading-relaxed text-mist">{track.why}</p>
      )}

      {/* the reasoning reveal */}
      <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
        <button
          type="button"
          onClick={() => setOpenWhy((v) => !v)}
          aria-expanded={openWhy}
          className="mono-meta border-b border-paper/25 pb-0.5 text-paper/60 transition-colors hover:border-blue hover:text-paper"
        >
          {openWhy ? "close the reasoning" : "(the reasoning)"}
        </button>
        <a
          href={resolved !== "pending" ? resolved?.appleUrl ?? links.youtube : links.youtube}
          target="_blank"
          rel="noreferrer"
          className="mono-meta text-paper/45 transition-colors hover:text-paper"
        >
          apple ↗
        </a>
        <a href={links.spotify} target="_blank" rel="noreferrer" className="mono-meta text-paper/45 transition-colors hover:text-paper">
          spotify ↗
        </a>
        <a href={links.youtube} target="_blank" rel="noreferrer" className="mono-meta text-paper/45 transition-colors hover:text-paper">
          youtube ↗
        </a>
        {onLiveMix && (
          <button
            type="button"
            onClick={() => onLiveMix(track)}
            className="mono-meta text-gold/80 transition-colors hover:text-gold"
          >
            live-mix this →
          </button>
        )}
      </div>

      <div
        className="grid transition-[grid-template-rows] duration-500 ease-out"
        style={{ gridTemplateRows: openWhy ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <div className="mt-5 grid gap-6 border-t hairline pt-5 md:grid-cols-[1fr_1fr]">
            <div className="flex flex-col gap-4">
              <Meter label="ENERGY" value={track.energy} />
              <Meter label="VALENCE" value={track.valence} />
              <Meter label="TEMPO" value={track.tempo} />
            </div>
            <div className="flex flex-col gap-3 text-sm leading-relaxed text-mist">
              <p>
                Picked for <span className="text-paper/80">{track.source_emotion}</span>,
                sitting at a {Math.round(track.match_score * 100)}% match to the
                target this playlist was built around.
              </p>
              {tech && <p>{tech}</p>}
            </div>
          </div>
        </div>
      </div>
    </article>
  );
}
