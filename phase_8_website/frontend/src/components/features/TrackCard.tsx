/**
 * AETHER — the shared track card (Pass 4 §3 rules baked in).
 * Two kinds of track render here. Store picks: match %, feature meters,
 * live-mixable. Fresh picks (`itunes:` ids): an honest (FRESH) marker,
 * no percentages, no meters, never live-mix. Both: `why` always visible,
 * a reveal that never looks hollow, artwork/preview from server delivery
 * first (§3.2) with the resolver as fallback, cards complete immediately
 * with only the artwork square pending (§8.3.1).
 */

import { useRef, useState } from "react";
import type { Track } from "../../lib/types";
import { isFreshPick } from "../../lib/tracks";
import { deepLinks } from "../../lib/itunes";
import { useTrackDelivery } from "../player/useTrackDelivery";
import { PlayButton } from "../player/PlayButton";

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
  /** Journey route mode: highlight + intercept plays. */
  active?: boolean;
  onPlayIntent?: () => void;
}

export function TrackCard({ track, onLiveMix, active = false, onPlayIntent }: Props) {
  const cardRef = useRef<HTMLElement>(null);
  const delivery = useTrackDelivery(track, cardRef);
  const [openWhy, setOpenWhy] = useState(false);

  const fresh = isFreshPick(track);
  const links = deepLinks(track.title, track.artist);
  const tech = (track.why_technical ?? "").trim();
  const matchPct =
    !fresh && typeof track.match_score === "number"
      ? Math.round(track.match_score * 100)
      : null;

  return (
    <article
      ref={cardRef}
      className={`glass rounded-sm p-5 transition-shadow duration-300 md:p-6 ${
        active ? "border-blue [box-shadow:0_0_22px_rgba(46,107,255,0.3)]" : ""
      }`}
    >
      <div className="flex items-center gap-4 md:gap-5">
        <span className="mono-meta w-7 shrink-0 text-paper/40">
          {String(track.rank).padStart(2, "0")}
        </span>

        {/* artwork: server-delivered or resolver; skeleton is a finished state */}
        <div className="relative h-14 w-14 shrink-0 overflow-hidden border hairline bg-ink">
          {delivery.artworkUrl ? (
            <img
              src={delivery.artworkUrl}
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
          <p className="truncate text-sm text-mist">
            {track.artist}
            {track.year ? <span className="text-paper/30"> · {track.year}</span> : null}
          </p>
        </div>

        <div className="hidden shrink-0 flex-col items-end gap-1 sm:flex">
          {fresh ? (
            <span className="mono-meta text-gold">(FRESH)</span>
          ) : matchPct !== null ? (
            <span className="mono-meta text-blue">{matchPct}% MATCH</span>
          ) : null}
          <span className="mono-meta text-paper/35">({track.source_emotion})</span>
        </div>

        <PlayButton
          trackKey={track.track_id}
          previewUrl={delivery.previewUrl}
          resolving={delivery.resolving}
          onPlayIntent={onPlayIntent}
        />
      </div>

      {/* why: populated on both kinds now, always visible */}
      {track.why && (
        <p className="mt-4 max-w-3xl text-sm leading-relaxed text-mist">{track.why}</p>
      )}

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
          href={delivery.appleUrl ?? links.youtube}
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
        {/* Live mixes on measured key + tempo, which a fresh pick lacks (§3). */}
        {onLiveMix && !fresh && (
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
          {fresh ? (
            /* Fresh reveal: intentionally different, never hollow. */
            <div className="mt-5 flex flex-col gap-3 border-t hairline pt-5 text-sm leading-relaxed text-mist">
              <p>
                Pulled fresh from Apple's current catalogue and placed inside
                the <span className="text-paper/80">{track.source_emotion}</span>{" "}
                mood{track.album ? (
                  <>
                    , from <span className="text-paper/70">{track.album}</span>
                  </>
                ) : null}
                . New releases carry no measured character yet, so it earns its
                seat by fit, not by a score.
              </p>
              {tech && <p>{tech}</p>}
            </div>
          ) : (
            <div className="mt-5 grid gap-6 border-t hairline pt-5 md:grid-cols-[1fr_1fr]">
              <div className="flex flex-col gap-4">
                {typeof track.energy === "number" && <Meter label="ENERGY" value={track.energy} />}
                {typeof track.valence === "number" && <Meter label="VALENCE" value={track.valence} />}
                {typeof track.tempo === "number" && <Meter label="TEMPO" value={track.tempo} />}
              </div>
              <div className="flex flex-col gap-3 text-sm leading-relaxed text-mist">
                {matchPct !== null && (
                  <p>
                    Picked for <span className="text-paper/80">{track.source_emotion}</span>,
                    sitting at a {matchPct}% match to the target this playlist
                    was built around.
                  </p>
                )}
                {tech && <p>{tech}</p>}
              </div>
            </div>
          )}
        </div>
      </div>
    </article>
  );
}
