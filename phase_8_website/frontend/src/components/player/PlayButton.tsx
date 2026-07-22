/**
 * AETHER — the one play control (Pass 4 §9.1), used by every card and the
 * Live player. Visible states: playing, paused, loading, resolving,
 * unavailable (calm, with the links still doing the work).
 */

import { useEffect, useState } from "react";
import { subscribePlayer, toggle, type PlayerState } from "../../lib/audio";
import { BarsLoader } from "../ui/BarsLoader";

interface Props {
  trackKey: string;
  previewUrl: string | null;
  resolving?: boolean;
  /** Journey's route mode intercepts plays to keep the queue in charge. */
  onPlayIntent?: () => void;
  size?: "md" | "lg";
  onEnded?: () => void;
}

export function PlayButton({
  trackKey,
  previewUrl,
  resolving = false,
  onPlayIntent,
  size = "md",
  onEnded,
}: Props) {
  const [player, setPlayer] = useState<PlayerState>({ key: null, status: "paused" });
  useEffect(() => subscribePlayer(setPlayer), []);

  const mine = player.key === trackKey;
  const playing = mine && player.status === "playing";
  const loading = mine && player.status === "loading";
  const box = size === "lg" ? "h-14 w-14" : "h-11 w-11";

  const onClick = () => {
    if (!previewUrl) return;
    if (onPlayIntent && !mine) {
      onPlayIntent();
      return;
    }
    toggle(trackKey, previewUrl, { onEnded });
  };

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!previewUrl && !resolving}
      aria-label={
        playing
          ? "Pause preview"
          : previewUrl
            ? "Play 30 second preview"
            : "No preview available"
      }
      title={
        resolving
          ? "finding the song…"
          : previewUrl
            ? "30 second preview"
            : "no preview on iTunes for this one. the links still work"
      }
      className={`flex ${box} shrink-0 items-center justify-center border transition-all duration-200 ${
        playing
          ? "border-blue [box-shadow:0_0_16px_rgba(46,107,255,0.4)]"
          : "hairline hover:border-paper/35"
      } disabled:opacity-35`}
    >
      {resolving || loading ? (
        <BarsLoader tone={loading ? "blue" : "silver"} />
      ) : playing ? (
        <span className="flex gap-[3px]">
          <span className="h-3.5 w-[3px] bg-paper" />
          <span className="h-3.5 w-[3px] bg-paper" />
        </span>
      ) : (
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            d="M7 4 L19 12 L7 20 Z"
            fill="var(--color-paper)"
            opacity={previewUrl ? 0.8 : 0.35}
          />
        </svg>
      )}
    </button>
  );
}
