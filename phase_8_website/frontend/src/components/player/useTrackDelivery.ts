/**
 * AETHER — per-track delivery (Pass 4 §3.2 + §8.3).
 * Server-resolved fields win outright: if the track shipped with a
 * preview_url it NEVER enters the resolver queue. Everything else goes
 * through the priority queue, and an IntersectionObserver on the card
 * boosts whatever the user can actually see.
 */

import { useEffect, useState, type RefObject } from "react";
import type { Track } from "../../lib/types";
import { serverDelivery } from "../../lib/tracks";
import { boostResolve, resolveTrack } from "../../lib/itunes";

export interface Delivery {
  previewUrl: string | null;
  artworkUrl: string | null;
  appleUrl: string | null;
  /** True only while the client resolver is still looking. */
  resolving: boolean;
}

export function useTrackDelivery(
  track: Track,
  viewRef?: RefObject<Element | null>,
  basePriority = 0,
): Delivery {
  const server = serverDelivery(track);
  const needsResolver = !server.previewUrl && !server.artworkUrl;

  const [resolved, setResolved] = useState<{
    previewUrl: string | null;
    artworkUrl: string | null;
    appleUrl: string | null;
  } | null>(null);
  const [resolving, setResolving] = useState(needsResolver);

  useEffect(() => {
    if (!needsResolver) return;
    let alive = true;
    setResolving(true);
    resolveTrack(track.title, track.artist, basePriority).then((r) => {
      if (!alive) return;
      setResolved(
        r
          ? { previewUrl: r.previewUrl, artworkUrl: r.artworkUrl, appleUrl: r.appleUrl }
          : { previewUrl: null, artworkUrl: null, appleUrl: null },
      );
      setResolving(false);
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [track.title, track.artist, needsResolver]);

  // Viewport boost: what's on screen resolves first (§8.3.2).
  useEffect(() => {
    const el = viewRef?.current;
    if (!el || !needsResolver) return;
    const io = new IntersectionObserver(([entry]) => {
      if (entry?.isIntersecting) boostResolve(track.title, track.artist);
    });
    io.observe(el);
    return () => io.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewRef, needsResolver, track.title, track.artist]);

  return {
    previewUrl: server.previewUrl ?? resolved?.previewUrl ?? null,
    artworkUrl: server.artworkUrl ?? resolved?.artworkUrl ?? null,
    appleUrl: server.appleUrl ?? resolved?.appleUrl ?? null,
    resolving: needsResolver && resolving,
  };
}
