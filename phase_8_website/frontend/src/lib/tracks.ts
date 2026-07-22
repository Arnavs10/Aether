/**
 * AETHER — the fresh/store distinction (Pass 4 §3), in ONE place.
 * A playlist now blends store picks (measured, matchable, mixable) with
 * fresh picks sourced live from Apple's current catalogue (playable,
 * unmeasured). Every card rule flows from this predicate.
 */

import type { Track } from "./types";

/** The whole test. Zero guessing (§3.1). */
export function isFreshPick(track: Track): boolean {
  return track.track_id.startsWith("itunes:");
}

export interface ServerDelivery {
  previewUrl: string | null;
  artworkUrl: string | null;
  appleUrl: string | null;
}

/**
 * Delivery data the server already resolved (§3.2). If previewUrl is
 * present here, the track must never enter the client resolver queue.
 */
export function serverDelivery(track: Track): ServerDelivery {
  return {
    previewUrl: track.preview_url ?? null,
    artworkUrl: track.cover ?? null,
    appleUrl: track.link ?? null,
  };
}
