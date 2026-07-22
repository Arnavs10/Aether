/**
 * AETHER — the one shared player (Pass 4 §9). Two <audio> elements so a
 * Live drift can run a REAL crossfade: equal-power volume ramps on plain
 * media elements, no Web Audio, no crossOrigin (Apple's preview CDN gives
 * no CORS promises). One track plays at a time across the whole app.
 * The volume ramp rides gsap's ticker: one rAF for everything.
 */

import { gsap } from "./gsap";

export type PlayStatus = "loading" | "playing" | "paused";

export interface PlayerState {
  key: string | null;
  status: PlayStatus;
}

let elA: HTMLAudioElement | null = null;
let elB: HTMLAudioElement | null = null;
let active: HTMLAudioElement | null = null;

let state: PlayerState = { key: null, status: "paused" };
const listeners = new Set<(s: PlayerState) => void>();
let endedCb: (() => void) | null = null;
let fading = false;

function emit(): void {
  for (const cb of listeners) cb(state);
}

function make(): HTMLAudioElement {
  const el = new Audio();
  el.preload = "auto";
  el.addEventListener("ended", () => {
    if (el !== active || fading) return;
    state = { key: null, status: "paused" };
    emit();
    const cb = endedCb;
    endedCb = null;
    cb?.();
  });
  el.addEventListener("playing", () => {
    if (el !== active) return;
    if (state.status !== "playing") {
      state = { ...state, status: "playing" };
      emit();
    }
  });
  return el;
}

function ensure(): void {
  if (!elA) elA = make();
  if (!elB) elB = make();
  if (!active) active = elA;
}

function standby(): HTMLAudioElement {
  ensure();
  return active === elA ? (elB as HTMLAudioElement) : (elA as HTMLAudioElement);
}

/* A tiny silent WAV. Playing it inside a user gesture "unlocks" both
   elements so later programmatic playback (route autoplay, crossfades)
   is allowed even after async waits (Pass 5 §2). */
const SILENT_WAV =
  "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQAAAAAAAAA";

/** Call synchronously inside a click that will later start playback. */
export function unlock(): void {
  ensure();
  for (const el of [elA, elB]) {
    if (!el) continue;
    try {
      el.muted = true;
      el.src = SILENT_WAV;
      const p = el.play();
      if (p) {
        p.then(() => {
          el.pause();
          el.muted = false;
        }).catch(() => {
          el.muted = false;
        });
      }
    } catch {
      el.muted = false;
    }
  }
}

/** Subscribe to playback state; called immediately with the current state. */
export function subscribePlayer(cb: (s: PlayerState) => void): () => void {
  listeners.add(cb);
  cb(state);
  return () => {
    listeners.delete(cb);
  };
}

/** Play a url under a key, replacing whatever is playing. */
export function play(
  key: string,
  url: string,
  opts: { onEnded?: () => void } = {},
): void {
  ensure();
  if (fading) return;
  const el = active as HTMLAudioElement;
  standby().pause();
  endedCb = opts.onEnded ?? null;
  state = { key, status: "loading" };
  emit();
  el.volume = 1;
  el.src = url;
  el.play().catch(() => {
    state = { key: null, status: "paused" };
    emit();
  });
}

/** Toggle a key: play it, pause it, or resume it. */
export function toggle(
  key: string,
  url: string,
  opts: { onEnded?: () => void } = {},
): void {
  ensure();
  if (fading) return;
  const el = active as HTMLAudioElement;
  if (state.key === key) {
    if (state.status === "playing") {
      el.pause();
      state = { ...state, status: "paused" };
      emit();
    } else {
      state = { ...state, status: "loading" };
      emit();
      el.play().catch(() => {
        state = { key: null, status: "paused" };
        emit();
      });
    }
    return;
  }
  play(key, url, opts);
}

export function stop(): void {
  if (fading) return;
  active?.pause();
  endedCb = null;
  state = { key: null, status: "paused" };
  emit();
}

/** Seconds of audio left on the current track (Infinity if unknown). */
export function remainingSeconds(): number {
  const el = active;
  if (!el || !Number.isFinite(el.duration)) return Infinity;
  return Math.max(0, el.duration - el.currentTime);
}

/**
 * The real crossfade (§9.3): buffer the next preview, then equal-power
 * fade A→B over the requested duration, capped against the audio that
 * actually remains. Resolves true on success, false if B never buffered
 * (caller does the visual-only fallback).
 */
export function crossfade(
  nextKey: string,
  nextUrl: string,
  durationS: number,
  opts: { onEnded?: () => void; onStart?: (actualDurationS: number) => void } = {},
): Promise<boolean> {
  ensure();
  return new Promise((resolve) => {
    if (fading || state.status !== "playing") {
      // Nothing audible to fade from: just play the next track.
      play(nextKey, nextUrl, opts);
      resolve(true);
      return;
    }
    const from = active as HTMLAudioElement;
    const to = standby();

    let settled = false;
    const bufferTimeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      to.removeEventListener("canplaythrough", onReady);
      resolve(false);
    }, 8_000);

    const onReady = () => {
      if (settled) return;
      settled = true;
      clearTimeout(bufferTimeout);

      // §3.2: a real, audible cross-blend. Floor the fade at ~5s so the
      // overlap is felt even when the engine plans a short one, still
      // capped by the audio that actually remains.
      const dur = Math.min(
        Math.max(durationS, 5),
        Math.max(2, remainingSeconds() - 0.4),
        26,
      );

      fading = true;
      opts.onStart?.(dur);
      to.volume = 0;
      void to.play().catch(() => {
        fading = false;
        resolve(false);
      });

      state = { key: nextKey, status: "playing" };
      emit();

      const start = performance.now();
      const tick = () => {
        const t = Math.min(1, (performance.now() - start) / (dur * 1000));
        // Equal-power: sounds like a mix, not a dip.
        from.volume = Math.cos((t * Math.PI) / 2);
        to.volume = Math.sin((t * Math.PI) / 2);
        if (t >= 1) {
          gsap.ticker.remove(tick);
          from.pause();
          from.volume = 1;
          active = to;
          endedCb = opts.onEnded ?? null;
          fading = false;
          emit();
          resolve(true);
        }
      };
      gsap.ticker.add(tick);
    };

    to.addEventListener("canplaythrough", onReady, { once: true });
    to.src = nextUrl;
    to.load();
  });
}
