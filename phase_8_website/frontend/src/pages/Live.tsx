/**
 * AETHER — Live (Pass 4 §9.3 + §10). The demo of the whole feature:
 * the seed PLAYS, and a triggered drift runs a REAL crossfade, two audio
 * elements ramping equal-power over the duration the engine specifies,
 * with the visual mixing animation synced to the actual audio. Store-only
 * by physics: mixing needs measured key and tempo, so seeds and
 * transitions come from the library, framed as the craft it is (§10.1).
 * The seed picker (§10.2): 50 loaded up front, shuffle accumulates,
 * filter-as-you-type over what's loaded, honestly labelled.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router";
import { gsap, useGSAP } from "../lib/gsap";
import { ApiError, getEmotions, getTracks, liveObserve, liveStart } from "../lib/api";
import type {
  CatalogTrack,
  CrossfadePlan,
  LiveNextTrack,
  LiveObserveRequest,
} from "../lib/types";
import { AETHER_EMOTIONS } from "../config/site";
import { deepLinks, resolveTrack } from "../lib/itunes";
import { crossfade, stop, subscribePlayer, unlock, type PlayerState } from "../lib/audio";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { PageHeader } from "../components/ui/PageScaffold";
import { ParenLabel } from "../components/ui/ParenLabel";
import { BarsLoader } from "../components/ui/BarsLoader";
import { EmotionChips } from "../components/features/EmotionChips";
import { VoiceMic } from "../components/features/VoiceMic";
import { PlayButton } from "../components/player/PlayButton";

interface Seed {
  track_id: string;
  title: string;
  artist: string;
  camelot?: string;
  bpm?: number;
}

interface Playing extends Seed {
  previewUrl: string | null;
  resolving: boolean;
}

interface LogEntry {
  id: number;
  headline: string;
  meta: string;
  kind: "info" | "baseline" | "drift";
}

interface MixState {
  phase: "buffering" | "fading";
  plan: CrossfadePlan;
  next: LiveNextTrack;
  reason: string;
  /** Actual fade seconds once known; visuals run against this. */
  durS: number;
  /** True when real audio is fading; false = visual-only fallback. */
  audio: boolean;
}

function nextField(next: LiveNextTrack, key: string): string | undefined {
  const v = next[key];
  return typeof v === "string" && v ? v : undefined;
}

export default function Live() {
  const location = useLocation();
  const reduced = usePrefersReducedMotion();
  const handoff = (location.state as { seed?: Seed } | null)?.seed;

  const [labels, setLabels] = useState<string[]>([...AETHER_EMOTIONS]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [nowPlaying, setNowPlaying] = useState<Playing | null>(null);
  const [artwork, setArtwork] = useState<string | null>(null);
  const [chip, setChip] = useState<string[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [starting, setStarting] = useState(false);
  const [observing, setObserving] = useState(false);
  const [mix, setMix] = useState<MixState | null>(null);
  const [note, setNote] = useState("");

  /* §10.2 — the seed pool */
  const [pool, setPool] = useState<CatalogTrack[]>([]);
  const [poolLoading, setPoolLoading] = useState(false);
  const [filter, setFilter] = useState("");

  const [player, setPlayer] = useState<PlayerState>({ key: null, status: "paused" });
  useEffect(() => subscribePlayer(setPlayer), []);

  const logId = useRef(0);
  const fadeRef = useRef<HTMLDivElement>(null);
  const mixRef = useRef<MixState | null>(null);
  mixRef.current = mix;

  useEffect(() => {
    getEmotions().then(setLabels);
  }, []);
  useEffect(() => stop, []); // leaving the page stops the audio

  const pushLog = (entry: Omit<LogEntry, "id">) => {
    setLog((prev) => [{ id: ++logId.current, ...entry }, ...prev].slice(0, 12));
  };

  /* ── seed pool: 50 on mount, shuffle accumulates (§10.2) ── */
  const loadPool = async () => {
    setPoolLoading(true);
    try {
      const batch = await getTracks();
      setPool((prev) => {
        const seen = new Map(prev.map((t) => [t.track_id, t]));
        for (const t of batch) if (!seen.has(t.track_id)) seen.set(t.track_id, t);
        return [...seen.values()];
      });
    } catch {
      setNote("couldn't sample the library. try the shuffle again");
    } finally {
      setPoolLoading(false);
    }
  };

  useEffect(() => {
    if (!sessionId && !handoff) void loadPool();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const filteredPool = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return pool;
    return pool.filter(
      (t) =>
        t.name.toLowerCase().includes(q) || t.artist.toLowerCase().includes(q),
    );
  }, [pool, filter]);

  /* ── step 1: seed → /live/start, then make it audible (§9.3.1) ── */
  const startSession = async (seed: Seed) => {
    unlock(); // the seed click is the gesture that permits the crossfade
    setStarting(true);
    setNote("");
    try {
      const res = await liveStart({ track_id: seed.track_id });
      setSessionId(res.session_id);
      setNowPlaying({ ...seed, previewUrl: null, resolving: true });
      setChip([]);
      setLog([]);
      setMix(null);
      setArtwork(null);
      pushLog({
        kind: "info",
        headline: "session started. press play, then tell it how you feel to set the baseline",
        meta: `NOW PLAYING · ${seed.title.toUpperCase()}`,
      });
      resolveTrack(seed.title, seed.artist, 900_000).then((r) => {
        setNowPlaying((p) =>
          p && p.track_id === seed.track_id
            ? { ...p, previewUrl: r?.previewUrl ?? null, resolving: false }
            : p,
        );
        setArtwork(r?.artworkUrl ?? null);
      });
    } catch (err) {
      setNote(
        err instanceof ApiError && err.status !== 0
          ? err.message
          : "the engine didn't answer. try again",
      );
    } finally {
      setStarting(false);
    }
  };

  /* Curate / Journey hand-off starts immediately. */
  useEffect(() => {
    if (handoff && !sessionId && !starting) void startSession(handoff);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── commit the transition ─────────────────────────────── */
  const commitNext = (m: MixState) => {
    const title = nextField(m.next, "title") ?? nextField(m.next, "name") ?? "the next track";
    const artist = nextField(m.next, "artist") ?? "";
    setNowPlaying({
      track_id: m.plan.in_track_id,
      title,
      artist,
      camelot: m.next.camelot,
      bpm: m.next.bpm,
      previewUrl: null,
      resolving: false,
    });
    setArtwork(null);
    if (artist) {
      resolveTrack(title, artist, 900_000).then((r) => {
        setNowPlaying((p) =>
          p && p.track_id === m.plan.in_track_id
            ? { ...p, previewUrl: r?.previewUrl ?? null }
            : p,
        );
        setArtwork(r?.artworkUrl ?? null);
      });
    }
    setMix(null);
  };

  /* ── step 2: observe → maybe a real crossfade (§9.3) ───── */
  const observe = async (input: { emotion?: string; distribution?: number[] }) => {
    if (!sessionId || observing || mix) return;
    const req: LiveObserveRequest = { session_id: sessionId };
    if (input.distribution && input.distribution.length === 15) {
      req.distribution = input.distribution;
    } else if (input.emotion) {
      req.emotion = input.emotion;
    } else {
      return;
    }

    setObserving(true);
    setNote("");
    try {
      const res = await liveObserve(req);
      const d = res.drift;
      if (!res.triggered) {
        pushLog({
          kind: d.from === d.to ? "baseline" : "info",
          headline:
            d.from === d.to
              ? `listening. holding steady on ${d.to}`
              : `noticed ${d.from} → ${d.to}, not enough to change the music yet`,
          meta: `${res.reason.toUpperCase()} · DRIFT ${d.distance.toFixed(2)}`,
        });
        return;
      }
      if (!res.next || !res.crossfade) return;

      const plan = res.crossfade;
      const next = res.next;
      pushLog({
        kind: "drift",
        headline: `mood moved ${d.from} → ${d.to}. mixing into ${next.camelot} at ${Math.round(next.bpm)} BPM`,
        meta: `${plan.curve.toUpperCase()} · ${plan.beats} BEATS · DRIFT ${d.distance.toFixed(2)}`,
      });

      const base: MixState = {
        phase: "buffering",
        plan,
        next,
        reason: res.reason,
        durS: Math.min(Math.max(plan.duration_s, 2), 20),
        audio: false,
      };
      setMix(base);

      const visualOnly = () => {
        setMix({ ...base, phase: "fading", durS: Math.min(base.durS, 8), audio: false });
      };

      const title = nextField(next, "title") ?? nextField(next, "name");
      const artist = nextField(next, "artist");
      if (!title || !artist) {
        visualOnly();
        return;
      }
      const resolved = await resolveTrack(title, artist, 1_500_000);
      if (mixRef.current?.plan !== plan) return; // session moved on
      if (!resolved?.previewUrl) {
        visualOnly();
        return;
      }
      const ok = await crossfade(plan.in_track_id, resolved.previewUrl, plan.duration_s, {
        onStart: (actual) => {
          setMix((m) =>
            m && m.plan === plan ? { ...m, phase: "fading", durS: actual, audio: true } : m,
          );
        },
      });
      if (mixRef.current?.plan !== plan) return;
      if (ok) commitNext({ ...base, audio: true });
      else visualOnly();
    } catch (err) {
      setNote(
        err instanceof ApiError && err.status !== 0
          ? err.message
          : "the engine didn't answer. try again",
      );
      setMix(null);
    } finally {
      setObserving(false);
    }
  };

  /* ── the mixing visual, synced to the fade (§9.3) ──────── */
  useGSAP(
    () => {
      const root = fadeRef.current;
      const m = mix;
      if (!root || !m || m.phase !== "fading") return;
      const duration = reduced ? 0.01 : m.durS;
      const bar = root.querySelector("[data-fade-bar]");
      const outWave = root.querySelector("[data-wave-out]");
      const inWave = root.querySelector("[data-wave-in]");
      gsap.set(bar, { scaleX: 0, transformOrigin: "left center" });
      gsap.set(outWave, { opacity: 0.95, x: 0 });
      gsap.set(inWave, { opacity: 0.15, x: 0 });
      gsap
        .timeline({
          onComplete: () => {
            // Audio path commits when the fade promise resolves; this
            // completes only the visual-only fallback.
            if (!m.audio && mixRef.current === m) commitNext(m);
          },
        })
        .to(bar, { scaleX: 1, duration, ease: "none" }, 0)
        .to(outWave, { opacity: 0.12, x: -30, duration, ease: "sine.inOut" }, 0)
        .to(inWave, { opacity: 0.95, x: 30, duration, ease: "sine.inOut" }, 0);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    { scope: fadeRef, dependencies: [mix?.phase, mix?.durS, reduced] },
  );

  const inSession = sessionId !== null && nowPlaying !== null;
  const nextTitle = mix ? nextField(mix.next, "title") ?? nextField(mix.next, "name") : undefined;
  const nextArtist = mix ? nextField(mix.next, "artist") : undefined;
  const nextLinks = nextTitle && nextArtist ? deepLinks(nextTitle, nextArtist) : null;

  return (
    <>
      <PageHeader
        eyebrow="FEATURE 03 · THE PLAYER"
        title="Live"
        lede="Start from one track and keep feeling. When your mood moves, the mix moves with it, in key and on beat."
      />

      {/* §10.1 — the craft, said plainly, once */}
      <section className="px-6 pb-8 md:px-10">
        <p className="max-w-2xl text-sm leading-relaxed text-mist">
          This is the part that mixes in key and on beat, and that only works
          on songs the library has measured. So the seeds and the transitions
          come from the library, on purpose, for now. More may open up here
          later.
        </p>
      </section>

      {/* ── no session: three ways in ────────────────────── */}
      {!inSession && (
        <section className="px-6 pb-16 md:px-10">
          <div className="grid gap-6 md:grid-cols-[1.1fr_1fr]">
            {/* the hero path */}
            <div className="glass rounded-sm border-blue/40 p-6 [box-shadow:0_0_28px_rgba(46,107,255,0.12)] md:p-8">
              <ParenLabel accent>THE GOOD WAY IN</ParenLabel>
              <h2 className="display mt-4 text-2xl text-paper md:text-3xl">
                Bring a song that means something
              </h2>
              <p className="mt-4 max-w-md text-sm leading-relaxed text-mist">
                Curate a playlist, or plan a journey, then tap "live-mix this"
                on any library track. It lands here with a session ready and a
                reason to care about what happens next.
              </p>
              <div className="mt-6 flex flex-wrap gap-x-6 gap-y-2">
                <a
                  href="/curate"
                  className="mono-meta border-b border-paper/30 pb-0.5 text-paper/80 transition-colors hover:border-blue hover:text-paper"
                >
                  open curate →
                </a>
                <a
                  href="/journey"
                  className="mono-meta border-b border-paper/30 pb-0.5 text-paper/80 transition-colors hover:border-blue hover:text-paper"
                >
                  open journey →
                </a>
              </div>
            </div>

            {/* §10.2 — the picker */}
            <div className="glass rounded-sm p-6 md:p-8">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <ParenLabel accent>OR PICK FROM THE LIBRARY</ParenLabel>
                <span className="mono-meta text-paper/40">({pool.length} LOADED)</span>
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-3">
                <input
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder="filter what's loaded…"
                  aria-label="Filter the loaded tracks"
                  className="min-w-0 flex-1 border hairline bg-transparent px-3 py-2.5 text-sm text-paper placeholder:text-paper/25 focus:border-paper/35 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => void loadPool()}
                  disabled={poolLoading}
                  className="mono-meta border hairline px-4 py-2.5 text-paper/70 transition-colors hover:border-blue hover:text-paper disabled:opacity-40"
                >
                  {poolLoading ? "sampling…" : "shuffle in 50 more"}
                </button>
              </div>
              <ul data-lenis-prevent className="mt-4 max-h-96 divide-y divide-paper/8 overflow-y-auto overscroll-contain border-y hairline pr-1">
                {filteredPool.map((t) => (
                  <li key={t.track_id}>
                    <button
                      type="button"
                      disabled={starting}
                      onClick={() =>
                        void startSession({
                          track_id: t.track_id,
                          title: t.name,
                          artist: t.artist,
                          camelot: t.camelot,
                          bpm: t.bpm,
                        })
                      }
                      className="flex w-full items-baseline justify-between gap-4 px-1 py-3 text-left transition-colors hover:bg-paper/5 disabled:opacity-40"
                    >
                      <span className="min-w-0">
                        <span className="block truncate text-sm text-paper">{t.name}</span>
                        <span className="block truncate text-xs text-mist">{t.artist}</span>
                      </span>
                      <span className="mono-meta shrink-0 text-paper/40">
                        {t.camelot} · {Math.round(t.bpm)} BPM
                      </span>
                    </button>
                  </li>
                ))}
                {!poolLoading && filteredPool.length === 0 && (
                  <li className="px-1 py-4 text-sm text-mist">
                    nothing loaded matches that. shuffle in more and it might.
                  </li>
                )}
              </ul>
            </div>
          </div>
          {(starting || note) && (
            <div className="mt-6 flex items-center gap-4" aria-live="polite">
              {starting && <BarsLoader tone="blue" />}
              <span className="text-sm text-mist">
                {starting ? "starting the session…" : note}
              </span>
            </div>
          )}
        </section>
      )}

      {/* ── in session ───────────────────────────────────── */}
      {inSession && nowPlaying && (
        <section className="px-6 pb-16 md:px-10">
          <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
            {/* the player */}
            <div ref={fadeRef} className="glass-liquid relative overflow-hidden rounded-sm p-6 md:p-8">
              <div className="flex items-center justify-between">
                <ParenLabel accent>NOW PLAYING</ParenLabel>
                <span className="flex items-center gap-2">
                  <span
                    className={`h-1.5 w-1.5 bg-blue ${
                      player.status === "playing" ? "animate-pulse" : "opacity-40"
                    }`}
                  />
                  <span className="mono-meta text-paper/50">LIVE</span>
                </span>
              </div>

              <div className="mt-6 flex items-center gap-5">
                <div className="h-20 w-20 shrink-0 overflow-hidden border hairline bg-ink">
                  {artwork ? (
                    <img src={artwork} alt="" className="h-full w-full object-cover" />
                  ) : (
                    <div className="flex h-full w-full items-end justify-center gap-1 pb-3 opacity-40">
                      <span className="h-4 w-1 bg-silver" />
                      <span className="h-7 w-1 bg-blue" />
                      <span className="h-3 w-1 bg-silver" />
                      <span className="h-5 w-1 bg-gold" />
                    </div>
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <h2 className="truncate text-xl font-medium text-paper md:text-2xl">
                    {nowPlaying.title}
                  </h2>
                  <p className="truncate text-sm text-mist">{nowPlaying.artist}</p>
                  {(nowPlaying.camelot || nowPlaying.bpm) && (
                    <p className="mono-meta mt-2 text-paper/45">
                      ({[nowPlaying.camelot, nowPlaying.bpm ? `${Math.round(nowPlaying.bpm)} BPM` : null]
                        .filter(Boolean)
                        .join(" · ")})
                    </p>
                  )}
                </div>
                <PlayButton
                  trackKey={nowPlaying.track_id}
                  previewUrl={nowPlaying.previewUrl}
                  resolving={nowPlaying.resolving}
                  size="lg"
                />
              </div>
              {!nowPlaying.resolving && !nowPlaying.previewUrl && (
                <p className="mt-4 text-xs text-paper/45">
                  no preview for this one on iTunes. the session still works,
                  the mix will just be visual until a playable track lands.
                </p>
              )}

              {/* the transition */}
              {mix && (
                <div className="mt-8 border-t hairline pt-6" aria-live="polite">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <ParenLabel accent>
                      {mix.phase === "buffering" ? "MIXING · BUFFERING" : "MIXING"}
                    </ParenLabel>
                    <span className="mono-meta text-paper/50">
                      {mix.next.camelot} · {Math.round(mix.next.bpm)} BPM ·{" "}
                      {Math.round(mix.durS)}S
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-paper/45">{mix.reason}</p>
                  {mix.phase === "buffering" ? (
                    <div className="mt-4 flex items-center gap-3">
                      <BarsLoader tone="blue" />
                      <span className="mono-meta text-paper/50">(FETCHING THE NEXT PREVIEW…)</span>
                    </div>
                  ) : (
                    <>
                      <div className="relative mt-4 h-16 overflow-hidden">
                        <svg data-wave-out viewBox="0 0 320 60" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
                          <path d="M0,30 Q20,8 40,30 T80,30 T120,30 T160,30 T200,30 T240,30 T280,30 T320,30" fill="none" stroke="var(--color-silver)" strokeWidth="1.6" />
                        </svg>
                        <svg data-wave-in viewBox="0 0 320 60" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
                          <path d="M0,30 Q20,52 40,30 T80,30 T120,30 T160,30 T200,30 T240,30 T280,30 T320,30" fill="none" stroke="var(--color-blue)" strokeWidth="2" />
                        </svg>
                      </div>
                      <div className="mt-3 h-px w-full bg-paper/10">
                        <div data-fade-bar className="h-px w-full bg-blue" />
                      </div>
                      {!mix.audio && (
                        <p className="mt-3 text-xs text-paper/45">
                          no playable preview for the next one, so the mix is
                          visual this time.{" "}
                          {nextLinks && (
                            <a
                              href={nextLinks.youtube}
                              target="_blank"
                              rel="noreferrer"
                              className="border-b border-paper/30 text-paper/70 transition-colors hover:border-blue hover:text-paper"
                            >
                              open it on youtube ↗
                            </a>
                          )}
                        </p>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>

            {/* the ear */}
            <div className="glass-liquid rounded-sm p-6 md:p-8">
              <ParenLabel>TELL IT HOW YOU FEEL, ANY TIME</ParenLabel>
              <div className="mt-4">
                <EmotionChips
                  labels={labels}
                  selected={chip}
                  multi={false}
                  disabled={observing || mix !== null}
                  onChange={(next) => {
                    setChip(next);
                    if (next[0]) void observe({ emotion: next[0] });
                  }}
                />
              </div>
              <div className="mt-6">
                <VoiceMic
                  disabled={observing || mix !== null}
                  onResult={(r) => {
                    setChip([]);
                    void observe(
                      r.distribution?.length === 15
                        ? { distribution: r.distribution }
                        : { emotion: r.emotion },
                    );
                  }}
                />
              </div>
              {(observing || note) && (
                <div className="mt-5 flex items-center gap-3" aria-live="polite">
                  {observing && <BarsLoader tone="blue" />}
                  <span className="text-sm text-mist">
                    {observing ? "listening…" : note}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* the honest log */}
          {log.length > 0 && (
            <div className="mt-8">
              <ParenLabel>WHAT IT HEARD</ParenLabel>
              <ol className="mt-4 flex flex-col gap-3">
                {log.map((entry) => (
                  <li key={entry.id} className="flex flex-col gap-1 border-l pl-4 hairline">
                    <span
                      className={`text-sm ${
                        entry.kind === "drift" ? "text-paper" : "text-paper/75"
                      }`}
                    >
                      {entry.headline}
                    </span>
                    <span className="mono-meta text-paper/35">({entry.meta})</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </section>
      )}
    </>
  );
}
