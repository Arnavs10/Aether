/**
 * AETHER — Live, built in full (§9). A two-call stateful loop, never
 * collapsed: seed a track (Curate hand-off or the quick picker backed by the
 * now-working GET /tracks, whose field is `name`), then observe. The first
 * observe sets the baseline; a real drift returns next + crossfade, and the
 * transition is animated as an actual crossfade over its real duration.
 * The always-present `drift` object and `reason` string narrate the loop.
 */

import { useEffect, useRef, useState } from "react";
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
import { resolveTrack } from "../lib/itunes";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { PageHeader } from "../components/ui/PageScaffold";
import { ParenLabel } from "../components/ui/ParenLabel";
import { BarsLoader } from "../components/ui/BarsLoader";
import { EmotionChips } from "../components/features/EmotionChips";
import { VoiceMic } from "../components/features/VoiceMic";

interface Seed {
  track_id: string;
  title: string;
  artist: string;
  camelot?: string;
  bpm?: number;
}

interface LogEntry {
  id: number;
  headline: string;
  meta: string;
  kind: "info" | "baseline" | "drift";
}

interface Crossfading {
  plan: CrossfadePlan;
  next: LiveNextTrack;
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
  const [nowPlaying, setNowPlaying] = useState<Seed | null>(null);
  const [artwork, setArtwork] = useState<string | null>(null);
  const [chip, setChip] = useState<string[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [starting, setStarting] = useState(false);
  const [observing, setObserving] = useState(false);
  const [crossfading, setCrossfading] = useState<Crossfading | null>(null);
  const [note, setNote] = useState("");

  const [catalog, setCatalog] = useState<CatalogTrack[] | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerLoading, setPickerLoading] = useState(false);

  const logId = useRef(0);
  const fadeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getEmotions().then(setLabels);
  }, []);

  const pushLog = (entry: Omit<LogEntry, "id">) => {
    setLog((prev) => [{ id: ++logId.current, ...entry }, ...prev].slice(0, 12));
  };

  /* ── step 1: seed → /live/start ─────────────────────── */
  const startSession = async (seed: Seed) => {
    setStarting(true);
    setNote("");
    try {
      const res = await liveStart({ track_id: seed.track_id });
      setSessionId(res.session_id);
      setNowPlaying(seed);
      setChip([]);
      setLog([]);
      setArtwork(null);
      pushLog({
        kind: "info",
        headline: "session started. tell it how you feel to set the baseline",
        meta: `NOW PLAYING · ${seed.title.toUpperCase()}`,
      });
      resolveTrack(seed.title, seed.artist).then((r) =>
        setArtwork(r?.artworkUrl ?? null),
      );
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

  /* Curate hand-off: arriving with a seed starts immediately. */
  useEffect(() => {
    if (handoff && !sessionId && !starting) void startSession(handoff);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openPicker = async () => {
    setPickerOpen(true);
    if (catalog || pickerLoading) return;
    setPickerLoading(true);
    try {
      setCatalog(await getTracks());
    } catch {
      setNote("couldn't fetch seed tracks. try again");
      setPickerOpen(false);
    } finally {
      setPickerLoading(false);
    }
  };

  /* ── step 2: observe ────────────────────────────────── */
  const observe = async (input: { emotion?: string; distribution?: number[] }) => {
    if (!sessionId || observing || crossfading) return;
    // §2.7: omit distribution unless it is exactly 15 floats.
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
      } else if (res.next && res.crossfade) {
        pushLog({
          kind: "drift",
          headline: `mood moved ${d.from} → ${d.to}. mixing into ${res.next.camelot} at ${Math.round(res.next.bpm)} BPM`,
          meta: `${res.crossfade.curve.toUpperCase()} · ${res.crossfade.beats} BEATS · DRIFT ${d.distance.toFixed(2)}`,
        });
        setCrossfading({ plan: res.crossfade, next: res.next });
      }
    } catch (err) {
      setNote(
        err instanceof ApiError && err.status !== 0
          ? err.message
          : "the engine didn't answer. try again",
      );
    } finally {
      setObserving(false);
    }
  };

  /* ── the crossfade, animated over its real duration ──── */
  useGSAP(
    () => {
      const root = fadeRef.current;
      if (!root || !crossfading) return;
      const { plan, next } = crossfading;
      const duration = reduced ? 0.01 : Math.min(Math.max(plan.duration_s, 2), 20);

      const commit = () => {
        setNowPlaying({
          track_id: plan.in_track_id,
          title: nextField(next, "title") ?? nextField(next, "name") ?? "the next track",
          artist: nextField(next, "artist") ?? "",
          camelot: next.camelot,
          bpm: next.bpm,
        });
        setArtwork(null);
        const t = nextField(next, "title") ?? nextField(next, "name");
        const a = nextField(next, "artist");
        if (t && a) resolveTrack(t, a).then((r) => setArtwork(r?.artworkUrl ?? null));
        setCrossfading(null);
      };

      const bar = root.querySelector("[data-fade-bar]");
      const outWave = root.querySelector("[data-wave-out]");
      const inWave = root.querySelector("[data-wave-in]");
      gsap.set(bar, { scaleX: 0, transformOrigin: "left center" });
      gsap.set(outWave, { opacity: 0.95 });
      gsap.set(inWave, { opacity: 0.15 });
      gsap
        .timeline({ onComplete: commit })
        .to(bar, { scaleX: 1, duration, ease: "none" }, 0)
        .to(outWave, { opacity: 0.12, x: -30, duration, ease: "sine.inOut" }, 0)
        .to(inWave, { opacity: 0.95, x: 30, duration, ease: "sine.inOut" }, 0);
    },
    { scope: fadeRef, dependencies: [crossfading, reduced] },
  );

  const inSession = sessionId !== null && nowPlaying !== null;

  return (
    <>
      <PageHeader
        eyebrow="FEATURE 03 · THE PLAYER"
        title="Live"
        lede="Start from one track and keep feeling. When your mood moves, the mix moves with it."
      />

      {/* ── no session: two ways in ──────────────────────── */}
      {!inSession && (
        <section className="px-6 pb-16 md:px-10">
          <div className="grid gap-6 md:grid-cols-2">
            <div className="glass rounded-sm p-6 md:p-8">
              <ParenLabel accent>FROM CURATE</ParenLabel>
              <p className="mt-4 max-w-md text-sm leading-relaxed text-mist">
                The best seed is a song that already means something. Curate a
                playlist, then tap "live-mix this" on any track and it lands
                here with a session ready.
              </p>
              <a
                href="/curate"
                className="mono-meta mt-6 inline-block border-b border-paper/30 pb-0.5 text-paper/80 transition-colors hover:border-blue hover:text-paper"
              >
                open curate →
              </a>
            </div>

            <div className="glass rounded-sm p-6 md:p-8">
              <ParenLabel accent>QUICK SEED</ParenLabel>
              <p className="mt-4 max-w-md text-sm leading-relaxed text-mist">
                Or grab one of fifty tracks sampled straight from the library,
                key and tempo already known.
              </p>
              {!pickerOpen ? (
                <button
                  type="button"
                  onClick={() => void openPicker()}
                  disabled={starting}
                  className="mono-meta mt-6 border border-blue px-5 py-2.5 text-paper transition-all hover:[box-shadow:0_0_22px_rgba(46,107,255,0.35)] disabled:opacity-40"
                >
                  pick a starting track →
                </button>
              ) : pickerLoading ? (
                <div className="mt-6 flex items-center gap-3">
                  <BarsLoader tone="blue" />
                  <span className="mono-meta text-paper/50">(SAMPLING THE LIBRARY…)</span>
                </div>
              ) : (
                <ul className="mt-6 max-h-72 divide-y divide-paper/8 overflow-y-auto border-y hairline pr-1">
                  {(catalog ?? []).map((t) => (
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
                </ul>
              )}
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
            <div ref={fadeRef} className="glass relative overflow-hidden rounded-sm p-6 md:p-8">
              <div className="flex items-center justify-between">
                <ParenLabel accent>NOW PLAYING</ParenLabel>
                <span className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 animate-pulse bg-blue" />
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
                <div className="min-w-0">
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
              </div>

              {/* the transition, when it fires */}
              {crossfading && (
                <div className="mt-8 border-t hairline pt-6" aria-live="polite">
                  <div className="flex items-baseline justify-between">
                    <ParenLabel accent>MIXING</ParenLabel>
                    <span className="mono-meta text-paper/50">
                      {crossfading.next.camelot} · {Math.round(crossfading.next.bpm)} BPM ·{" "}
                      {crossfading.plan.duration_s}S
                    </span>
                  </div>
                  <div className="relative mt-4 h-16 overflow-hidden">
                    <svg data-wave-out viewBox="0 0 320 60" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
                      <path
                        d="M0,30 Q20,8 40,30 T80,30 T120,30 T160,30 T200,30 T240,30 T280,30 T320,30"
                        fill="none" stroke="var(--color-silver)" strokeWidth="1.6"
                      />
                    </svg>
                    <svg data-wave-in viewBox="0 0 320 60" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
                      <path
                        d="M0,30 Q20,52 40,30 T80,30 T120,30 T160,30 T200,30 T240,30 T280,30 T320,30"
                        fill="none" stroke="var(--color-blue)" strokeWidth="2"
                      />
                    </svg>
                  </div>
                  <div className="mt-3 h-px w-full bg-paper/10">
                    <div data-fade-bar className="h-px w-full bg-blue" />
                  </div>
                </div>
              )}
            </div>

            {/* the ear */}
            <div className="glass rounded-sm p-6 md:p-8">
              <ParenLabel>TELL IT HOW YOU FEEL, ANY TIME</ParenLabel>
              <div className="mt-4">
                <EmotionChips
                  labels={labels}
                  selected={chip}
                  multi={false}
                  disabled={observing || crossfading !== null}
                  onChange={(next) => {
                    setChip(next);
                    if (next[0]) void observe({ emotion: next[0] });
                  }}
                />
              </div>
              <div className="mt-6">
                <VoiceMic
                  disabled={observing || crossfading !== null}
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
