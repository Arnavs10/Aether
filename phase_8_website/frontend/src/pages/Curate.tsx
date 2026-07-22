/**
 * AETHER — Curate (Pass 4 §8).
 * §8.1 the precedence is now told truthfully: one live status line says
 *      exactly what is driving the request, and the textarea visibly dims
 *      when chips take over (still editable, clearly not in play).
 * §8.2 the mic APPENDS to the text and leaves a dismissible pill holding
 *      the full voice reading; nothing typed is ever destroyed.
 * §8.4 one quiet line tells people what to actually do with the list.
 * §8.5 the box teaches the language feature by example.
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { gsap, useGSAP } from "../lib/gsap";
import {
  ApiError,
  curate,
  distributionFor,
  getEmotions,
  postFeelingFeed,
} from "../lib/api";
import type { CurateRequest, CurateResponse, Track } from "../lib/types";
import { AETHER_EMOTIONS } from "../config/site";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { PageHeader } from "../components/ui/PageScaffold";
import { ParenLabel } from "../components/ui/ParenLabel";
import { BarsLoader } from "../components/ui/BarsLoader";
import { unlock } from "../lib/audio";
import { EmotionChips } from "../components/features/EmotionChips";
import { VoiceMic } from "../components/features/VoiceMic";
import { TrackCard } from "../components/features/TrackCard";
import { DownloadActions } from "../components/features/DownloadActions";

type Phase = "idle" | "working" | "done" | "error";

/* §8.5 — rotate examples that quietly demonstrate the range. */
const PLACEHOLDERS = [
  "late night drive, a little nostalgic…",
  "some sad hindi songs",
  "punjabi for the gym",
  "kpop while i study",
  "something calm, mostly instrumental",
];

/** Persist this browser's own moments for the Home marquee (§4H). */
function rememberMoment(emotion: string, tracks: Track[]): void {
  try {
    const KEY = "aether.moments.v1";
    const prev = JSON.parse(localStorage.getItem(KEY) ?? "[]") as unknown[];
    const entry = {
      emotion,
      topTrackTitles: tracks.slice(0, 3).map((t) => t.title),
      at: Date.now(),
    };
    localStorage.setItem(KEY, JSON.stringify([entry, ...prev].slice(0, 20)));
  } catch {
    /* private mode: the marquee just shows quotes */
  }
}

export default function Curate() {
  const navigate = useNavigate();
  const reduced = usePrefersReducedMotion();

  const [labels, setLabels] = useState<string[]>([...AETHER_EMOTIONS]);
  const [selected, setSelected] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [focused, setFocused] = useState(false);
  const [phIdx, setPhIdx] = useState(0);
  const [voicePill, setVoicePill] = useState<{
    emotion: string;
    distribution: number[];
    /** §6.2 (Pass 7): tone is an explicit opt-in, not an automatic winner. */
    useTone: boolean;
  } | null>(null);
  const [length, setLength] = useState(12);
  const [explain, setExplain] = useState(true);

  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<CurateResponse | null>(null);
  const [errMsg, setErrMsg] = useState("");
  const lastReq = useRef<CurateRequest | null>(null);
  const resultRef = useRef<HTMLElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    getEmotions().then(setLabels);
  }, []);

  // §8.5 rotating placeholder, only while empty and unfocused.
  useEffect(() => {
    if (text || focused) return;
    const id = setInterval(
      () => setPhIdx((i) => (i + 1) % PLACEHOLDERS.length),
      3_600,
    );
    return () => clearInterval(id);
  }, [text, focused]);

  /* §6 (Pass 7) — the send rule, in order: chips → TEXT → tone.
     An explicit request beats an inferred mood. Tone wins only when the
     user opts in via the pill, or when the transcript is too short to
     mean anything (three words or fewer), where tone is the only signal. */
  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
  const toneAuto = voicePill !== null && wordCount <= 3;
  const toneOn = voicePill !== null && (voicePill.useTone || toneAuto);

  const buildRequest = (): CurateRequest | null => {
    const base = { length, explain };
    if (selected.length === 1) return { ...base, emotion: selected[0] };
    if (selected.length > 1) {
      const dist = distributionFor(selected, labels);
      if (dist) return { ...base, distribution: dist };
    }
    if (toneOn && voicePill) return { ...base, distribution: voicePill.distribution };
    if (text.trim()) return { ...base, text: text.trim() };
    return null;
  };

  const driving: "chips" | "voice" | "text" | null =
    selected.length > 0
      ? "chips"
      : toneOn
        ? "voice"
        : text.trim()
          ? "text"
          : null;

  const canSubmit = phase !== "working" && driving !== null;

  const run = async (req: CurateRequest) => {
    lastReq.current = req;
    setPhase("working");
    setErrMsg("");
    try {
      const res = await curate(req);
      setResult(res);
      setPhase("done");
      postFeelingFeed(res.mood).catch(() => undefined);
      rememberMoment(res.mood, res.tracks);
    } catch (err) {
      setPhase("error");
      setErrMsg(
        err instanceof ApiError && err.status !== 0
          ? err.message
          : "the engine didn't answer. try again",
      );
    }
  };

  const submit = () => {
    const req = buildRequest();
    if (!req) return;
    unlock(); // the curate click primes previews (§2 pattern)
    void run(req);
  };

  /* Results header: squeeze-and-settle entrance. */
  useGSAP(
    () => {
      const root = resultRef.current;
      if (!root || phase !== "done") return;
      const mood = root.querySelector("[data-mood]");
      const metas = root.querySelectorAll("[data-meta]");
      if (reduced) {
        gsap.set([mood, ...metas], { autoAlpha: 1 });
        return;
      }
      gsap.set(mood, { transformOrigin: "left center" });
      gsap
        .timeline()
        .fromTo(
          mood,
          { scaleX: 1.28, autoAlpha: 0, letterSpacing: "0.14em" },
          { scaleX: 1, autoAlpha: 1, letterSpacing: "-0.015em", duration: 0.9, ease: "expo.out" },
        )
        .fromTo(
          metas,
          { y: 14, autoAlpha: 0 },
          { y: 0, autoAlpha: 1, duration: 0.5, ease: "power3.out", stagger: 0.08 },
          "-=0.45",
        );
    },
    { scope: resultRef, dependencies: [phase, result, reduced] },
  );

  const onLiveMix = (track: Track) => {
    navigate("/live", {
      state: { seed: { track_id: track.track_id, title: track.title, artist: track.artist } },
    });
  };

  return (
    <>
      <PageHeader
        eyebrow="FEATURE 01 · THE CORE"
        title="Curate"
        lede="Type how you feel, in English or Hindi, or just say it in English. The library answers with its reasoning attached."
      />

      {/* ── the controls ─────────────────────────────────── */}
      <section className="px-6 pb-10 md:px-10">
        <div className="glass-liquid rounded-sm p-6 md:p-8">
          <ParenLabel>PICK A FEELING, OR BLEND A FEW</ParenLabel>
          <div className="mt-4">
            <EmotionChips
              labels={labels}
              selected={selected}
              onChange={setSelected}
              disabled={phase === "working"}
            />
            {selected.length > 0 && (
              <p className="mt-2 text-xs text-paper/40">
                using your picked feelings.
              </p>
            )}
          </div>

          <div className="mt-8">
            <label htmlFor="curate-text" className="mono-meta text-paper/45">
              (OR DESCRIBE IT IN YOUR OWN WORDS)
            </label>
            <div className="relative mt-3">
              <textarea
                ref={textRef}
                id="curate-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onFocus={() => setFocused(true)}
                onBlur={() => setFocused(false)}
                disabled={phase === "working"}
                rows={3}
                className={`w-full resize-none border hairline bg-transparent p-4 text-base text-paper transition-opacity duration-500 focus:border-paper/35 focus:outline-none disabled:opacity-50 ${
                  driving === "chips" ? "opacity-40" : "opacity-100"
                }`}
              />
              {/* §8.5 rotating placeholder, fading, never typing out */}
              {!text && !focused && (
                <span
                  key={phIdx}
                  aria-hidden="true"
                  className="fade-in pointer-events-none absolute left-4 top-4 text-base text-paper/25"
                >
                  {PLACEHOLDERS[phIdx]}
                </span>
              )}
            </div>
            <p className="mt-2 text-xs text-paper/40">
              ask for a language if you want one. hindi, punjabi, korean, whatever.
            </p>
            <p className="mt-1 text-xs text-paper/40">
              asking for a language leans the new picks that way. the library picks are chosen for feel, so a few may land in another language.
            </p>

            {/* §6.2 (Pass 7): the pill shows what the model detected and
                is a clear opt-in toggle — the words win unless the user
                chooses the tone, or barely spoke any words at all. */}
            {voicePill && (
              <div className="mt-3">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setVoicePill((p) => (p ? { ...p, useTone: !p.useTone } : p))
                    }
                    aria-pressed={toneOn}
                    className={`mono-meta border px-3 py-1.5 transition-all duration-200 ${
                      toneOn
                        ? "border-gold text-gold [box-shadow:0_0_16px_rgba(200,162,75,0.3)]"
                        : "border-gold/40 text-gold/70 hover:border-gold/70 hover:text-gold/90"
                    }`}
                  >
                    {toneOn
                      ? `USING: ${voicePill.emotion} · MY TONE`
                      : `HEARD: ${voicePill.emotion} · USE MY TONE`}
                  </button>
                  <button
                    type="button"
                    onClick={() => setVoicePill(null)}
                    aria-label="Dismiss the voice reading"
                    className="mono-meta px-1 text-gold/60 transition-colors hover:text-gold"
                  >
                    ×
                  </button>
                </div>
                {toneOn && (
                  <p className="mt-1.5 text-xs text-paper/40">
                    using how you sounded, not what you typed.
                  </p>
                )}
              </div>
            )}

            {/* §8.1 the live status line: one clear active path */}
            <p className="mono-meta mt-3 text-paper/45" aria-live="polite">
              {driving === "chips" &&
                "(YOUR CHIPS ARE DRIVING THIS. CLEAR THEM TO USE YOUR WORDS INSTEAD)"}
              {driving === "voice" &&
                (toneAuto && !voicePill?.useTone
                  ? "(NOT MANY WORDS TO GO ON, SO YOUR TONE IS DRIVING)"
                  : "(YOUR TONE IS DRIVING THIS. TAP THE PILL TO USE YOUR WORDS INSTEAD)")}
              {driving === "text" && "(YOUR WORDS ARE DRIVING THIS)"}
            </p>
          </div>

          <div className="mt-8 flex flex-wrap items-center gap-x-10 gap-y-6">
            <VoiceMic
              disabled={phase === "working"}
              onResult={(r) => {
                // §8.2: append, never destroy. Cursor lands at the end.
                setText((prev) => (prev ? `${prev} ${r.text}` : r.text));
                if (r.distribution?.length === 15) {
                  setVoicePill({
                    emotion: r.emotion,
                    distribution: r.distribution,
                    useTone: false,
                  });
                }
                requestAnimationFrame(() => {
                  const el = textRef.current;
                  if (el) {
                    el.focus();
                    el.setSelectionRange(el.value.length, el.value.length);
                  }
                });
              }}
            />

            <div className="flex items-center gap-4">
              <span className="mono-meta text-paper/45">(LENGTH)</span>
              <input
                type="range"
                min={8}
                max={20}
                value={length}
                onChange={(e) => setLength(Number(e.target.value))}
                disabled={phase === "working"}
                className="w-36 accent-[#2e6bff]"
                aria-label="Playlist length"
              />
              <span className="mono-meta w-8 text-paper/70">{length}</span>
            </div>

            <button
              type="button"
              role="switch"
              aria-checked={explain}
              onClick={() => setExplain((v) => !v)}
              disabled={phase === "working"}
              className="mono-meta flex items-center gap-3 text-paper/60 transition-colors hover:text-paper"
            >
              <span
                className={`flex h-5 w-9 items-center border px-0.5 transition-colors ${
                  explain ? "justify-end border-blue" : "justify-start hairline"
                }`}
              >
                <span className={`h-3 w-3 ${explain ? "bg-blue" : "bg-paper/40"}`} />
              </span>
              EXPLAIN EVERY PICK
            </button>

            <button
              type="button"
              onClick={submit}
              disabled={!canSubmit}
              className="mono-meta ml-auto border border-blue px-6 py-3 text-paper transition-all hover:[box-shadow:0_0_22px_rgba(46,107,255,0.35)] disabled:border-paper/20 disabled:text-paper/30 disabled:hover:[box-shadow:none]"
            >
              curate →
            </button>
          </div>
        </div>
      </section>

      {/* ── working / error ──────────────────────────────── */}
      {phase === "working" && (
        <section className="px-6 pb-10 md:px-10" aria-live="polite">
          <div className="glass flex items-center gap-5 rounded-sm p-6">
            <BarsLoader tone="blue" />
            <div>
              <p className="text-sm text-paper/80">matching against 1.2 million songs…</p>
              <p className="mono-meta mt-1 text-paper/40">(THIS CAN TAKE A MOMENT)</p>
            </div>
          </div>
        </section>
      )}
      {phase === "error" && (
        <section className="px-6 pb-10 md:px-10" aria-live="polite">
          <div className="glass flex flex-wrap items-center gap-4 rounded-sm p-6">
            <p className="text-sm text-mist">{errMsg}</p>
            <button
              type="button"
              onClick={() => lastReq.current && void run(lastReq.current)}
              className="mono-meta border-b border-paper/30 pb-0.5 text-paper/70 transition-colors hover:border-blue hover:text-paper"
            >
              try again →
            </button>
          </div>
        </section>
      )}

      {/* ── the results ──────────────────────────────────── */}
      {phase === "done" && result && (
        <section ref={resultRef} className="px-6 pb-16 md:px-10">
          <div className="border-t hairline pt-10">
            <h2 data-mood className="display text-5xl text-paper md:text-7xl">
              {result.mood}
            </h2>
            <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2">
              <span data-meta>
                <ParenLabel accent>{result.intensity_label.toUpperCase()}</ParenLabel>
              </span>
              <span data-meta>
                <ParenLabel>ARC · {result.arc_shape.toUpperCase()}</ParenLabel>
              </span>
              <span data-meta>
                <ParenLabel>{result.size} TRACKS</ParenLabel>
              </span>
            </div>
            {result.reason && (
              <p data-meta className="serif-accent mt-6 max-w-2xl text-lg text-paper/70">
                {result.reason}
              </p>
            )}
            <div data-meta className="mt-8 flex flex-col gap-3">
              <DownloadActions
                tracks={result.tracks}
                headerLabel={`${result.mood} (${result.intensity_label})`}
                payload={result}
                name={`aether_${result.mood}`}
              />
              {/* §8.4 what to do with the list */}
              <p className="max-w-xl text-xs leading-relaxed text-paper/40">
                open any track on Apple, Spotify or YouTube and save it there.
                or download the whole set as a file and import it into whatever
                you actually listen on.
              </p>
            </div>
          </div>

          <div className="mt-10 flex flex-col gap-4">
            {result.tracks.map((t) => (
              <TrackCard key={`${t.rank}-${t.track_id}`} track={t} onLiveMix={onLiveMix} />
            ))}
          </div>
        </section>
      )}
    </>
  );
}
