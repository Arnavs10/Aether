/**
 * AETHER — Curate, the main feature, built in full (§7).
 * Request building follows the verified server resolution order (§2.4):
 * one chip → {emotion} · a blend → 15-float {distribution} aligned to the
 * live GET /emotions order · voice → its distribution passed straight
 * through · otherwise free text. The results header lands with the same
 * squeeze-and-settle motion as the Home FEEL/HEARD moment (§4C).
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
import { EmotionChips } from "../components/features/EmotionChips";
import { VoiceMic } from "../components/features/VoiceMic";
import { TrackCard } from "../components/features/TrackCard";
import { DownloadActions } from "../components/features/DownloadActions";

type Phase = "idle" | "working" | "done" | "error";

export default function Curate() {
  const navigate = useNavigate();
  const reduced = usePrefersReducedMotion();

  const [labels, setLabels] = useState<string[]>([...AETHER_EMOTIONS]);
  const [selected, setSelected] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [voiceDist, setVoiceDist] = useState<number[] | null>(null);
  const [length, setLength] = useState(12);
  const [explain, setExplain] = useState(true);

  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<CurateResponse | null>(null);
  const [errMsg, setErrMsg] = useState("");
  const lastReq = useRef<CurateRequest | null>(null);
  const resultRef = useRef<HTMLElement>(null);

  useEffect(() => {
    getEmotions().then(setLabels);
  }, []);

  /* §2.4 resolution order: chips beat voice beat text. */
  const buildRequest = (): CurateRequest | null => {
    const base = { length, explain };
    if (selected.length === 1) return { ...base, emotion: selected[0] };
    if (selected.length > 1) {
      const dist = distributionFor(selected, labels);
      if (dist) return { ...base, distribution: dist };
    }
    if (voiceDist && voiceDist.length === 15)
      return { ...base, distribution: voiceDist };
    if (text.trim()) return { ...base, text: text.trim() };
    return null;
  };

  const canSubmit =
    phase !== "working" &&
    (selected.length > 0 || text.trim().length > 0 || voiceDist !== null);

  const run = async (req: CurateRequest) => {
    lastReq.current = req;
    setPhase("working");
    setErrMsg("");
    try {
      const res = await curate(req);
      setResult(res);
      setPhase("done");
      // Anonymized single word into the public feed; never blocks the UI.
      postFeelingFeed(res.mood).catch(() => undefined);
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
    if (req) void run(req);
  };

  /* Results header: squeeze-and-settle entrance (§4C motion, §7). */
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
        lede="Say how you feel, typed or spoken, in English or Hindi. The library answers with its reasoning attached."
      />

      {/* ── the controls ─────────────────────────────────── */}
      <section className="px-6 pb-10 md:px-10">
        <div className="glass rounded-sm p-6 md:p-8">
          <ParenLabel>PICK A FEELING, OR BLEND A FEW</ParenLabel>
          <div className="mt-4">
            <EmotionChips
              labels={labels}
              selected={selected}
              onChange={setSelected}
              disabled={phase === "working"}
            />
          </div>

          <div className="mt-8">
            <label htmlFor="curate-text" className="mono-meta text-paper/45">
              (OR DESCRIBE IT IN YOUR OWN WORDS)
            </label>
            <textarea
              id="curate-text"
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setVoiceDist(null); // typed intent replaces the voice reading
              }}
              disabled={phase === "working"}
              rows={3}
              placeholder="late night drive, a little nostalgic, don't want anything loud…"
              className="mt-3 w-full resize-none border hairline bg-transparent p-4 text-base text-paper placeholder:text-paper/25 focus:border-paper/35 focus:outline-none disabled:opacity-50"
            />
            {selected.length > 0 && (
              <p className="mono-meta mt-2 text-paper/35">
                (YOUR SELECTED FEELINGS TAKE THE LEAD, THE TEXT RIDES ALONG)
              </p>
            )}
            {voiceDist && selected.length === 0 && (
              <p className="mono-meta mt-2 text-gold/70">
                (USING THE FULL SHAPE OF YOUR VOICE READING)
              </p>
            )}
          </div>

          <div className="mt-8 flex flex-wrap items-center gap-x-10 gap-y-6">
            <VoiceMic
              disabled={phase === "working"}
              onResult={(r) => {
                setText(r.text);
                setVoiceDist(r.distribution?.length === 15 ? r.distribution : null);
                setSelected([]);
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
            <div data-meta className="mt-8">
              <DownloadActions
                tracks={result.tracks}
                headerLabel={`${result.mood} (${result.intensity_label})`}
                payload={result}
                name={`aether_${result.mood}`}
              />
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
