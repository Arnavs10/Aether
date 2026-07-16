/**
 * AETHER — Journey, built in full (§8).
 * The response is FLAT (§2.5): start, waypoints[], target, direction at the
 * top level; trace is an array of plain strings. The arc reveal uses the
 * site's signature settle motion; the trace is shown honestly, labelled in
 * human words with the raw token kept as the mono meta.
 */

import { useRef, useState } from "react";
import { useNavigate } from "react-router";
import { gsap, useGSAP } from "../lib/gsap";
import { ApiError, journey } from "../lib/api";
import type { JourneyRequest, JourneyResponse, Track } from "../lib/types";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { PageHeader } from "../components/ui/PageScaffold";
import { ParenLabel } from "../components/ui/ParenLabel";
import { BarsLoader } from "../components/ui/BarsLoader";
import { VoiceMic } from "../components/features/VoiceMic";
import { TrackCard } from "../components/features/TrackCard";
import { DownloadActions } from "../components/features/DownloadActions";

type Phase = "idle" | "working" | "done" | "error";

/** Human labels for the honest step trace. */
function humanizeStep(raw: string): { label: string; meta: string } {
  const base = raw.split("(")[0];
  const map: Record<string, string> = {
    perceive: "reading the request",
    plan: "planning the route",
    act: "choosing the songs",
    reflect: "checking the fit",
    explain: "writing the reasons",
  };
  return { label: map[base] ?? base, meta: raw };
}

export default function Journey() {
  const navigate = useNavigate();
  const reduced = usePrefersReducedMotion();

  const [text, setText] = useState("");
  const [length, setLength] = useState(12);
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<JourneyResponse | null>(null);
  const [errMsg, setErrMsg] = useState("");
  const lastReq = useRef<JourneyRequest | null>(null);
  const resultRef = useRef<HTMLElement>(null);

  const run = async (req: JourneyRequest) => {
    lastReq.current = req;
    setPhase("working");
    setErrMsg("");
    try {
      const res = await journey(req);
      setResult(res);
      setPhase("done");
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
    if (text.trim()) void run({ text: text.trim(), length });
  };

  /* Arc reveal: line draws stop to stop, nodes light in sequence. */
  useGSAP(
    () => {
      const root = resultRef.current;
      if (!root || phase !== "done") return;
      const nodes = root.querySelectorAll("[data-arc-node]");
      const lines = root.querySelectorAll("[data-arc-line]");
      const rest = root.querySelectorAll("[data-after-arc]");
      if (reduced) {
        gsap.set([...nodes, ...lines, ...rest], { autoAlpha: 1, scaleX: 1 });
        return;
      }
      gsap.set(lines, { scaleX: 0, transformOrigin: "left center" });
      gsap.set(nodes, { autoAlpha: 0, y: 10 });
      gsap.set(rest, { autoAlpha: 0, y: 16 });
      const tl = gsap.timeline();
      nodes.forEach((node, i) => {
        tl.to(node, { autoAlpha: 1, y: 0, duration: 0.4, ease: "power3.out" }, i * 0.34);
        if (lines[i]) {
          tl.to(lines[i], { scaleX: 1, duration: 0.3, ease: "power1.inOut" }, i * 0.34 + 0.18);
        }
      });
      tl.to(rest, { autoAlpha: 1, y: 0, duration: 0.6, ease: "power3.out", stagger: 0.07 }, ">-0.1");
    },
    { scope: resultRef, dependencies: [phase, result, reduced] },
  );

  const onLiveMix = (track: Track) => {
    navigate("/live", {
      state: { seed: { track_id: track.track_id, title: track.title, artist: track.artist } },
    });
  };

  const stops =
    result && result.waypoints.length > 0
      ? result.waypoints
      : result
        ? [result.start, result.target]
        : [];

  return (
    <>
      <PageHeader
        eyebrow="FEATURE 02 · THE ROUTE"
        title="Journey"
        lede="Say where you are and where you want to end up. Aether plans the route and shows its work."
      />

      {/* ── controls ─────────────────────────────────────── */}
      <section className="px-6 pb-10 md:px-10">
        <div className="glass rounded-sm p-6 md:p-8">
          <label htmlFor="journey-text" className="mono-meta text-paper/45">
            (WHERE ARE YOU, AND WHERE DO YOU WANT TO GO)
          </label>
          <textarea
            id="journey-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={phase === "working"}
            rows={3}
            placeholder="wind me down after a stressful day…"
            className="mt-3 w-full resize-none border hairline bg-transparent p-4 text-base text-paper placeholder:text-paper/25 focus:border-paper/35 focus:outline-none disabled:opacity-50"
          />
          <div className="mt-6 flex flex-wrap items-center gap-x-10 gap-y-6">
            <VoiceMic
              disabled={phase === "working"}
              onResult={(r) => setText(r.text || r.emotion)}
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
              onClick={submit}
              disabled={phase === "working" || !text.trim()}
              className="mono-meta ml-auto border border-blue px-6 py-3 text-paper transition-all hover:[box-shadow:0_0_22px_rgba(46,107,255,0.35)] disabled:border-paper/20 disabled:text-paper/30 disabled:hover:[box-shadow:none]"
            >
              plan the route →
            </button>
          </div>
        </div>
      </section>

      {phase === "working" && (
        <section className="px-6 pb-10 md:px-10" aria-live="polite">
          <div className="glass flex items-center gap-5 rounded-sm p-6">
            <BarsLoader tone="blue" />
            <div>
              <p className="text-sm text-paper/80">planning the emotional stops…</p>
              <p className="mono-meta mt-1 text-paper/40">(THIS ONE THINKS A LITTLE LONGER)</p>
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

      {/* ── results ──────────────────────────────────────── */}
      {phase === "done" && result && (
        <section ref={resultRef} className="px-6 pb-16 md:px-10">
          <div className="border-t hairline pt-10">
            <div className="flex flex-wrap items-baseline justify-between gap-4">
              <ParenLabel accent>THE ROUTE</ParenLabel>
              <ParenLabel>DIRECTION · {result.direction.toUpperCase()}</ParenLabel>
            </div>

            {/* the arc: start → waypoints → target */}
            <div className="mt-8 flex flex-wrap items-center gap-y-4">
              {stops.map((stop, i) => (
                <div key={`${stop}-${i}`} className="flex items-center">
                  <span
                    data-arc-node
                    className={`border px-4 py-2.5 font-mono text-xs uppercase tracking-[0.18em] ${
                      i === 0
                        ? "border-paper/40 text-paper"
                        : i === stops.length - 1
                          ? "border-gold text-gold [box-shadow:0_0_16px_rgba(200,162,75,0.25)]"
                          : "border-blue text-paper [box-shadow:0_0_16px_rgba(46,107,255,0.25)]"
                    }`}
                  >
                    {stop}
                  </span>
                  {i < stops.length - 1 && (
                    <span data-arc-line className="mx-2 h-px w-8 bg-paper/30 md:w-14" />
                  )}
                </div>
              ))}
            </div>

            <p data-after-arc className="serif-accent mt-8 max-w-2xl text-lg text-paper/70">
              {result.summary}
            </p>

            {/* the honest trace */}
            <div data-after-arc className="mt-10">
              <ParenLabel>HOW IT THOUGHT ABOUT IT</ParenLabel>
              <ol className="mt-4 flex flex-col gap-2">
                {result.trace.map((raw, i) => {
                  const step = humanizeStep(raw);
                  return (
                    <li key={`${raw}-${i}`} className="flex flex-wrap items-baseline gap-x-4">
                      <span className="mono-meta w-7 text-paper/35">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="text-sm text-paper/85">{step.label}</span>
                      <span className="mono-meta text-paper/35">({step.meta})</span>
                    </li>
                  );
                })}
              </ol>
            </div>

            <div data-after-arc className="mt-10">
              <DownloadActions
                tracks={result.tracks}
                headerLabel={`journey (${result.direction})`}
                payload={result}
                name={`aether_journey_${result.target}`}
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
