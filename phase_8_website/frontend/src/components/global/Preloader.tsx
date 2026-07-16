/**
 * AETHER — functional preloader (§3.4 rebuild).
 * Boxy, linear, geometric: a hard-edged vertical bar spectrum stepping in a
 * mechanical rhythm, framed by a thin border that draws itself stroke by
 * stroke, crossed by a diagonal blue scan sweep. Hairlines in blue, silver
 * and gold. No circles.
 *
 * The working parts are unchanged: GET /health polled every ~2s (cold starts
 * stay honest on screen), a minimum show floor, and a dissolve into the hero.
 * If the engine stays quiet past 25s, a calm escape appears. No URLs, no
 * technical language, never an alarm.
 */

import { useEffect, useRef, useState } from "react";
import { gsap, useGSAP } from "../../lib/gsap";
import { getHealth } from "../../lib/api";
import { useAppState } from "../../state/AppState";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";

const MIN_SHOW_MS = 1_700;
const POLL_EVERY_MS = 2_000;
const ESCAPE_AFTER_MS = 25_000;
const EXIT_MS = 850;

const WAIT_COPY = ["warming the engine…", "waking the library…", "almost there…"];

const BAR_COUNT = 16;
const BAR_COLORS = (i: number): string => {
  if (i === 4 || i === 11) return "var(--color-blue)";
  if (i === 8) return "var(--color-gold)";
  return "var(--color-silver)";
};

export function Preloader() {
  const { setEngine, setEngineInfo, setAppReady } = useAppState();
  const reduced = usePrefersReducedMotion();

  const [copyIdx, setCopyIdx] = useState(0);
  const [tracksOnline, setTracksOnline] = useState<number | null>(null);
  const [showEscape, setShowEscape] = useState(false);
  const [exiting, setExiting] = useState(false);
  const [gone, setGone] = useState(false);

  const rootRef = useRef<HTMLDivElement>(null);
  const startedAt = useRef(Date.now());
  const settled = useRef(false);

  /** Dissolve out, then release the app (hero entrance keys off appReady). */
  const finish = (status: "ready" | "degraded") => {
    if (settled.current) return;
    settled.current = true;
    setEngine(status);
    setExiting(true);
    window.setTimeout(() => {
      setGone(true);
      setAppReady(true);
    }, EXIT_MS);
  };

  /* The stepped spectrum + self-drawing frame + diagonal scan. */
  useGSAP(
    () => {
      const root = rootRef.current;
      if (!root) return;
      const bars = root.querySelectorAll("[data-pl-bar]");
      const edges = {
        top: root.querySelector("[data-edge-top]"),
        right: root.querySelector("[data-edge-right]"),
        bottom: root.querySelector("[data-edge-bottom]"),
        left: root.querySelector("[data-edge-left]"),
      };
      const scan = root.querySelector("[data-scan]");

      gsap.set(bars, { transformOrigin: "50% 100%", scaleY: 0.2 });

      if (reduced) {
        gsap.set(bars, { scaleY: (i: number) => 0.3 + ((i * 7) % 10) / 18 });
        gsap.set(Object.values(edges), { scaleX: 1, scaleY: 1 });
        return;
      }

      // Frame draws itself stroke by stroke, then holds.
      gsap.set(edges.top, { scaleX: 0, transformOrigin: "left center" });
      gsap.set(edges.right, { scaleY: 0, transformOrigin: "center top" });
      gsap.set(edges.bottom, { scaleX: 0, transformOrigin: "right center" });
      gsap.set(edges.left, { scaleY: 0, transformOrigin: "center bottom" });
      gsap
        .timeline()
        .to(edges.top, { scaleX: 1, duration: 0.3, ease: "power1.in" })
        .to(edges.right, { scaleY: 1, duration: 0.22, ease: "none" })
        .to(edges.bottom, { scaleX: 1, duration: 0.3, ease: "none" })
        .to(edges.left, { scaleY: 1, duration: 0.22, ease: "power1.out" });

      // Bars step mechanically: hard steps, fresh heights each cycle.
      gsap.to(bars, {
        scaleY: () => gsap.utils.random(0.15, 1),
        duration: 0.55,
        ease: "steps(3)",
        stagger: { each: 0.045, from: "random" },
        repeat: -1,
        repeatRefresh: true,
      });

      // Diagonal blue scan sweep across the frame.
      if (scan) {
        gsap.fromTo(
          scan,
          { xPercent: -220 },
          {
            xPercent: 480,
            duration: 1.7,
            ease: "none",
            repeat: -1,
            repeatDelay: 0.55,
          },
        );
      }
    },
    { scope: rootRef, dependencies: [reduced] },
  );

  // Poll /health until it answers.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      if (cancelled || settled.current) return;
      try {
        const health = await getHealth(1_800);
        if (cancelled) return;
        setEngineInfo(health);
        setTracksOnline(health.tracks);
        const wait = Math.max(0, MIN_SHOW_MS - (Date.now() - startedAt.current));
        timer = setTimeout(() => finish("ready"), wait + 350);
      } catch {
        if (cancelled) return;
        timer = setTimeout(poll, POLL_EVERY_MS);
      }
    };
    poll();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cycle the wait copy.
  useEffect(() => {
    const id = setInterval(
      () => setCopyIdx((i) => (i + 1) % WAIT_COPY.length),
      2_200,
    );
    return () => clearInterval(id);
  }, []);

  // Offer the quiet escape after a long silent stretch.
  useEffect(() => {
    const id = setTimeout(() => {
      if (!settled.current) setShowEscape(true);
    }, ESCAPE_AFTER_MS);
    return () => clearTimeout(id);
  }, []);

  if (gone) return null;

  return (
    <div
      ref={rootRef}
      aria-live="polite"
      className={`fixed inset-0 z-[100] flex flex-col items-center justify-center bg-ink transition-all duration-700 ease-out ${
        exiting ? "pointer-events-none scale-[1.03] opacity-0" : "opacity-100"
      }`}
    >
      {/* the framed spectrum */}
      <div className="relative h-36 w-72 overflow-hidden">
        <span data-edge-top className="absolute left-0 top-0 h-px w-full bg-blue" />
        <span data-edge-right className="absolute right-0 top-0 h-full w-px bg-silver/70" />
        <span data-edge-bottom className="absolute bottom-0 left-0 h-px w-full bg-gold/80" />
        <span data-edge-left className="absolute left-0 top-0 h-full w-px bg-silver/70" />

        <div className="absolute inset-0 flex items-end justify-center gap-[7px] px-5 pb-4 pt-5">
          {Array.from({ length: BAR_COUNT }).map((_, i) => (
            <span
              key={i}
              data-pl-bar
              className="h-full w-[3px]"
              style={{ background: BAR_COLORS(i), opacity: 0.85 }}
            />
          ))}
        </div>

        <span
          data-scan
          aria-hidden="true"
          className="absolute inset-y-[-20%] left-0 w-16 -skew-x-[22deg]"
          style={{
            background:
              "linear-gradient(90deg, transparent, rgba(46,107,255,0.28), transparent)",
          }}
        />
      </div>

      {/* honest status line */}
      <p className="mono-meta mt-9 text-paper/60" key={copyIdx}>
        {tracksOnline !== null
          ? `(${tracksOnline.toLocaleString("en-IN")} TRACKS ONLINE)`
          : `(${WAIT_COPY[copyIdx]})`}
      </p>

      {/* quiet long-wait escape (§3.5): no URLs, no alarm */}
      {showEscape && tracksOnline === null && (
        <div className="mt-8 flex flex-col items-center gap-3 text-center">
          <p className="max-w-sm px-6 text-sm text-mist">
            The engine hasn't answered yet. It may still be waking up.
          </p>
          <button
            type="button"
            onClick={() => finish("degraded")}
            className="mono-meta border-b border-paper/30 pb-0.5 text-paper/70 transition-colors hover:border-blue hover:text-paper"
          >
            look around while it wakes →
          </button>
        </div>
      )}
    </div>
  );
}
