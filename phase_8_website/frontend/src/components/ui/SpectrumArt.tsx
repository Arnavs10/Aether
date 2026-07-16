/**
 * AETHER — the one reusable art component (§4F performance rule).
 * SVG + GSAP, no per-widget rAF loops: each piece builds a single paused
 * timeline, and an IntersectionObserver plays/pauses it with visibility.
 * Static composed frame under prefers-reduced-motion.
 *
 * Variants:
 *  curate   scattered spectrum reorganising into an ordered ladder
 *  journey  a route whose waypoints light in sequence, a runner travelling it
 *  live     two waveforms sliding across each other, crossfading
 *  wall     tall angular equalizer wall (footer / about panels)
 *
 * Plus <EmotionSpectrum/> (§4E): a hard-edged spectrum that characterises a
 * hovered emotion via a per-emotion shape profile.
 */

import { useEffect, useRef, useState, type RefObject } from "react";
import { gsap, useGSAP } from "../../lib/gsap";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";
import type { EmotionLabel } from "../../config/site";

const BLUE = "var(--color-blue)";
const SILVER = "var(--color-silver)";
const GOLD = "var(--color-gold)";
const EMBER = "var(--color-ember)";

/* ── shared: visibility gate ────────────────────────────── */

function useInView(ref: RefObject<Element | null>): boolean {
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([entry]) => {
      setInView(entry?.isIntersecting ?? false);
    });
    io.observe(el);
    return () => io.disconnect();
  }, [ref]);
  return inView;
}

function useGatedTimeline(
  ref: RefObject<SVGSVGElement | null>,
  build: (tl: gsap.core.Timeline, root: SVGSVGElement) => void,
  reducedSetup: (root: SVGSVGElement) => void,
  deps: unknown[] = [],
) {
  const reduced = usePrefersReducedMotion();
  const inView = useInView(ref);
  const tlRef = useRef<gsap.core.Timeline | null>(null);

  useGSAP(
    () => {
      const root = ref.current;
      if (!root) return;
      if (reduced) {
        reducedSetup(root);
        return;
      }
      const tl = gsap.timeline({ paused: true, repeat: -1 });
      build(tl, root);
      tlRef.current = tl;
      return () => {
        tlRef.current = null;
      };
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    { scope: ref, dependencies: [reduced, ...deps] },
  );

  useEffect(() => {
    const tl = tlRef.current;
    if (!tl) return;
    if (inView) tl.play();
    else tl.pause();
  }, [inView, reduced]);
}

/* ── variant: curate (scatter → ordered ladder) ─────────── */

const CURATE_SCATTER = [0.55, 0.2, 0.85, 0.4, 0.95, 0.3, 0.7, 0.15, 0.6, 0.9, 0.35, 0.75];
const CURATE_SORTED = [...CURATE_SCATTER].sort((a, b) => a - b);

function CurateArt() {
  const ref = useRef<SVGSVGElement>(null);
  useGatedTimeline(
    ref,
    (tl, root) => {
      const bars = root.querySelectorAll("[data-bar]");
      gsap.set(bars, {
        transformOrigin: "50% 100%",
        scaleY: (i: number) => CURATE_SCATTER[i] ?? 0.4,
      });
      tl.to(bars, {
        scaleY: (i: number) => CURATE_SORTED[i] ?? 0.4,
        duration: 1.1,
        ease: "power3.inOut",
        stagger: 0.05,
      })
        .to({}, { duration: 0.9 }) // hold the ordered ladder
        .to(bars, {
          scaleY: (i: number) => CURATE_SCATTER[i] ?? 0.4,
          duration: 0.9,
          ease: "power2.inOut",
          stagger: { each: 0.04, from: "random" },
        })
        .to({}, { duration: 0.5 });
    },
    (root) => {
      const bars = root.querySelectorAll("[data-bar]");
      gsap.set(bars, {
        transformOrigin: "50% 100%",
        scaleY: (i: number) => CURATE_SORTED[i] ?? 0.4,
      });
    },
  );

  return (
    <svg ref={ref} viewBox="0 0 320 160" className="h-full w-full" aria-hidden="true">
      {CURATE_SCATTER.map((_, i) => (
        <rect
          key={i}
          data-bar
          x={14 + i * 25}
          y={18}
          width={13}
          height={124}
          fill={i === 4 || i === 9 ? BLUE : i === 7 ? GOLD : SILVER}
          opacity={i === 4 || i === 9 ? 0.9 : 0.45}
        />
      ))}
      <line x1="8" y1="146" x2="312" y2="146" stroke={SILVER} strokeOpacity="0.25" />
    </svg>
  );
}

/* ── variant: journey (waypoints light in sequence) ─────── */

const ROUTE: Array<[number, number]> = [
  [24, 118],
  [92, 66],
  [160, 98],
  [228, 42],
  [296, 82],
];

function JourneyArt() {
  const ref = useRef<SVGSVGElement>(null);
  useGatedTimeline(
    ref,
    (tl, root) => {
      const path = root.querySelector("[data-route]") as SVGPathElement | null;
      const nodes = root.querySelectorAll("[data-node]");
      const runner = root.querySelector("[data-runner]");
      if (!path || !runner) return;
      const len = path.getTotalLength();
      gsap.set(path, { strokeDasharray: len, strokeDashoffset: len });
      gsap.set(nodes, { fill: "transparent", stroke: SILVER, strokeOpacity: 0.5 });
      gsap.set(runner, { x: ROUTE[0][0], y: ROUTE[0][1], opacity: 0 });

      tl.to(path, { strokeDashoffset: 0, duration: 1.4, ease: "power1.inOut" });
      nodes.forEach((node, i) => {
        tl.to(
          node,
          { fill: i === ROUTE.length - 1 ? GOLD : BLUE, strokeOpacity: 1, duration: 0.2, ease: "steps(1)" },
          0.25 + i * 0.32,
        );
      });
      tl.to(runner, { opacity: 1, duration: 0.15 }, 0.2).to(
        runner,
        {
          duration: 1.8,
          ease: "power1.inOut",
          motionPath: undefined, // no plugin: keyframes below
          keyframes: ROUTE.slice(1).map(([x, y]) => ({ x, y })),
        },
        0.3,
      );
      tl.to({}, { duration: 0.7 });
      tl.to([...nodes, runner], { opacity: 0.15, duration: 0.4 }).set(path, {
        strokeDashoffset: len,
      });
      tl.set(nodes, { fill: "transparent", strokeOpacity: 0.5, opacity: 1 });
      tl.set(runner, { x: ROUTE[0][0], y: ROUTE[0][1], opacity: 0 });
    },
    (root) => {
      const path = root.querySelector("[data-route]");
      const nodes = root.querySelectorAll("[data-node]");
      if (path) gsap.set(path, { strokeDashoffset: 0 });
      gsap.set(nodes, { fill: BLUE, stroke: BLUE });
    },
  );

  const d = `M ${ROUTE.map(([x, y]) => `${x},${y}`).join(" L ")}`;

  return (
    <svg ref={ref} viewBox="0 0 320 160" className="h-full w-full" aria-hidden="true">
      <path data-route d={d} fill="none" stroke={SILVER} strokeOpacity="0.6" strokeWidth="1.5" strokeDasharray="1" />
      {ROUTE.map(([x, y], i) => (
        <rect key={i} data-node x={x - 6} y={y - 6} width={12} height={12} strokeWidth="1.5" />
      ))}
      <rect data-runner x={-4} y={-4} width={8} height={8} fill={EMBER} />
    </svg>
  );
}

/* ── variant: live (two waveforms crossfading) ──────────── */

function wavePath(offsetY: number, amp: number, phase: number): string {
  const pts: string[] = [];
  for (let x = 0; x <= 320; x += 8) {
    const y = offsetY + Math.sin(x / 26 + phase) * amp;
    pts.push(`${x},${y.toFixed(1)}`);
  }
  return `M ${pts.join(" L ")}`;
}

function LiveArt() {
  const ref = useRef<SVGSVGElement>(null);
  useGatedTimeline(
    ref,
    (tl, root) => {
      const a = root.querySelector("[data-wave-a]");
      const b = root.querySelector("[data-wave-b]");
      if (!a || !b) return;
      gsap.set(a, { x: 0, opacity: 0.95 });
      gsap.set(b, { x: 0, opacity: 0.3 });
      tl.to(a, { x: -46, opacity: 0.3, duration: 2.3, ease: "sine.inOut" }, 0)
        .to(b, { x: 46, opacity: 0.95, duration: 2.3, ease: "sine.inOut" }, 0)
        .to(a, { x: 0, opacity: 0.95, duration: 2.3, ease: "sine.inOut" }, 2.3)
        .to(b, { x: 0, opacity: 0.3, duration: 2.3, ease: "sine.inOut" }, 2.3);
    },
    (root) => {
      gsap.set(root.querySelectorAll("path"), { opacity: 0.7 });
    },
  );

  return (
    <svg ref={ref} viewBox="0 0 320 160" className="h-full w-full" aria-hidden="true">
      <path data-wave-a d={wavePath(70, 26, 0)} fill="none" stroke={BLUE} strokeWidth="2" />
      <path data-wave-b d={wavePath(96, 22, 2.2)} fill="none" stroke={EMBER} strokeWidth="1.6" />
      <line x1="160" y1="14" x2="160" y2="146" stroke={SILVER} strokeOpacity="0.25" strokeDasharray="3 5" />
    </svg>
  );
}

/* ── variant: wall (tall angular equalizer) ─────────────── */

const WALL_BARS = 10;

function WallArt() {
  const ref = useRef<SVGSVGElement>(null);
  useGatedTimeline(
    ref,
    (tl, root) => {
      const bars = root.querySelectorAll("[data-bar]");
      gsap.set(bars, { transformOrigin: "50% 100%", scaleY: 0.3 });
      tl.to(bars, {
        scaleY: () => gsap.utils.random(0.18, 1),
        duration: 0.7,
        ease: "steps(3)",
        stagger: { each: 0.06, from: "random" },
        repeat: -1,
        repeatRefresh: true,
      });
    },
    (root) => {
      const bars = root.querySelectorAll("[data-bar]");
      gsap.set(bars, {
        transformOrigin: "50% 100%",
        scaleY: (i: number) => 0.25 + ((i * 13) % 10) / 14,
      });
    },
  );

  return (
    <svg
      ref={ref}
      viewBox="0 0 220 320"
      preserveAspectRatio="none"
      className="h-full w-full"
      aria-hidden="true"
    >
      {Array.from({ length: WALL_BARS }).map((_, i) => (
        <rect
          key={i}
          data-bar
          x={10 + i * 21}
          y={14}
          width={11}
          height={292}
          fill={i === 2 || i === 7 ? BLUE : i === 5 ? EMBER : SILVER}
          opacity={i === 2 || i === 7 ? 0.8 : i === 5 ? 0.75 : 0.28}
        />
      ))}
    </svg>
  );
}

/* ── public: SpectrumArt ────────────────────────────────── */

export type ArtVariant = "curate" | "journey" | "live" | "wall";

export function SpectrumArt({
  variant,
  className = "",
}: {
  variant: ArtVariant;
  className?: string;
}) {
  return (
    <div className={className}>
      {variant === "curate" && <CurateArt />}
      {variant === "journey" && <JourneyArt />}
      {variant === "live" && <LiveArt />}
      {variant === "wall" && <WallArt />}
    </div>
  );
}

/* ── public: EmotionSpectrum (§4E) ──────────────────────── */

interface EmotionProfile {
  bars: number;
  /** Seconds per step cycle: small = fast/urgent, large = slow/settled. */
  speed: number;
  base: number;
  amp: number;
}

/** Simple per-emotion shape profiles: how the feeling moves as bars. */
export const EMOTION_PROFILES: Record<EmotionLabel, EmotionProfile> = {
  happy: { bars: 14, speed: 0.55, base: 0.4, amp: 0.5 },
  sad: { bars: 8, speed: 1.7, base: 0.14, amp: 0.24 },
  angry: { bars: 18, speed: 0.3, base: 0.45, amp: 0.55 },
  calm: { bars: 7, speed: 2.2, base: 0.3, amp: 0.16 },
  anxious: { bars: 20, speed: 0.34, base: 0.22, amp: 0.42 },
  energetic: { bars: 22, speed: 0.26, base: 0.5, amp: 0.5 },
  focused: { bars: 10, speed: 1.2, base: 0.52, amp: 0.12 },
  nostalgic: { bars: 9, speed: 1.5, base: 0.3, amp: 0.3 },
  romantic: { bars: 8, speed: 1.35, base: 0.36, amp: 0.28 },
  melancholic: { bars: 6, speed: 2.0, base: 0.18, amp: 0.42 },
  confident: { bars: 12, speed: 0.7, base: 0.58, amp: 0.3 },
  hopeful: { bars: 11, speed: 0.85, base: 0.32, amp: 0.46 },
  frustrated: { bars: 16, speed: 0.38, base: 0.36, amp: 0.5 },
  lonely: { bars: 5, speed: 2.4, base: 0.14, amp: 0.28 },
  dreamy: { bars: 13, speed: 1.8, base: 0.26, amp: 0.4 },
};

export function EmotionSpectrum({
  emotion,
  className = "",
}: {
  emotion: EmotionLabel;
  className?: string;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const profile = EMOTION_PROFILES[emotion];

  useGatedTimeline(
    ref,
    (tl, root) => {
      const bars = root.querySelectorAll("[data-bar]");
      gsap.set(bars, { transformOrigin: "50% 100%", scaleY: profile.base });
      tl.to(bars, {
        scaleY: () => profile.base + Math.random() * profile.amp,
        duration: profile.speed,
        ease: "steps(2)",
        stagger: { each: profile.speed / profile.bars, from: "random" },
        repeat: -1,
        repeatRefresh: true,
      });
    },
    (root) => {
      const bars = root.querySelectorAll("[data-bar]");
      gsap.set(bars, {
        transformOrigin: "50% 100%",
        scaleY: (i: number) => profile.base + ((i * 7) % 10 / 10) * profile.amp,
      });
    },
    [emotion],
  );

  const gap = 300 / profile.bars;
  const width = Math.min(18, gap * 0.55);

  return (
    <svg
      ref={ref}
      key={emotion}
      viewBox="0 0 320 160"
      className={`h-full w-full ${className}`}
      aria-hidden="true"
    >
      {Array.from({ length: profile.bars }).map((_, i) => (
        <rect
          key={i}
          data-bar
          x={12 + i * gap}
          y={14}
          width={width}
          height={132}
          fill={i % 3 === 1 ? BLUE : SILVER}
          opacity={i % 3 === 1 ? 0.85 : 0.4}
        />
      ))}
      <line x1="8" y1="148" x2="312" y2="148" stroke={SILVER} strokeOpacity="0.25" />
    </svg>
  );
}
