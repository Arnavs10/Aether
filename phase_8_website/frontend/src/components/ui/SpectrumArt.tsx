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

/* ── variant: wall (tall angular equalizer, §6.2: designed) ── */

interface WallBar {
  x: number;
  w: number;
  color: string;
  opacity: number;
  /** Holding bars stand still while the others move. */
  hold?: boolean;
  holdAt?: number;
}

const WALL_SPEC: WallBar[] = [
  { x: 8, w: 7, color: SILVER, opacity: 0.26 },
  { x: 22, w: 16, color: BLUE, opacity: 0.8 },
  { x: 46, w: 5, color: SILVER, opacity: 0.3, hold: true, holdAt: 0.62 },
  { x: 58, w: 12, color: SILVER, opacity: 0.28 },
  { x: 78, w: 22, color: SILVER, opacity: 0.22 },
  { x: 108, w: 6, color: GOLD, opacity: 0.8, hold: true, holdAt: 0.4 },
  { x: 122, w: 14, color: BLUE, opacity: 0.75 },
  { x: 144, w: 9, color: SILVER, opacity: 0.3 },
  { x: 160, w: 18, color: EMBER, opacity: 0.72 },
  { x: 186, w: 5, color: SILVER, opacity: 0.32, hold: true, holdAt: 0.8 },
  { x: 198, w: 12, color: SILVER, opacity: 0.26 },
];

function WallArt() {
  const ref = useRef<SVGSVGElement>(null);
  useGatedTimeline(
    ref,
    (tl, root) => {
      const movers = root.querySelectorAll("[data-wbar='move']");
      const holders = root.querySelectorAll("[data-wbar='hold']");
      const scan = root.querySelector("[data-wscan]");
      const holdSpecs = WALL_SPEC.filter((b) => b.hold);
      gsap.set([...movers, ...holders], { transformOrigin: "50% 100%" });
      gsap.set(holders, { scaleY: (i: number) => holdSpecs[i]?.holdAt ?? 0.5 });
      gsap.set(movers, { scaleY: 0.3 });
      tl.to(
        movers,
        {
          scaleY: () => gsap.utils.random(0.16, 1),
          duration: 0.7,
          ease: "steps(3)",
          stagger: { each: 0.07, from: "random" },
          repeat: 3,
          repeatRefresh: true,
        },
        0,
      );
      if (scan) {
        tl.fromTo(scan, { y: -50 }, { y: 360, duration: 2.6, ease: "none" }, 0.3);
      }
      tl.to({}, { duration: 0.5 });
    },
    (root) => {
      const bars = root.querySelectorAll("[data-wbar]");
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
      {WALL_SPEC.map((b, i) => (
        <g key={i} data-wbar={b.hold ? "hold" : "move"}>
          <rect x={b.x} y={14} width={b.w} height={292} fill={b.color} opacity={b.opacity} />
          <rect x={b.x} y={14} width={b.w} height={4} fill={b.color} opacity={Math.min(1, b.opacity + 0.3)} />
        </g>
      ))}
      <rect data-wscan x={0} y={-50} width={220} height={10} fill="url(#wall-scan)" opacity={0.5} />
      <defs>
        <linearGradient id="wall-scan" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#2e6bff" stopOpacity="0" />
          <stop offset="0.5" stopColor="#2e6bff" stopOpacity="0.6" />
          <stop offset="1" stopColor="#2e6bff" stopOpacity="0" />
        </linearGradient>
      </defs>
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
  ease: string;
  /** How motion distributes: random scatter, left-to-right, mirrored, one-at-a-time. */
  order: "random" | "seq" | "center" | "solo";
  /** Bar width as a fraction of the slot (dense vs sparse). */
  widthK: number;
  /** Targets climb with index (hopeful). */
  riser?: boolean;
  /** Hard ceiling on targets (frustrated: pushing, blocked). */
  clip?: number;
  /** Gentle vertical drift off the baseline (dreamy). */
  floatY?: boolean;
}

/** Fifteen distinct shapes: the SHAPE is the emotion, not the numbers (§4E.2). */
export const EMOTION_PROFILES: Record<EmotionLabel, EmotionProfile> = {
  happy: { bars: 16, speed: 0.5, base: 0.45, amp: 0.42, ease: "steps(2)", order: "random", widthK: 0.55 },
  sad: { bars: 7, speed: 1.8, base: 0.12, amp: 0.18, ease: "sine.inOut", order: "random", widthK: 0.4 },
  angry: { bars: 24, speed: 0.22, base: 0.55, amp: 0.45, ease: "steps(1)", order: "random", widthK: 0.85 },
  calm: { bars: 7, speed: 2.4, base: 0.3, amp: 0.1, ease: "sine.inOut", order: "seq", widthK: 0.6 },
  anxious: { bars: 18, speed: 0.38, base: 0.25, amp: 0.4, ease: "steps(2)", order: "random", widthK: 0.5 },
  energetic: { bars: 26, speed: 0.2, base: 0.55, amp: 0.45, ease: "steps(1)", order: "random", widthK: 0.7 },
  focused: { bars: 9, speed: 1.3, base: 0.55, amp: 0.05, ease: "steps(1)", order: "seq", widthK: 0.35 },
  nostalgic: { bars: 10, speed: 1.5, base: 0.35, amp: 0.22, ease: "sine.inOut", order: "random", widthK: 0.55 },
  romantic: { bars: 8, speed: 1.4, base: 0.38, amp: 0.22, ease: "sine.inOut", order: "center", widthK: 0.6 },
  melancholic: { bars: 5, speed: 2.6, base: 0.12, amp: 0.3, ease: "sine.inOut", order: "seq", widthK: 0.35 },
  confident: { bars: 12, speed: 0.8, base: 0.6, amp: 0.08, ease: "steps(1)", order: "seq", widthK: 0.65 },
  hopeful: { bars: 12, speed: 0.6, base: 0.28, amp: 0.5, ease: "power1.inOut", order: "seq", widthK: 0.55, riser: true },
  frustrated: { bars: 16, speed: 0.35, base: 0.5, amp: 0.4, ease: "steps(1)", order: "random", widthK: 0.8, clip: 0.88 },
  lonely: { bars: 5, speed: 2.2, base: 0.1, amp: 0.3, ease: "sine.inOut", order: "solo", widthK: 0.3 },
  dreamy: { bars: 13, speed: 2.0, base: 0.3, amp: 0.32, ease: "sine.inOut", order: "random", widthK: 0.5, floatY: true },
};

/** One or two lines worth reading, per emotion (§4E.3). BPM is allowed;
 *  feature names and 0–1 values are not. The five quoted lines are exact. */
export const EMOTION_LINES: Record<EmotionLabel, string> = {
  happy: "bright, quick on its feet, and openly major. music that isn't hiding how it feels.",
  sad: "slow and sunken, mostly minor. it sits low and stays there, and that's the point.",
  angry: "hard, loud and dense, distortion welcome. the target that leaves no room to breathe.",
  calm: "barely moving on purpose. soft edges, low volume, nothing asking for your attention.",
  anxious: "restless and unresolved. it keeps building without landing, the way the feeling does.",
  energetic: "about 142 beats a minute and nearly nothing acoustic. the loudest target in the set.",
  focused: "mostly without words. the one target that does not care whether you can dance to it.",
  nostalgic: "warm and a little worn. acoustic textures, older shapes, gently uneven.",
  romantic: "slow and close. tender, rounded, written for exactly two people.",
  melancholic: "the slowest thing Aether looks for. around 62 beats a minute, almost no lift, mostly real strings in a quiet room.",
  confident: "firm and even, bass forward. it walks in like it owns the room, without rushing.",
  hopeful: "rising and openly bright. it keeps climbing even when it starts low.",
  frustrated: "fast and blocked. it pushes hard against something and doesn't get through.",
  lonely: "the most acoustic target there is, and the least danceable. one instrument, one room.",
  dreamy: "floating, wordless, and quietly positive. it drifts rather than lands.",
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
      const target = (i: number) => {
        let v = profile.riser
          ? profile.base + (i / profile.bars) * profile.amp * 0.85 + Math.random() * profile.amp * 0.25
          : profile.base + Math.random() * profile.amp;
        if (profile.clip) v = Math.min(v, profile.clip);
        return v;
      };
      const staggerFrom =
        profile.order === "center" ? "center" : profile.order === "random" ? "random" : "start";
      const each =
        profile.order === "solo" ? profile.speed * 0.9 : profile.speed / profile.bars;
      tl.to(bars, {
        scaleY: (i: number) => target(i),
        duration: profile.speed,
        ease: profile.ease,
        stagger: { each, from: staggerFrom as "center" | "random" | "start" },
        repeat: -1,
        repeatRefresh: true,
      });
      if (profile.floatY) {
        tl.to(
          bars,
          {
            y: () => gsap.utils.random(-5, 5),
            duration: profile.speed * 1.4,
            ease: "sine.inOut",
            stagger: { each: each / 2, from: "random" },
            repeat: -1,
            repeatRefresh: true,
          },
          0,
        );
      }
    },
    (root) => {
      const bars = root.querySelectorAll("[data-bar]");
      gsap.set(bars, {
        transformOrigin: "50% 100%",
        scaleY: (i: number) => {
          let v = profile.riser
            ? profile.base + (i / profile.bars) * profile.amp * 0.85
            : profile.base + (((i * 7) % 10) / 10) * profile.amp;
          if (profile.clip) v = Math.min(v, profile.clip);
          return v;
        },
      });
    },
    [emotion],
  );

  const gap = 300 / profile.bars;
  const width = Math.max(3, Math.min(20, gap * profile.widthK));
  const emberIdx =
    emotion === "angry" || emotion === "frustrated" ? Math.floor(profile.bars / 2) : -1;
  const goldIdx = emotion === "hopeful" ? profile.bars - 1 : -1;

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
          fill={
            i === emberIdx
              ? EMBER
              : i === goldIdx
                ? GOLD
                : i % 3 === 1
                  ? BLUE
                  : SILVER
          }
          opacity={i === emberIdx || i === goldIdx ? 0.85 : i % 3 === 1 ? 0.85 : 0.4}
        />
      ))}
      <line x1="8" y1="148" x2="312" y2="148" stroke={SILVER} strokeOpacity="0.25" />
    </svg>
  );
}

/* ── public: the fifteen-node constellation (§4G) ────────── */

const NODES: Array<[number, number]> = [
  [58, 44], [140, 26], [226, 52], [286, 96], [252, 168],
  [292, 238], [214, 282], [128, 296], [52, 262], [26, 186],
  [40, 112], [116, 92], [186, 122], [156, 202], [86, 176],
];

const EDGES: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
  [8, 9], [9, 10], [10, 0], [11, 0], [11, 12], [12, 4], [13, 12],
  [13, 7], [14, 13], [14, 9], [11, 14], [2, 12],
];

/** Pulses cycle through these edges: a feeling finding its neighbours. */
const PULSE_SEQUENCE = [3, 13, 8, 18, 11, 15, 1, 16];

export function ConstellationArt({ className = "" }: { className?: string }) {
  const ref = useRef<SVGSVGElement>(null);

  useGatedTimeline(
    ref,
    (tl, root) => {
      const nodes = root.querySelectorAll("[data-node]");
      const edges = root.querySelectorAll("[data-edge]");
      const runner = root.querySelector("[data-runner]");
      gsap.set(nodes, { transformOrigin: "50% 50%" });
      gsap.set(runner, { opacity: 0 });

      // Slow breath across the whole figure.
      tl.to(
        nodes,
        {
          scale: 1.18,
          opacity: 0.95,
          duration: 3.2,
          ease: "sine.inOut",
          stagger: { each: 0.12, from: "random" },
          yoyo: true,
          repeat: 1,
        },
        0,
      );

      // A path lights between two nodes and travels, then fades.
      PULSE_SEQUENCE.forEach((edgeIdx, k) => {
        const at = 0.6 + k * 1.35;
        const edge = edges[edgeIdx];
        if (!edge) return;
        const [a, b] = EDGES[edgeIdx];
        tl.to(edge, { opacity: 0.9, duration: 0.18, ease: "steps(1)" }, at)
          .set(runner, { x: NODES[a][0], y: NODES[a][1], opacity: 1 }, at)
          .to(
            runner,
            { x: NODES[b][0], y: NODES[b][1], duration: 0.7, ease: "power1.inOut" },
            at + 0.05,
          )
          .to(runner, { opacity: 0, duration: 0.2 }, at + 0.78)
          .to(edge, { opacity: 0.16, duration: 0.6, ease: "sine.out" }, at + 0.5);
      });
      tl.to({}, { duration: 0.8 }); // breathing room before the loop
    },
    (root) => {
      gsap.set(root.querySelectorAll("[data-edge]"), { opacity: 0.2 });
    },
  );

  return (
    <div className={className}>
      <svg ref={ref} viewBox="0 0 320 320" className="h-full w-full" aria-hidden="true">
        {EDGES.map(([a, b], i) => (
          <line
            key={i}
            data-edge
            x1={NODES[a][0]}
            y1={NODES[a][1]}
            x2={NODES[b][0]}
            y2={NODES[b][1]}
            stroke={BLUE}
            strokeWidth="1"
            opacity="0.16"
          />
        ))}
        {NODES.map(([x, y], i) => (
          <rect
            key={i}
            data-node
            x={x - 3.5}
            y={y - 3.5}
            width={7}
            height={7}
            fill={i === 4 ? EMBER : i === 11 ? GOLD : i % 3 === 0 ? BLUE : SILVER}
            opacity={i === 4 || i === 11 ? 0.9 : 0.6}
          />
        ))}
        <rect data-runner x={-3} y={-3} width={6} height={6} fill={BLUE} />
      </svg>
    </div>
  );
}
