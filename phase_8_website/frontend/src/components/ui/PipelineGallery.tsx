/**
 * AETHER — the §4D sticky catalogue gallery. On desktop the section pins
 * and scroll drives it: the numbered index advances, the description
 * swaps, and a hard-edged visual for each of the five stages crossfades
 * in beside it, with a scrub catch-up that gives it the same spring-eased
 * settle as the FEEL/HEARD moment. Snaps to rest on whole stages.
 * On mobile, and under prefers-reduced-motion, it renders as a clean
 * stacked catalogue instead: nothing pins, everything reads.
 */

import { useRef } from "react";
import { gsap, useGSAP } from "../../lib/gsap";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";
import { ParenLabel } from "./ParenLabel";

export interface PipelineStep {
  numeral: string;
  title: string;
  body: string;
}

const BLUE = "var(--color-blue)";
const SILVER = "var(--color-silver)";
const GOLD = "var(--color-gold)";
const EMBER = "var(--color-ember)";

/* ── five hard-edged stage visuals, one per system ───────── */

function StageHear() {
  const bars = [26, 54, 38, 70, 46, 82, 58, 40, 66, 30];
  return (
    <svg viewBox="0 0 320 220" className="h-full w-full" aria-hidden="true">
      {bars.map((h, i) => (
        <rect key={i} x={34 + i * 26} y={130 - h} width={10} height={h} fill={i % 3 === 1 ? BLUE : SILVER} opacity={i % 3 === 1 ? 0.85 : 0.4} />
      ))}
      <line x1="24" y1="132" x2="296" y2="132" stroke={SILVER} strokeOpacity="0.3" />
      {/* the listening arc */}
      <path d="M 90 176 Q 160 216 230 176" fill="none" stroke={BLUE} strokeOpacity="0.7" strokeWidth="1.5" />
      <rect x="154" y="150" width="12" height="26" fill={SILVER} opacity="0.6" />
    </svg>
  );
}

function StageUnderstand() {
  return (
    <svg viewBox="0 0 320 220" className="h-full w-full" aria-hidden="true">
      {Array.from({ length: 15 }).map((_, i) => {
        const col = i % 5;
        const row = Math.floor(i / 5);
        const lit = i === 7;
        return (
          <rect
            key={i}
            x={62 + col * 42}
            y={40 + row * 48}
            width={16}
            height={16}
            fill={lit ? BLUE : SILVER}
            opacity={lit ? 0.95 : 0.3}
            style={lit ? { filter: "drop-shadow(0 0 8px rgba(46,107,255,0.8))" } : undefined}
          />
        );
      })}
      <rect x="48" y="26" width="230" height="160" fill="none" stroke={SILVER} strokeOpacity="0.25" />
    </svg>
  );
}

function StageMatch() {
  const pts: Array<[number, number]> = [
    [60, 60], [110, 40], [230, 52], [270, 100], [250, 170],
    [90, 180], [50, 130], [200, 140], [140, 70], [180, 44],
    [70, 92], [244, 66], [120, 168], [286, 148], [40, 172],
  ];
  const near = [8, 7, 10];
  return (
    <svg viewBox="0 0 320 220" className="h-full w-full" aria-hidden="true">
      {/* the target */}
      <line x1="160" y1="82" x2="160" y2="138" stroke={GOLD} strokeOpacity="0.8" />
      <line x1="132" y1="110" x2="188" y2="110" stroke={GOLD} strokeOpacity="0.8" />
      {near.map((idx) => (
        <line key={idx} x1={160} y1={110} x2={pts[idx][0]} y2={pts[idx][1]} stroke={BLUE} strokeOpacity="0.5" />
      ))}
      {pts.map(([x, y], i) => (
        <rect key={i} x={x - 4} y={y - 4} width={8} height={8} fill={near.includes(i) ? BLUE : SILVER} opacity={near.includes(i) ? 0.95 : 0.35} />
      ))}
    </svg>
  );
}

function StageArrange() {
  const arc = [30, 52, 78, 98, 108, 98, 74, 48];
  return (
    <svg viewBox="0 0 320 220" className="h-full w-full" aria-hidden="true">
      {arc.map((h, i) => (
        <rect key={i} x={44 + i * 30} y={166 - h} width={14} height={h} fill={i === 4 ? BLUE : SILVER} opacity={i === 4 ? 0.9 : 0.42} />
      ))}
      <path d="M 51 128 Q 160 26 279 132" fill="none" stroke={BLUE} strokeOpacity="0.6" strokeWidth="1.5" strokeDasharray="4 5" />
      <line x1="34" y1="168" x2="290" y2="168" stroke={SILVER} strokeOpacity="0.3" />
    </svg>
  );
}

function StageExplain() {
  return (
    <svg viewBox="0 0 320 220" className="h-full w-full" aria-hidden="true">
      <rect x="52" y="36" width="216" height="148" fill="none" stroke={SILVER} strokeOpacity="0.4" />
      <rect x="70" y="58" width="70" height="9" fill={SILVER} opacity="0.6" />
      <rect x="70" y="84" width="180" height="6" fill={SILVER} opacity="0.3" />
      <rect x="70" y="100" width="164" height="6" fill={SILVER} opacity="0.3" />
      <rect x="70" y="116" width="172" height="6" fill={BLUE} opacity="0.7" />
      <rect x="70" y="148" width="96" height="6" fill={GOLD} opacity="0.7" />
      <rect x="60" y="44" width="3" height="132" fill={EMBER} opacity="0.6" />
    </svg>
  );
}

const STAGE_ART = [StageHear, StageUnderstand, StageMatch, StageArrange, StageExplain];

/* ── the gallery ─────────────────────────────────────────── */

export function PipelineGallery({ steps }: { steps: readonly PipelineStep[] }) {
  const pinRef = useRef<HTMLDivElement>(null);
  const reduced = usePrefersReducedMotion();

  useGSAP(
    () => {
      const root = pinRef.current;
      if (!root || reduced) return;

      // §4: create the pin after the route has painted, not during mount.
      const mm = gsap.matchMedia();
      const boot = gsap.delayedCall(0.25, () =>
        mm.add("(min-width: 768px)", () => {
        const rows = gsap.utils.toArray<HTMLElement>("[data-pg-row]", root);
        const arts = gsap.utils.toArray<HTMLElement>("[data-pg-art]", root);
        const n = steps.length;

        gsap.set(rows, { opacity: 0.35, x: 0 });
        gsap.set(arts, { autoAlpha: 0, y: 20, scale: 0.98 });
        gsap.set([rows[0]], { opacity: 1, x: 8 });
        gsap.set([arts[0]], { autoAlpha: 1, y: 0, scale: 1 });

        const tl = gsap.timeline({
          defaults: { ease: "power3.out" },
          scrollTrigger: {
            trigger: root,
            start: "top top",
            end: `+=${(n - 1) * 85}%`,
            pin: true,
            scrub: 0.6,
            snap: { snapTo: 1 / (n - 1), duration: 0.35, ease: "power2.out" },
          },
        });

        for (let i = 1; i < n; i++) {
          tl.to(rows[i - 1], { opacity: 0.35, x: 0, duration: 0.4 }, i)
            .to(rows[i], { opacity: 1, x: 8, duration: 0.4 }, i)
            .to(arts[i - 1], { autoAlpha: 0, y: -20, scale: 0.98, duration: 0.4 }, i)
            .to(arts[i], { autoAlpha: 1, y: 0, scale: 1, duration: 0.5 }, i + 0.05);
        }
        }),
      );
      return () => {
        boot.kill();
        mm.revert();
      };
    },
    { scope: pinRef, dependencies: [reduced, steps.length] },
  );

  /* Mobile + reduced-motion: the honest stacked catalogue. */
  const stacked = (
    <div className={`flex flex-col ${reduced ? "" : "md:hidden"}`}>
      {steps.map((step, i) => {
        const Art = STAGE_ART[i] ?? StageHear;
        return (
          <div key={step.numeral} className="py-7">
            <div className="flex items-baseline gap-5">
              <span className="serif-accent text-3xl text-paper/30">{step.numeral}</span>
              <h3 className="display text-xl text-paper">{step.title}</h3>
            </div>
            <p className="mt-4 max-w-2xl text-sm leading-relaxed text-mist">{step.body}</p>
            <div className="mt-6 h-40 max-w-sm opacity-80">
              <Art />
            </div>
          </div>
        );
      })}
    </div>
  );

  if (reduced) return stacked;

  return (
    <>
      {stacked}
      {/* the pinned gallery, desktop only */}
      <div ref={pinRef} className="hidden md:block">
        <div className="grid h-screen grid-cols-[1.1fr_1fr] items-center gap-14 py-10">
          <div className="flex flex-col justify-center">
            {/* §5: heading + its short description together, smaller, no
                divider lines. The scrub still lights each point in turn. */}
            <div className="flex flex-col gap-6">
              {steps.map((step) => (
                <div key={step.numeral} data-pg-row className="max-w-xl">
                  <div className="flex items-baseline gap-5">
                    <span className="serif-accent w-12 shrink-0 text-3xl text-paper/40 md:text-4xl">
                      {step.numeral}
                    </span>
                    <h3 className="display text-xl text-paper md:text-2xl">
                      {step.title}
                    </h3>
                  </div>
                  <p className="mt-1.5 pl-[4.25rem] text-sm leading-relaxed text-mist">
                    {step.body}
                  </p>
                </div>
              ))}
            </div>
            <div className="mt-10">
              <ParenLabel>SCROLL TO WALK THE PIPELINE</ParenLabel>
            </div>
          </div>
          <div className="relative h-[26rem]">
            {STAGE_ART.map((Art, i) => (
              <div key={i} data-pg-art className="glass absolute inset-0 rounded-sm p-8">
                <Art />
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
