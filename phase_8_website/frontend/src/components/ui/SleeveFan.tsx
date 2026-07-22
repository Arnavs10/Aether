/**
 * AETHER — hero art (Pass 4 §4A): a loose fan of album sleeves, drifting.
 * Every sleeve is GENERATED abstract geometry from the palette. No real
 * covers, no bands, no logos, nothing recognisable. A vinyl edge slides
 * out from behind the front sleeve and eases back; the front sleeve
 * occasionally flips to reveal a second face. SVG + GSAP on the shared
 * ticker, IO-gated, static under prefers-reduced-motion. Drifting, not
 * animating: one long varied cycle, nothing that reads as a loop.
 */

import { useEffect, useRef, useState, type RefObject } from "react";
import { gsap, useGSAP } from "../../lib/gsap";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";

function useInView(ref: RefObject<Element | null>): boolean {
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setInView(e?.isIntersecting ?? false));
    io.observe(el);
    return () => io.disconnect();
  }, [ref]);
  return inView;
}

const BLUE = "var(--color-blue)";
const SILVER = "var(--color-silver)";
const GOLD = "var(--color-gold)";
const EMBER = "var(--color-ember)";
const INK = "var(--color-ink)";

export function SleeveFan({ className = "" }: { className?: string }) {
  const ref = useRef<SVGSVGElement>(null);
  const reduced = usePrefersReducedMotion();
  const inView = useInView(ref);
  const tlRef = useRef<gsap.core.Timeline | null>(null);

  useGSAP(
    () => {
      const root = ref.current;
      if (!root) return;
      const back = root.querySelector("[data-sleeve-back]");
      const mid = root.querySelector("[data-sleeve-mid]");
      const front = root.querySelector("[data-sleeve-front]");
      const faceA = root.querySelector("[data-face-a]");
      const faceB = root.querySelector("[data-face-b]");
      const vinyl = root.querySelector("[data-vinyl]");
      const arm = root.querySelector("[data-arm]");
      if (!back || !mid || !front || !vinyl) return;

      gsap.set([back, mid, front], { transformOrigin: "50% 50%" });
      gsap.set(vinyl, { transformOrigin: "50% 50%" });
      gsap.set(faceB, { opacity: 0 });

      if (reduced) {
        gsap.set(vinyl, { x: 26 });
        return;
      }

      const tl = gsap.timeline({ paused: true, repeat: -1 });
      // Slow independent drift on each sleeve: alive at rest.
      tl.to(back, { y: -7, rotation: -8.5, duration: 6.5, ease: "sine.inOut", yoyo: true, repeat: 1 }, 0)
        .to(mid, { y: 5, rotation: 4.2, duration: 7.5, ease: "sine.inOut", yoyo: true, repeat: 1 }, 0.4)
        .to(front, { y: -4, rotation: 10.2, duration: 8, ease: "sine.inOut", yoyo: true, repeat: 1 }, 0.9)
        // The vinyl slides a little way out, hangs, eases back.
        .to(vinyl, { x: 40, rotation: 26, duration: 3.4, ease: "power1.inOut" }, 1.2)
        .to(vinyl, { x: 0, rotation: 0, duration: 3.8, ease: "power2.inOut" }, 7.6)
        // The front sleeve flips once per cycle to its second face.
        .to(front, { scaleX: 0, duration: 0.55, ease: "power2.in" }, 9.6)
        .set(faceA, { opacity: 0 }, 10.15)
        .set(faceB, { opacity: 1 }, 10.15)
        .to(front, { scaleX: 1, duration: 0.55, ease: "power2.out" }, 10.15)
        .to(front, { scaleX: 0, duration: 0.55, ease: "power2.in" }, 13.1)
        .set(faceB, { opacity: 0 }, 13.65)
        .set(faceA, { opacity: 1 }, 13.65)
        .to(front, { scaleX: 1, duration: 0.55, ease: "power2.out" }, 13.65);
      if (arm) {
        tl.to(arm, { rotation: 6, transformOrigin: "90% 10%", duration: 7, ease: "sine.inOut", yoyo: true, repeat: 1 }, 0);
      }
      tlRef.current = tl;
      return () => {
        tlRef.current = null;
      };
    },
    { scope: ref, dependencies: [reduced] },
  );

  useEffect(() => {
    const tl = tlRef.current;
    if (!tl) return;
    if (inView) tl.play();
    else tl.pause();
  }, [inView, reduced]);

  return (
    <svg ref={ref} viewBox="0 0 470 380" className={`h-full w-full ${className}`} aria-hidden="true">
      {/* the vinyl: its own object now, only ~a quarter tucked behind
          the sleeves (§7.3), reading side by side with them */}
      <g data-vinyl>
        <circle cx="309" cy="190" r="96" fill={INK} stroke={SILVER} strokeOpacity="0.85" strokeWidth="2" />
        <circle cx="309" cy="190" r="74" fill="none" stroke={SILVER} strokeOpacity="0.4" />
        <circle cx="309" cy="190" r="54" fill="none" stroke={SILVER} strokeOpacity="0.32" />
        <circle cx="309" cy="190" r="84" fill="none" stroke={BLUE} strokeOpacity="0.8" strokeWidth="2" strokeDasharray="52 476" />
        <circle cx="309" cy="190" r="34" fill="none" stroke={SILVER} strokeOpacity="0.4" />
        <circle cx="309" cy="190" r="10" fill={GOLD} opacity="1" />
      </g>

      {/* back sleeve: concentric arcs */}
      <g data-sleeve-back transform="rotate(-7 140 150)">
        <rect x="66" y="76" width="148" height="148" fill={INK} stroke={SILVER} strokeOpacity="0.55" />
        <circle cx="140" cy="150" r="52" fill="none" stroke={GOLD} strokeOpacity="0.95" strokeWidth="2" />
        <circle cx="140" cy="150" r="36" fill="none" stroke={SILVER} strokeOpacity="0.6" />
        <circle cx="140" cy="150" r="20" fill="none" stroke={BLUE} strokeOpacity="0.7" strokeWidth="1.5" />
      </g>

      {/* mid sleeve: hard diagonals */}
      <g data-sleeve-mid transform="rotate(3 190 190)">
        <rect x="116" y="116" width="148" height="148" fill={INK} stroke={SILVER} strokeOpacity="0.6" />
        <line x1="128" y1="252" x2="252" y2="128" stroke={EMBER} strokeOpacity="0.95" strokeWidth="2.5" />
        <line x1="128" y1="232" x2="232" y2="128" stroke={SILVER} strokeOpacity="0.55" />
        <line x1="148" y1="252" x2="252" y2="148" stroke={GOLD} strokeOpacity="0.55" strokeWidth="1.5" />
      </g>

      {/* front sleeve: split field + off-centre dot, flips to face B */}
      <g data-sleeve-front transform="rotate(9 156 226)">
        <rect x="82" y="152" width="148" height="148" fill={INK} stroke={SILVER} strokeOpacity="0.7" />
        <g data-face-a>
          <path d="M82 300 L230 152 L230 300 Z" fill={BLUE} opacity="0.5" />
          <circle cx="128" cy="200" r="8" fill="var(--color-paper)" opacity="0.95" />
          <circle cx="128" cy="200" r="14" fill="none" stroke={GOLD} strokeOpacity="0.7" />
        </g>
        <g data-face-b>
          <rect x="98" y="168" width="116" height="10" fill={SILVER} opacity="0.75" />
          <rect x="98" y="188" width="84" height="10" fill={BLUE} opacity="0.9" />
          <rect x="98" y="208" width="100" height="10" fill={EMBER} opacity="0.6" />
          <circle cx="196" cy="264" r="10" fill={GOLD} opacity="1" />
        </g>
      </g>

      {/* tonearm hairline, resting over the vinyl */}
      <g data-arm>
        <line x1="438" y1="78" x2="352" y2="158" stroke={SILVER} strokeOpacity="0.75" strokeWidth="1.5" />
        <circle cx="438" cy="78" r="5" fill={SILVER} opacity="0.85" />
        <circle cx="352" cy="158" r="3" fill={BLUE} opacity="0.9" />
      </g>
    </svg>
  );
}
