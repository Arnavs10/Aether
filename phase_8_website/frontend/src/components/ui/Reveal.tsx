/**
 * AETHER — scroll reveal primitive. A quiet fade-and-settle on entering the
 * viewport (power3.out) via ScrollTrigger; renders instantly visible under
 * prefers-reduced-motion. Used for the editorial scaffolds; the heavier
 * choreography (§4D/4F pins) lands in Pass 5.
 */

import { useRef, type ReactNode } from "react";
import { gsap, useGSAP } from "../../lib/gsap";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";

interface Props {
  children: ReactNode;
  delay?: number;
  y?: number;
  className?: string;
}

export function Reveal({ children, delay = 0, y = 28, className = "" }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = usePrefersReducedMotion();

  useGSAP(
    () => {
      if (reduced || !ref.current) return;
      gsap.from(ref.current, {
        y,
        autoAlpha: 0,
        duration: 0.9,
        delay,
        ease: "power3.out",
        scrollTrigger: { trigger: ref.current, start: "top 86%" },
      });
    },
    { scope: ref, dependencies: [reduced] },
  );

  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}
