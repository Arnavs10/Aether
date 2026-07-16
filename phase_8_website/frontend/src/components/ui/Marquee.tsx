/**
 * AETHER — infinite marquee (§4H). Content is duplicated once and translated
 * -50% in a seamless loop; hover pauses (CSS play-state). Reduced motion →
 * a static wrapped row.
 */

import type { ReactNode } from "react";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";

interface Props {
  children: ReactNode;
  /** Seconds per loop. */
  duration?: number;
}

export function Marquee({ children, duration = 42 }: Props) {
  const reduced = usePrefersReducedMotion();

  if (reduced) {
    return <div className="flex flex-wrap gap-6">{children}</div>;
  }

  return (
    <div className="marquee-paused overflow-hidden">
      <div
        className="animate-marquee flex w-max gap-6 pr-6"
        style={{ "--marquee-s": `${duration}s` } as React.CSSProperties}
      >
        {children}
        {children}
      </div>
    </div>
  );
}
