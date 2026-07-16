/**
 * AETHER — global smooth momentum scrolling (§3.3).
 * One Lenis instance driven by GSAP's ticker (the canonical pairing), with
 * ScrollTrigger kept in sync. Skipped entirely under prefers-reduced-motion.
 * Also owns route-change behavior: jump to top + refresh triggers.
 */

import { useEffect } from "react";
import { useLocation } from "react-router";
import Lenis from "lenis";
import { gsap, ScrollTrigger } from "../../lib/gsap";
import { useAppState } from "../../state/AppState";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";

export function SmoothScroll() {
  const { setLenis } = useAppState();
  const reduced = usePrefersReducedMotion();
  const location = useLocation();

  useEffect(() => {
    if (reduced) {
      setLenis(null);
      return;
    }

    const lenis = new Lenis({
      duration: 1.15, // silky but not floaty
      smoothWheel: true,
    });
    setLenis(lenis);

    lenis.on("scroll", ScrollTrigger.update);
    const tick = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);

    // Re-measure trigger positions once webfonts have settled.
    document.fonts?.ready.then(() => ScrollTrigger.refresh());

    return () => {
      gsap.ticker.remove(tick);
      lenis.destroy();
      setLenis(null);
    };
  }, [reduced, setLenis]);

  // Route change: start each page at the top, then re-measure triggers.
  useEffect(() => {
    window.scrollTo(0, 0);
    const id = requestAnimationFrame(() => ScrollTrigger.refresh());
    return () => cancelAnimationFrame(id);
  }, [location.pathname]);

  return null;
}
