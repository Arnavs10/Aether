/**
 * AETHER — global smooth momentum scrolling (§3.3).
 * One Lenis instance driven by GSAP's ticker (the canonical pairing), with
 * ScrollTrigger kept in sync. Skipped entirely under prefers-reduced-motion.
 * Also owns route-change behavior: jump to top + refresh triggers.
 */

import { useEffect, useRef } from "react";
import { useLocation } from "react-router";
import Lenis from "lenis";
import { gsap, ScrollTrigger } from "../../lib/gsap";
import { useAppState } from "../../state/AppState";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";

export function SmoothScroll() {
  const { setLenis, lenis } = useAppState();
  const lenisRef = useRef(lenis);
  lenisRef.current = lenis;
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

    ScrollTrigger.refresh();
    // Re-measure trigger positions once webfonts have settled.
    document.fonts?.ready.then(() => ScrollTrigger.refresh());

    return () => {
      gsap.ticker.remove(tick);
      lenis.destroy();
      setLenis(null);
    };
  }, [reduced, setLenis]);

  // Route change: start at the top — unless a hash points somewhere
  // (Pass 5 §5: "write to Arnav" lands on the form, not the page top).
  useEffect(() => {
    const hash = location.hash;
    if (hash) {
      const timer = setTimeout(() => {
        ScrollTrigger.refresh();
        const el = document.querySelector(hash);
        if (!el) return;
        const l = lenisRef.current;
        if (l) l.scrollTo(el as HTMLElement, { offset: -90 });
        else el.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 90);
      return () => clearTimeout(timer);
    }
    window.scrollTo(0, 0);
    // §4: re-measuring pinned triggers is the expensive part of coming
    // back to Home in Safari — let the page paint first.
    const w = window as Window & {
      requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number;
      cancelIdleCallback?: (id: number) => void;
    };
    if (typeof w.requestIdleCallback === "function") {
      const id = w.requestIdleCallback(() => ScrollTrigger.refresh(), { timeout: 500 });
      return () => w.cancelIdleCallback?.(id);
    }
    const id = setTimeout(() => ScrollTrigger.refresh(), 120);
    return () => clearTimeout(id);
  }, [location.pathname, location.hash]);

  return null;
}
