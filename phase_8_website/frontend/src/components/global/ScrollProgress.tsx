/**
 * AETHER — scroll progress bar (§3.1 Safari fix).
 * No layout reads inside the scroll tick: with Lenis we use its own computed
 * progress; without it, the max scroll is cached and only recalculated when
 * the document actually resizes (ResizeObserver on <body> covers async
 * results growing the page).
 */

import { useEffect, useRef } from "react";
import type Lenis from "lenis";
import { useAppState } from "../../state/AppState";

export function ScrollProgress() {
  const barRef = useRef<HTMLDivElement>(null);
  const { lenis } = useAppState();

  useEffect(() => {
    const bar = barRef.current;
    if (!bar) return;

    if (lenis) {
      const onScroll = (l: Lenis) => {
        bar.style.transform = `scaleX(${l.progress ?? 0})`;
      };
      lenis.on("scroll", onScroll);
      onScroll(lenis);
      return () => lenis.off("scroll", onScroll);
    }

    // Fallback path (reduced motion): cached max, rAF-throttled ticks.
    let max = 1;
    const recalc = () => {
      max = Math.max(
        1,
        document.documentElement.scrollHeight - window.innerHeight,
      );
    };
    recalc();

    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        bar.style.transform = `scaleX(${Math.min(1, window.scrollY / max)})`;
        ticking = false;
      });
    };

    const ro = new ResizeObserver(recalc);
    ro.observe(document.body);
    window.addEventListener("resize", recalc);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    return () => {
      ro.disconnect();
      window.removeEventListener("resize", recalc);
      window.removeEventListener("scroll", onScroll);
    };
  }, [lenis]);

  return (
    <div
      ref={barRef}
      aria-hidden="true"
      className="fixed left-0 top-0 z-[70] h-[2px] w-full origin-left scale-x-0 bg-blue"
    />
  );
}
