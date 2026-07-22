/**
 * AETHER — the FAQ disc (Pass 4 §5). Wireframe record geometry that idles
 * at a slow constant spin, follows a drag or flick with real momentum,
 * and decays smoothly back to idle on release. Velocity decay, not a
 * canned animation. Pointer events (mouse + touch), the shared gsap
 * ticker (one rAF for everything), IO-gated, static under reduced motion.
 */

import { useEffect, useRef } from "react";
import { gsap } from "../../lib/gsap";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";

const IDLE_DEG_S = 9;

export function VinylDisc({ className = "" }: { className?: string }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const discRef = useRef<SVGGElement>(null);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const wrap = wrapRef.current;
    const disc = discRef.current;
    if (!wrap || !disc) return;
    if (reduced) {
      gsap.set(disc, { rotation: 24, transformOrigin: "50% 50%" });
      return;
    }

    gsap.set(disc, { transformOrigin: "50% 50%" });

    let angle = 0;
    let vel = IDLE_DEG_S;
    let dragging = false;
    let lastPointerAngle = 0;
    let lastMoveAt = 0;
    let inView = true;
    let lastTick = performance.now();

    const centerAngle = (e: PointerEvent): number => {
      const r = wrap.getBoundingClientRect();
      return (
        (Math.atan2(e.clientY - (r.top + r.height / 2), e.clientX - (r.left + r.width / 2)) * 180) /
        Math.PI
      );
    };

    const tick = () => {
      const now = performance.now();
      const dt = Math.min(0.05, (now - lastTick) / 1000);
      lastTick = now;
      if (!inView) return;
      if (!dragging) {
        // Weighted decay back toward the slow idle.
        vel = IDLE_DEG_S + (vel - IDLE_DEG_S) * Math.exp(-dt * 0.85);
        angle += vel * dt;
        gsap.set(disc, { rotation: angle });
      }
    };
    gsap.ticker.add(tick);

    const onDown = (e: PointerEvent) => {
      dragging = true;
      lastPointerAngle = centerAngle(e);
      lastMoveAt = performance.now();
      wrap.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      const a = centerAngle(e);
      let delta = a - lastPointerAngle;
      if (delta > 180) delta -= 360;
      if (delta < -180) delta += 360;
      lastPointerAngle = a;
      const now = performance.now();
      const dt = Math.max(0.008, (now - lastMoveAt) / 1000);
      lastMoveAt = now;
      angle += delta;
      // Smoothed instantaneous velocity: this is the flick.
      vel = vel * 0.55 + (delta / dt) * 0.45;
      gsap.set(disc, { rotation: angle });
    };
    const onUp = () => {
      dragging = false;
    };

    wrap.addEventListener("pointerdown", onDown);
    wrap.addEventListener("pointermove", onMove);
    wrap.addEventListener("pointerup", onUp);
    wrap.addEventListener("pointercancel", onUp);

    const io = new IntersectionObserver(([entry]) => {
      inView = entry?.isIntersecting ?? true;
      lastTick = performance.now();
    });
    io.observe(wrap);

    return () => {
      gsap.ticker.remove(tick);
      io.disconnect();
      wrap.removeEventListener("pointerdown", onDown);
      wrap.removeEventListener("pointermove", onMove);
      wrap.removeEventListener("pointerup", onUp);
      wrap.removeEventListener("pointercancel", onUp);
    };
  }, [reduced]);

  return (
    <div
      ref={wrapRef}
      className={`cursor-grab touch-none select-none active:cursor-grabbing ${className}`}
      role="img"
      aria-label="A spinning wireframe record. Drag it."
    >
      <svg viewBox="0 0 320 320" className="h-full w-full">
        <g ref={discRef}>
          <circle cx="160" cy="160" r="150" fill="none" stroke="var(--color-silver)" strokeOpacity="0.8" strokeWidth="2" />
          <circle cx="160" cy="160" r="126" fill="none" stroke="var(--color-silver)" strokeOpacity="0.42" />
          <circle cx="160" cy="160" r="104" fill="none" stroke="var(--color-silver)" strokeOpacity="0.34" />
          <circle cx="160" cy="160" r="82" fill="none" stroke="var(--color-silver)" strokeOpacity="0.28" />
          {/* one ember hairline groove */}
          <circle cx="160" cy="160" r="115" fill="none" stroke="var(--color-ember)" strokeOpacity="0.9" strokeWidth="1.5" strokeDasharray="40 683" />
          {/* rotation ticks */}
          {Array.from({ length: 8 }).map((_, i) => {
            const a = (i * Math.PI) / 4;
            const x1 = 160 + Math.cos(a) * 138;
            const y1 = 160 + Math.sin(a) * 138;
            const x2 = 160 + Math.cos(a) * 150;
            const y2 = 160 + Math.sin(a) * 150;
            return (
              <line
                key={i}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={i === 0 ? "var(--color-blue)" : "var(--color-silver)"}
                strokeOpacity={i === 0 ? 1 : 0.55}
                strokeWidth={i === 0 ? 2.5 : 1}
              />
            );
          })}
          <circle cx="160" cy="160" r="30" fill="none" stroke="var(--color-silver)" strokeOpacity="0.5" />
          <circle cx="160" cy="160" r="6" fill="var(--color-gold)" opacity="1" />
        </g>
      </svg>
    </div>
  );
}
