/**
 * AETHER — the bottom field, rebuilt for real (§6).
 *
 * A particle lattice riding two interfering traveling waves, drawn in two
 * contrasting colours (blue against silver, rare ember sparks). Interaction:
 * as the pointer moves over the field, nearby particles GATHER toward it and
 * STIFFEN (their wave motion freezes as they approach). On release or leave
 * they get a spring impulse outward, then ease back into the calm flow.
 * Real velocity/spring integration per particle, not a canned tween.
 *
 * Performance (§3.1): the entire frame is THREE batched Path2D fills (one
 * per colour), never a path per dot. DPR capped at 1.5, IntersectionObserver
 * pauses it offscreen, a single static frame under prefers-reduced-motion.
 */

import { useEffect, useRef } from "react";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";
import { getEffective, subscribeEffective } from "../../lib/theme";

const SPACING_FINE = 10; // CSS px between rest positions (§7.1: denser)
const SPACING_COARSE = 13; // phones get fewer particles (§2)
const DOT = 2.6; // square side, CSS px
const ATTRACT_R = 150; // pointer influence radius, CSS px
const PULL = 0.085; // gather strength
const BURST = 4.2; // release impulse
const SPRING = 0.045; // pull back toward the flow target
const DAMP = 0.87; // velocity damping

/* §6: two palettes, same physics. On light ground the near-pointer
   highlight goes deep ink and the depth dots go white — same contrast
   logic, flipped for the paper. */
const PALETTES = {
  dark: {
    blue: "rgba(46,107,255,0.6)",
    silver: "rgba(199,204,212,0.42)",
    ember: "rgba(214,69,61,0.75)",
    bright: "rgba(245,246,248,0.9)", // gathered, near the pointer
    depth: "rgba(0,0,0,0.55)",
  },
  light: {
    blue: "rgba(36,86,224,0.55)",
    silver: "rgba(86,92,104,0.38)",
    ember: "rgba(192,58,50,0.7)",
    bright: "rgba(18,21,28,0.88)",
    depth: "rgba(255,255,255,0.8)",
  },
} as const;

export function GoopField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const coarse = window.matchMedia("(pointer: coarse)").matches;
    const dpr = coarse ? 1 : Math.min(window.devicePixelRatio || 1, 1.5);
    const spacing = coarse ? SPACING_COARSE : SPACING_FINE;
    let raf = 0;
    let visible = true;
    let running = true;

    // Particle state (canvas-space units).
    let restX = new Float32Array(0);
    let restY = new Float32Array(0);
    let px = new Float32Array(0);
    let py = new Float32Array(0);
    let vx = new Float32Array(0);
    let vy = new Float32Array(0);
    let count = 0;

    const pointer = { x: 0, y: 0, active: false, down: false };
    let pal: (typeof PALETTES)[keyof typeof PALETTES] = PALETTES[getEffective()];
    const unsubTheme = subscribeEffective((t) => {
      pal = PALETTES[t];
      if (reduced) frame(400); // repaint the static frame in the new palette
    });

    const build = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const step = spacing * dpr;
      const cols = Math.ceil(canvas.width / step);
      const rows = Math.ceil(canvas.height / step);
      count = cols * rows;
      restX = new Float32Array(count);
      restY = new Float32Array(count);
      px = new Float32Array(count);
      py = new Float32Array(count);
      vx = new Float32Array(count);
      vy = new Float32Array(count);
      let i = 0;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          restX[i] = c * step + step / 2;
          restY[i] = r * step + step / 2;
          px[i] = restX[i];
          py[i] = restY[i];
          i++;
        }
      }
    };
    build();

    const attractR = ATTRACT_R * dpr;
    const dot = DOT * dpr;
    const half = dot / 2;

    const frame = (tMs: number) => {
      const t = tMs / 1000;
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const pBlue = new Path2D();
      const pSilver = new Path2D();
      const pEmber = new Path2D();
      const pWhite = new Path2D();
      const pBlack = new Path2D();
      const whiteR = attractR * 0.45;

      for (let i = 0; i < count; i++) {
        const rx = restX[i];
        const ry = restY[i];

        // The calm flow: two interfering traveling waves.
        let waveX = Math.sin(rx * 0.010 - t * 1.05 + ry * 0.018) * 3.2 * dpr;
        let waveY =
          (Math.sin(rx * 0.017 + t * 0.7 - ry * 0.012) +
            Math.sin(ry * 0.021 + t * 0.9)) *
          2.4 *
          dpr;

        // Stiffen near the pointer: flow amplitude dies as it approaches.
        let dxp = 0;
        let dyp = 0;
        let dist = Infinity;
        if (pointer.active) {
          dxp = pointer.x - px[i];
          dyp = pointer.y - py[i];
          dist = Math.hypot(dxp, dyp);
          if (dist < attractR) {
            const k = dist / attractR; // 0 at pointer → 1 at edge
            waveX *= k;
            waveY *= k;
          }
        }

        // Spring toward the flow target.
        const tx = rx + waveX;
        const ty = ry + waveY;
        vx[i] += (tx - px[i]) * SPRING;
        vy[i] += (ty - py[i]) * SPRING;

        // Gather toward the pointer.
        if (pointer.active && dist < attractR) {
          const s = (1 - dist / attractR) * PULL;
          vx[i] += dxp * s;
          vy[i] += dyp * s;
        }

        vx[i] *= DAMP;
        vy[i] *= DAMP;
        px[i] += vx[i];
        py[i] += vy[i];

        const x = px[i] - half;
        const y = py[i] - half;
        // §7.3: the brightest gathered particles glow white at the pointer.
        if (pointer.active && dist < whiteR) pWhite.rect(x, y, dot, dot);
        else if (i % 41 === 0) pEmber.rect(x, y, dot, dot);
        else if (i % 23 === 0) pBlack.rect(x, y, dot, dot);
        else if (i % 2 === 0) pBlue.rect(x, y, dot, dot);
        else pSilver.rect(x, y, dot, dot);
      }

      ctx.fillStyle = pal.depth;
      ctx.fill(pBlack);
      ctx.fillStyle = pal.silver;
      ctx.fill(pSilver);
      ctx.fillStyle = pal.blue;
      ctx.fill(pBlue);
      ctx.fillStyle = pal.ember;
      ctx.fill(pEmber);
      ctx.fillStyle = pal.bright;
      ctx.fill(pWhite);
    };

    const loop = (tMs: number) => {
      if (!running) return;
      if (visible) frame(tMs);
      raf = requestAnimationFrame(loop);
    };

    if (reduced) {
      frame(400); // static composed texture
    } else {
      raf = requestAnimationFrame(loop);
    }

    /* Release: spring the gathered particles outward, then the flow reclaims
       them naturally through the same integrator. */
    const release = () => {
      if (!pointer.active) return;
      for (let i = 0; i < count; i++) {
        const dx = px[i] - pointer.x;
        const dy = py[i] - pointer.y;
        const d = Math.hypot(dx, dy);
        if (d < attractR && d > 0.001) {
          const s = (1 - d / attractR) * BURST;
          vx[i] += (dx / d) * s;
          vy[i] += (dy / d) * s;
        }
      }
      pointer.active = false;
      pointer.down = false;
    };

    const toCanvas = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = (e.clientX - rect.left) * dpr;
      pointer.y = (e.clientY - rect.top) * dpr;
    };

    const onMove = (e: PointerEvent) => {
      if (reduced) return;
      toCanvas(e);
      // Mouse gathers on hover; touch gathers only while pressed.
      if (e.pointerType === "mouse") pointer.active = true;
      else pointer.active = pointer.down;
    };
    const onDown = (e: PointerEvent) => {
      if (reduced) return;
      toCanvas(e);
      pointer.down = true;
      pointer.active = true;
    };
    const onUp = () => {
      if (reduced) return;
      release();
    };
    const onLeave = () => {
      if (reduced) return;
      release();
    };

    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);

    const io = new IntersectionObserver(([entry]) => {
      visible = entry?.isIntersecting ?? true;
    });
    io.observe(canvas);

    const ro = new ResizeObserver(() => {
      build();
      if (reduced) frame(400);
    });
    ro.observe(canvas);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      unsubTheme();
      io.disconnect();
      ro.disconnect();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
    };
  }, [reduced]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="block h-56 w-full touch-pan-y md:h-72"
    />
  );
}
