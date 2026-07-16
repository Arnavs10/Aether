/**
 * AETHER — ambient background motion (§3.1 Safari fix).
 * The blob field now renders to a SMALL fixed-resolution canvas (240×135,
 * DPR 1) at ~30fps; the browser stretches it to the viewport with a CSS
 * blur. Same look, an order of magnitude cheaper than the old three
 * full-viewport gradient fills per frame. Paused on tab-hide; a single
 * static frame under prefers-reduced-motion.
 */

import { useEffect, useRef } from "react";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion";

const W = 240;
const H = 135;
const FRAME_MS = 33; // ~30fps cap

interface Field {
  cx: number;
  cy: number;
  orbitX: number;
  orbitY: number;
  speed: number;
  phase: number;
  radius: number; // fraction of max side
  color: [number, number, number];
  alpha: number;
}

const FIELDS: Field[] = [
  { cx: 0.22, cy: 0.3, orbitX: 0.1, orbitY: 0.08, speed: 0.045, phase: 0,
    radius: 0.55, color: [120, 130, 150], alpha: 0.055 },
  { cx: 0.78, cy: 0.62, orbitX: 0.09, orbitY: 0.1, speed: 0.035, phase: 2.1,
    radius: 0.6, color: [46, 107, 255], alpha: 0.05 },
  { cx: 0.5, cy: 0.95, orbitX: 0.12, orbitY: 0.05, speed: 0.028, phase: 4.2,
    radius: 0.5, color: [90, 110, 160], alpha: 0.045 },
];

export function AmbientWaves() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = W;
    canvas.height = H;

    let raf = 0;
    let running = true;
    let last = 0;

    const draw = (tMs: number) => {
      const t = tMs / 1000;
      ctx.clearRect(0, 0, W, H);
      ctx.globalCompositeOperation = "lighter";
      for (const b of FIELDS) {
        const x = (b.cx + Math.sin(t * b.speed * 2 + b.phase) * b.orbitX) * W;
        const y = (b.cy + Math.cos(t * b.speed * 1.6 + b.phase) * b.orbitY) * H;
        const r = b.radius * Math.max(W, H);
        const a = b.alpha * (0.85 + 0.15 * Math.sin(t * 0.3 + b.phase));
        const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
        const [cr, cg, cb] = b.color;
        grad.addColorStop(0, `rgba(${cr},${cg},${cb},${a})`);
        grad.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H);
      }
    };

    const loop = (tMs: number) => {
      if (!running) return;
      if (tMs - last >= FRAME_MS) {
        last = tMs;
        draw(tMs);
      }
      raf = requestAnimationFrame(loop);
    };

    if (reduced) {
      draw(1_000);
    } else {
      raf = requestAnimationFrame(loop);
    }

    const onVisibility = () => {
      running = document.visibilityState === "visible";
      if (running && !reduced) raf = requestAnimationFrame(loop);
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [reduced]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="fixed inset-0 z-0 h-full w-full"
      style={{ filter: "blur(46px)", transform: "scale(1.12)" }}
    />
  );
}
