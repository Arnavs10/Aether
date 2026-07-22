/**
 * AETHER — the ambient ground (Pass 7 §1: made STILL, permanently).
 *
 * Three passes of blur-tuning couldn't stop Chrome resolving the drifting
 * blobs as faint moving shapes, so the approach changes per the punch
 * list: the field is now rendered ONCE, at high internal resolution, and
 * the animation loop no longer exists. No per-frame repaint, no alpha
 * pulse, no drift — nothing left that can shimmer, in any engine. The
 * composition and palette are unchanged; the page's life comes from the
 * art and interactions, not the wallpaper.
 *
 * Redraws happen only on theme change (the light palette multiplies
 * colour washes into the paper instead of adding glow to the ink).
 */

import { useEffect, useRef } from "react";
import { getEffective, subscribeEffective } from "../../lib/theme";

interface Field {
  cx: number;
  cy: number;
  radius: number;
  color: [number, number, number];
  alpha: number;
}

/* The same composition the animated field settled around. */
const FIELDS_DARK: Field[] = [
  { cx: 0.24, cy: 0.32, radius: 0.55, color: [120, 130, 150], alpha: 0.055 },
  { cx: 0.76, cy: 0.6, radius: 0.6, color: [46, 107, 255], alpha: 0.05 },
  { cx: 0.52, cy: 0.94, radius: 0.5, color: [90, 110, 160], alpha: 0.045 },
];

const FIELDS_LIGHT: Field[] = [
  { cx: 0.24, cy: 0.32, radius: 0.55, color: [126, 122, 106], alpha: 0.07 },
  { cx: 0.76, cy: 0.6, radius: 0.6, color: [36, 86, 224], alpha: 0.075 },
  { cx: 0.52, cy: 0.94, radius: 0.5, color: [165, 126, 43], alpha: 0.05 },
];

/* High internal resolution: no upscale interpolation artifacts (§1.3). */
const W = 960;
const H = 540;

export function AmbientWaves() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = W;
    canvas.height = H;

    const ctxFilterOK =
      typeof ctx.filter === "string" &&
      (() => {
        try {
          ctx.filter = "blur(1px)";
          const ok = ctx.filter === "blur(1px)";
          ctx.filter = "none";
          return ok;
        } catch {
          return false;
        }
      })();
    canvas.style.filter = ctxFilterOK ? "blur(20px)" : "blur(46px)";

    const paint = (theme: "dark" | "light") => {
      ctx.filter = "none";
      ctx.clearRect(0, 0, W, H);
      if (ctxFilterOK) ctx.filter = "blur(26px)";
      ctx.globalCompositeOperation = theme === "light" ? "multiply" : "lighter";
      const fields = theme === "light" ? FIELDS_LIGHT : FIELDS_DARK;
      for (const b of fields) {
        const x = b.cx * W;
        const y = b.cy * H;
        const r = b.radius * Math.max(W, H);
        const g = ctx.createRadialGradient(x, y, 0, x, y, r);
        g.addColorStop(0, `rgba(${b.color[0]},${b.color[1]},${b.color[2]},${b.alpha})`);
        g.addColorStop(1, `rgba(${b.color[0]},${b.color[1]},${b.color[2]},0)`);
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, W, H);
      }
      ctx.filter = "none";
    };

    paint(getEffective());
    const unsubTheme = subscribeEffective(paint);
    return () => unsubTheme();
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="fixed inset-0 z-0 h-full w-full"
      style={{
        transform: "scale(1.12) translateZ(0)",
        willChange: "transform",
      }}
    />
  );
}
