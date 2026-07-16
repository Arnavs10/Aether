/**
 * AETHER — film-grain overlay (§3.2 continuation).
 * A texture you only notice if you look for it: one generated noise tile,
 * STATIC (no jitter, no blend-layer churn), opacity 0.02. Near-zero cost.
 */

import { useEffect, useState } from "react";

function makeNoiseTile(size = 160): string {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.floor(Math.random() * 255);
    img.data[i] = v;
    img.data[i + 1] = v;
    img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

export function GrainOverlay() {
  const [tile, setTile] = useState("");
  useEffect(() => setTile(makeNoiseTile()), []);
  if (!tile) return null;

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-[60] opacity-[0.02]"
      style={{ backgroundImage: `url(${tile})`, backgroundRepeat: "repeat" }}
    />
  );
}
