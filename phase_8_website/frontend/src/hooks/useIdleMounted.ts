/**
 * AETHER — Pass 5 §11.1. True after the browser has had an idle moment
 * (or a short fallback delay). Used to defer heavy decorative visuals so
 * a page paints and becomes interactive first, then the art fades in.
 */

import { useEffect, useState } from "react";

export function useIdleMounted(fallbackDelayMs = 140): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const w = window as Window & {
      requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number;
      cancelIdleCallback?: (id: number) => void;
    };
    if (typeof w.requestIdleCallback === "function") {
      const id = w.requestIdleCallback(() => setReady(true), { timeout: 1200 });
      return () => w.cancelIdleCallback?.(id);
    }
    const id = setTimeout(() => setReady(true), fallbackDelayMs);
    return () => clearTimeout(id);
  }, [fallbackDelayMs]);
  return ready;
}
