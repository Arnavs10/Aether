import { useEffect, useState } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

/** Live media-query hook — every motion system checks this (§3.4). */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState<boolean>(
    () => typeof window !== "undefined" && window.matchMedia(QUERY).matches,
  );

  useEffect(() => {
    const mq = window.matchMedia(QUERY);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return reduced;
}
