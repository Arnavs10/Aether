/**
 * AETHER — the theme store (Pass 5 §6). Three modes: dark (the default
 * character), light (a real designed theme, not an inversion), system
 * (follows prefers-color-scheme live). One module owns the state:
 * persists the choice, applies `.theme-light` on <html>, wraps every
 * switch in a short `.theme-anim` window so the change crossfades
 * instead of flipping, and lets the two canvases subscribe so their
 * hand-drawn palettes swap with everything else.
 */

export type ThemeMode = "dark" | "light" | "system";
export type EffectiveTheme = "dark" | "light";

const KEY = "aether.theme.v1";
const mql =
  typeof window !== "undefined"
    ? window.matchMedia("(prefers-color-scheme: light)")
    : null;

let mode: ThemeMode = "dark";
try {
  const saved = localStorage.getItem(KEY);
  if (saved === "dark" || saved === "light" || saved === "system") mode = saved;
} catch {
  /* private mode: session-only */
}

const modeListeners = new Set<(m: ThemeMode) => void>();
const effectiveListeners = new Set<(t: EffectiveTheme) => void>();
let animTimer: ReturnType<typeof setTimeout> | null = null;

export function getMode(): ThemeMode {
  return mode;
}

export function getEffective(): EffectiveTheme {
  if (mode === "system") return mql?.matches ? "light" : "dark";
  return mode;
}

function apply(animated: boolean): void {
  const root = document.documentElement;
  if (animated) {
    root.classList.add("theme-anim");
    if (animTimer) clearTimeout(animTimer);
    animTimer = setTimeout(() => root.classList.remove("theme-anim"), 650);
  }
  root.classList.toggle("theme-light", getEffective() === "light");
  root.style.colorScheme = getEffective() === "light" ? "light" : "dark";
  for (const cb of effectiveListeners) cb(getEffective());
}

export function setMode(next: ThemeMode): void {
  mode = next;
  try {
    localStorage.setItem(KEY, next);
  } catch {
    /* fine */
  }
  for (const cb of modeListeners) cb(mode);
  apply(true);
}

export function cycleMode(): ThemeMode {
  const order: ThemeMode[] = ["dark", "light", "system"];
  setMode(order[(order.indexOf(mode) + 1) % order.length]);
  return mode;
}

export function subscribeMode(cb: (m: ThemeMode) => void): () => void {
  modeListeners.add(cb);
  cb(mode);
  return () => {
    modeListeners.delete(cb);
  };
}

export function subscribeEffective(cb: (t: EffectiveTheme) => void): () => void {
  effectiveListeners.add(cb);
  cb(getEffective());
  return () => {
    effectiveListeners.delete(cb);
  };
}

/* System changes flow through live while in system mode (§6). */
mql?.addEventListener("change", () => {
  if (mode === "system") apply(true);
});

/* Apply once at import so the first paint is already correct. */
if (typeof document !== "undefined") apply(false);
