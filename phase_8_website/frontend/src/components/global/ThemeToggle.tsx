/**
 * AETHER — the theme toggle (Pass 5 §6): a glass bead stacked directly
 * above the AetherBot bubble, same size, same material. Click cycles
 * dark → light → system; the icon says which one you're on.
 */

import { useEffect, useState } from "react";
import { cycleMode, subscribeMode, type ThemeMode } from "../../lib/theme";

const LABEL: Record<ThemeMode, string> = {
  dark: "Theme: dark. Click for light",
  light: "Theme: light. Click for system",
  system: "Theme: system. Click for dark",
};

function Icon({ mode }: { mode: ThemeMode }) {
  if (mode === "dark") {
    return (
      <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
        <path
          d="M 15.5 3.5 A 9 9 0 1 0 20.5 13.5 A 7 7 0 0 1 15.5 3.5 Z"
          fill="none"
          stroke="var(--color-paper)"
          strokeOpacity="0.8"
          strokeWidth="1.6"
        />
      </svg>
    );
  }
  if (mode === "light") {
    return (
      <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
        <circle cx="12" cy="12" r="4.2" fill="none" stroke="var(--color-paper)" strokeOpacity="0.85" strokeWidth="1.6" />
        {Array.from({ length: 8 }).map((_, i) => {
          const a = (i * Math.PI) / 4;
          return (
            <line
              key={i}
              x1={12 + Math.cos(a) * 7}
              y1={12 + Math.sin(a) * 7}
              x2={12 + Math.cos(a) * 9.6}
              y2={12 + Math.sin(a) * 9.6}
              stroke="var(--color-paper)"
              strokeOpacity="0.85"
              strokeWidth="1.6"
            />
          );
        })}
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
      <circle cx="12" cy="12" r="8" fill="none" stroke="var(--color-paper)" strokeOpacity="0.8" strokeWidth="1.6" />
      <path d="M 12 4 A 8 8 0 0 1 12 20 Z" fill="var(--color-paper)" opacity="0.7" />
    </svg>
  );
}

export function ThemeToggle() {
  const [mode, setModeState] = useState<ThemeMode>("dark");
  useEffect(() => subscribeMode(setModeState), []);

  return (
    <button
      type="button"
      onClick={() => cycleMode()}
      aria-label={LABEL[mode]}
      title={LABEL[mode]}
      className="glass-liquid fixed bottom-[5.5rem] right-6 z-[84] flex h-14 w-14 items-center justify-center rounded-full transition-all duration-300 hover:border-paper/30"
    >
      <Icon mode={mode} />
    </button>
  );
}
