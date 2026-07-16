/**
 * AETHER — minimal app-wide state.
 *  • engine:   result of the preloader's /health poll — "ready" | "degraded"
 *              ("degraded" = user chose to enter while the API was unreachable;
 *              feature pages surface a quiet banner instead of raw errors).
 *  • engineInfo: the last successful /health payload (real track count).
 *  • appReady: flips true when the preloader finishes — gates hero entrance.
 *  • lenis:    the shared smooth-scroll instance (scroll progress, route jumps).
 */

import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type Lenis from "lenis";
import type { HealthResponse } from "../lib/types";

export type EngineStatus = "checking" | "ready" | "degraded";

interface AppState {
  engine: EngineStatus;
  setEngine: (s: EngineStatus) => void;
  engineInfo: HealthResponse | null;
  setEngineInfo: (h: HealthResponse | null) => void;
  appReady: boolean;
  setAppReady: (v: boolean) => void;
  lenis: Lenis | null;
  setLenis: (l: Lenis | null) => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [engine, setEngine] = useState<EngineStatus>("checking");
  const [engineInfo, setEngineInfo] = useState<HealthResponse | null>(null);
  const [appReady, setAppReady] = useState(false);
  const [lenis, setLenis] = useState<Lenis | null>(null);

  const value = useMemo(
    () => ({
      engine,
      setEngine,
      engineInfo,
      setEngineInfo,
      appReady,
      setAppReady,
      lenis,
      setLenis,
    }),
    [engine, engineInfo, appReady, lenis],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAppState(): AppState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAppState must be used inside AppStateProvider");
  return ctx;
}
