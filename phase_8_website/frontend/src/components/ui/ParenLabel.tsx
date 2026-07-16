/**
 * AETHER — the (PARENTHETICAL LABEL) system (§2 signature).
 * MONOLOG-style mono meta captions, purpose-built for real ML data:
 * (1.2M SONGS) (CAMELOT 10A) (DRIFT 1.00). Accent renders blue.
 */

import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
  accent?: boolean;
  className?: string;
}

export function ParenLabel({ children, accent = false, className = "" }: Props) {
  return (
    <span
      className={`mono-meta ${accent ? "text-blue" : "text-paper/45"} ${className}`}
    >
      ({children})
    </span>
  );
}
