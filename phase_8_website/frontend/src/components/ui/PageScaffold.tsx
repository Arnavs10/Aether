/**
 * AETHER — shared page header for the feature pages.
 */

import type { ReactNode } from "react";
import { ParenLabel } from "./ParenLabel";

interface PageHeaderProps {
  eyebrow: string;
  title: ReactNode;
  lede: string;
}

export function PageHeader({ eyebrow, title, lede }: PageHeaderProps) {
  return (
    <header className="px-6 pb-14 pt-40 md:px-10 md:pb-16 md:pt-48">
      <ParenLabel accent>{eyebrow}</ParenLabel>
      <h1 className="display mt-6 text-5xl text-paper md:text-7xl">{title}</h1>
      <p className="serif-accent mt-6 max-w-2xl text-xl text-paper/65 md:text-2xl">
        {lede}
      </p>
    </header>
  );
}
