/**
 * AETHER — shared final block on every page (§5 continuation).
 * FAQ first, then the MONOLOG close: the big stacked nav now shares the row
 * with a tall equalizer wall (§5.1), the columns drop to DETAILS + SOCIALS
 * with two links only (§5.2), and the field sits at the very bottom.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router";
import { LINKS, NAV, SITE } from "../../config/site";
import { ParenLabel } from "../ui/ParenLabel";
import { SpectrumArt } from "../ui/SpectrumArt";
import { FaqAccordion } from "./FaqAccordion";
import { GoopField } from "./GoopField";

function useLocalTime(): string {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 30_000);
    return () => clearInterval(id);
  }, []);
  return new Intl.DateTimeFormat("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: SITE.timezone,
  }).format(now);
}

export function Footer() {
  const time = useLocalTime();

  return (
    <footer className="relative z-10 mt-32">
      {/* ── FAQ block ────────────────────────────────────── */}
      <section className="border-t hairline px-6 py-24 md:px-10 md:py-32">
        <div className="grid gap-14 md:grid-cols-2 md:gap-10">
          <div>
            <ParenLabel>FAQ</ParenLabel>
            <h2 className="display mt-5 text-4xl text-paper md:text-6xl">
              Everything
              <br />
              you might ask
            </h2>
            <p className="serif-accent mt-6 text-xl text-paper/60">
              answered plainly, nothing hidden.
            </p>
          </div>
          <div className="flex flex-col gap-10">
            <FaqAccordion />
            <div className="glass rounded-sm p-6">
              <ParenLabel>STILL CURIOUS</ParenLabel>
              <p className="mt-3 max-w-md text-sm leading-relaxed text-mist">
                Questions about how it works, internship conversations, or a
                feature you wish existed. Arnav reads everything sent through
                the Connect page.
              </p>
              <Link
                to="/connect"
                className="mono-meta mt-5 inline-block border-b border-paper/30 pb-0.5 text-paper/80 transition-colors hover:border-blue hover:text-paper"
              >
                write to Arnav →
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── MONOLOG-style close ─────────────────────────── */}
      <section className="border-t hairline px-6 pb-16 pt-20 md:px-10">
        <div className="grid gap-12 md:grid-cols-[1.2fr_1fr] md:items-stretch">
          <nav aria-label="Footer" className="flex flex-col gap-1">
            {NAV.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                className="display group w-fit text-5xl text-paper/85 transition-all duration-300 hover:translate-x-3 hover:text-paper md:text-7xl"
              >
                {item.label}
                <span className="ml-4 inline-block text-blue opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                  ↗
                </span>
              </Link>
            ))}
          </nav>
          <div className="hidden md:block">
            <SpectrumArt variant="wall" className="h-full min-h-80 w-full" />
          </div>
        </div>

        <div className="mt-20 grid gap-10 text-sm md:grid-cols-2">
          <div className="flex flex-col gap-2">
            <ParenLabel>DETAILS</ParenLabel>
            <a
              href={`mailto:${LINKS.email}`}
              className="mt-2 w-fit text-paper/70 transition-colors hover:text-paper"
            >
              {LINKS.email}
            </a>
            <span className="text-mist">{SITE.location}</span>
          </div>
          <div className="flex flex-col gap-2">
            <ParenLabel>SOCIALS</ParenLabel>
            <a
              href={LINKS.repo}
              target="_blank"
              rel="noreferrer"
              className="mt-2 w-fit text-paper/70 transition-colors hover:text-paper"
            >
              GitHub ↗
            </a>
            <a
              href={LINKS.linkedin}
              target="_blank"
              rel="noreferrer"
              className="w-fit text-paper/70 transition-colors hover:text-paper"
            >
              LinkedIn ↗
            </a>
          </div>
        </div>

        <div className="mt-20 flex flex-col gap-4 border-t hairline pt-8 md:flex-row md:items-baseline md:justify-between">
          <span className="mono-meta text-paper/40">
            {SITE.location.toUpperCase()} · {time} IST
          </span>
          <span className="serif-accent text-lg text-paper/60">
            {SITE.tagline}
          </span>
          <span className="mono-meta text-paper/40">
            © {SITE.year} ARNAV SHUKLA
          </span>
        </div>
      </section>

      {/* ── the field (§6) ───────────────────────────────── */}
      <GoopField />
    </footer>
  );
}
