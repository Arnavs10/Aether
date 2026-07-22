/**
 * AETHER — shared final block (Pass 4 §6).
 * §6.1 the stacked nav and the art column sit vertically centred on the
 *      same axis, one composition, equal air above and below both.
 * §6.3 the bottom bar shows the VISITOR: their city, country and live
 *      local time. IP lookup first (display-only, cached per visit),
 *      timezone-derived city second, time-only third. Never a wrong city,
 *      never a spinner, never undefined. No geolocation prompt, ever.
 * §6.4 (DETAILS) is Arnav's block: Bhopal.
 * §2   the art shrinks on mobile, it does not vanish.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router";
import { LINKS, NAV, SITE } from "../../config/site";
import { ParenLabel } from "../ui/ParenLabel";
import { SpectrumArt } from "../ui/SpectrumArt";
import { FaqAccordion } from "./FaqAccordion";
import { GoopField } from "./GoopField";
import { VinylDisc } from "../ui/VinylDisc";

interface VisitorLocale {
  city: string | null;
  country: string | null;
  timezone: string | null; // null = the browser's own zone
}

const VISITOR_KEY = "aether.visitor.v1";

/** Timezone id → a displayable city ("Asia/Kolkata" → "KOLKATA"). */
function cityFromTimezone(tz: string): string | null {
  const part = tz.split("/").pop();
  if (!part) return null;
  return part.replace(/_/g, " ");
}

function useVisitorLocale(): VisitorLocale {
  const [locale, setLocale] = useState<VisitorLocale>(() => {
    try {
      const cached = sessionStorage.getItem(VISITOR_KEY);
      if (cached) return JSON.parse(cached) as VisitorLocale;
    } catch {
      /* fall through */
    }
    return { city: null, country: null, timezone: null };
  });

  useEffect(() => {
    try {
      if (sessionStorage.getItem(VISITOR_KEY)) return; // once per visit
    } catch {
      /* storage blocked: still try, just uncached */
    }

    const fallback = (): VisitorLocale => {
      try {
        const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
        return { city: cityFromTimezone(tz), country: null, timezone: tz };
      } catch {
        return { city: null, country: null, timezone: null };
      }
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4_000);

    fetch("https://ipapi.co/json/", { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: { city?: string; country_name?: string; timezone?: string }) => {
        const next: VisitorLocale =
          typeof data.city === "string" && data.city
            ? {
                city: data.city,
                country: typeof data.country_name === "string" ? data.country_name : null,
                timezone: typeof data.timezone === "string" ? data.timezone : null,
              }
            : fallback();
        setLocale(next);
        try {
          sessionStorage.setItem(VISITOR_KEY, JSON.stringify(next));
        } catch {
          /* fine */
        }
      })
      .catch(() => {
        const next = fallback();
        setLocale(next);
        try {
          sessionStorage.setItem(VISITOR_KEY, JSON.stringify(next));
        } catch {
          /* fine */
        }
      })
      .finally(() => clearTimeout(timer));

    return () => controller.abort();
  }, []);

  return locale;
}

/** Live clock in the visitor's zone, with the right abbreviation. */
function useVisitorClock(timezone: string | null): string {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 10_000);
    return () => clearInterval(id);
  }, []);
  try {
    const parts = new Intl.DateTimeFormat("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
      timeZoneName: "short",
      ...(timezone ? { timeZone: timezone } : {}),
    }).formatToParts(now);
    const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
    const tzName = get("timeZoneName");
    return `${get("hour")}:${get("minute")}${tzName ? ` ${tzName}` : ""}`;
  } catch {
    return "";
  }
}

export function Footer() {
  const visitor = useVisitorLocale();
  const clock = useVisitorClock(visitor.timezone);

  const place = visitor.city
    ? `${visitor.city}${visitor.country ? `, ${visitor.country}` : ""}`.toUpperCase()
    : null;

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
            {/* §5: the weighted disc. drag it, flick it, it settles. */}
            <VinylDisc className="mx-auto mt-16 h-72 w-72 md:mt-20 md:h-[21rem] md:w-[21rem] md:-translate-x-6" />
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
                to="/connect#talk-back"
                className="mono-meta mt-5 inline-block border-b border-paper/30 pb-0.5 text-paper/80 transition-colors hover:border-blue hover:text-paper"
              >
                write to Arnav →
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── MONOLOG-style close, one centred composition ── */}
      <section className="border-t hairline px-6 pb-16 pt-20 md:px-10">
        <div className="grid gap-12 md:grid-cols-[1.2fr_1fr] md:items-center">
          <nav aria-label="Footer" className="flex flex-col gap-1 md:justify-center">
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
          {/* §2: shrinks on mobile, never vanishes */}
          <div className="h-40 md:h-80">
            <SpectrumArt variant="wall" className="h-full w-full" />
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
          <div className="flex flex-col gap-2 md:-translate-x-30">
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

        {/* §6.3 — the visitor's own city and time */}
        <div className="mt-20 flex flex-col gap-4 border-t hairline pt-8 md:grid md:grid-cols-3 md:items-baseline">
          <span className="mono-meta text-paper/40">
            {place ? `${place} · ` : ""}
            {clock}
          </span>
          <span className="serif-accent text-lg text-paper/60 md:justify-self-center md:text-center">
            {SITE.tagline}
          </span>
          <span className="mono-meta text-paper/40 md:justify-self-end">
            © {SITE.year} ARNAV SHUKLA
          </span>
        </div>
      </section>

      <GoopField />
    </footer>
  );
}
