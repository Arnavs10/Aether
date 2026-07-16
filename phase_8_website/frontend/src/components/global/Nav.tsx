/**
 * AETHER — top navigation (§4: Home · Curate · Journey · Live · Connect).
 * Frosted-glass bar once scrolled; mono meta links with a blue active dot.
 * Spotify login (§14): while FLAGS.spotifyLogin is off, NOTHING renders.
 * When credentials exist, flipping the flag is the only change needed.
 */

import { useEffect, useState } from "react";
import { Link, NavLink } from "react-router";
import { FLAGS, NAV, SITE } from "../../config/site";

export function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Close the mobile overlay on Escape; lock body scroll while open.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open]);

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `mono-meta relative transition-colors duration-300 ${
      isActive ? "text-paper" : "text-paper/45 hover:text-paper"
    } ${isActive ? "nav-active" : ""}`;

  return (
    <>
      <header
        className={`fixed inset-x-0 top-0 z-[80] transition-all duration-500 ${
          scrolled ? "glass border-b" : "border-b border-transparent"
        }`}
      >
        <div className="flex h-16 items-center justify-between px-6 md:px-10">
          <Link
            to="/"
            className="display text-base tracking-normal text-paper"
            aria-label="Aether, home"
          >
            {SITE.wordmark}
          </Link>

          {/* desktop links */}
          <nav aria-label="Primary" className="hidden items-center gap-8 md:flex">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.to === "/"} className={linkClass}>
                {({ isActive }) => (
                  <span className="flex items-center gap-2">
                    {isActive && (
                      <span className="h-1 w-1 rounded-full bg-blue" aria-hidden="true" />
                    )}
                    {item.label}
                  </span>
                )}
              </NavLink>
            ))}
          </nav>

          {FLAGS.spotifyLogin && (
            <div className="hidden md:block">
              <button
                type="button"
                className="mono-meta rounded-full border border-paper/20 px-4 py-2 text-paper/80 transition-colors hover:border-blue hover:text-paper"
              >
                Log in with Spotify
              </button>
            </div>
          )}

          {/* mobile toggle */}
          <button
            type="button"
            onClick={() => setOpen(true)}
            aria-label="Open menu"
            aria-expanded={open}
            className="flex h-10 w-10 flex-col items-center justify-center gap-1.5 md:hidden"
          >
            <span className="h-px w-6 bg-paper" />
            <span className="h-px w-6 bg-paper" />
          </button>
        </div>
      </header>

      {/* mobile overlay menu */}
      {open && (
        <div className="fixed inset-0 z-[90] flex flex-col bg-ink/95 backdrop-blur-xl md:hidden">
          <div className="flex h-16 items-center justify-between px-6">
            <span className="display text-base text-paper">{SITE.wordmark}</span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close menu"
              autoFocus
              className="mono-meta text-paper/70"
            >
              CLOSE
            </button>
          </div>
          <nav
            aria-label="Primary"
            className="flex flex-1 flex-col justify-center gap-2 px-6"
          >
            {NAV.map((item, i) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                onClick={() => setOpen(false)}
                style={{ transitionDelay: `${i * 40}ms` }}
                className={({ isActive }) =>
                  `display text-5xl transition-colors ${
                    isActive ? "text-paper" : "text-paper/40 hover:text-paper"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      )}
    </>
  );
}
