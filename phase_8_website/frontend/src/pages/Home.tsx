/**
 * AETHER — Home (§4 continuation edits applied).
 * 4A hero (wordmark treatment locked as final, bracket stats removed)
 * 4B intro + diagonal blue glare sweep tied to scroll
 * 4C FEEL/HEARD squeeze with its label folded into the same motion
 * 4D pipeline catalogue, tech tags removed, plain-language copy
 * 4E hover list + per-emotion spectrum reveal
 * 4F three ways in, with the reusable SpectrumArt pieces
 * 4G about, recomposed: bio + links left, art panel right
 * 4H human quote marquee (QUOTES is the seam for real /contact messages)
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router";
import { gsap, useGSAP } from "../lib/gsap";
import { useAppState } from "../state/AppState";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { useIdleMounted } from "../hooks/useIdleMounted";
import { AETHER_EMOTIONS, EMOTION_NOTES, LINKS, SITE } from "../config/site";
import type { EmotionLabel } from "../config/site";
import { ParenLabel } from "../components/ui/ParenLabel";
import { Reveal } from "../components/ui/Reveal";
import { Marquee } from "../components/ui/Marquee";
import {
  SpectrumArt,
  EmotionSpectrum,
  ConstellationArt,
  EMOTION_LINES,
} from "../components/ui/SpectrumArt";
import { getFeelingFeed } from "../lib/api";
import { SleeveFan } from "../components/ui/SleeveFan";
import { PipelineGallery } from "../components/ui/PipelineGallery";

/* ── §4F: the arc-wheel scroll. Each card swings in along an arc about a
   far-below pivot as it scrolls to centre, like spokes passing, and the
   three settle into a slight stack. Scrub-tied, cheap, static under
   prefers-reduced-motion. ── */

function ArcCard({ index, children }: { index: number; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = usePrefersReducedMotion();

  useGSAP(
    () => {
      const el = ref.current;
      if (!el) return;
      if (reduced) {
        gsap.set(el, { autoAlpha: 1, y: 0, rotation: 0, scale: 1 });
        return;
      }
      gsap.fromTo(
        el,
        {
          autoAlpha: 0,
          y: 52,
          rotation: index % 2 === 0 ? -4 : 4,
          scale: 0.97,
          transformOrigin: "50% 140%",
        },
        {
          autoAlpha: 1,
          y: 0,
          rotation: 0,
          scale: 1,
          duration: 0.95,
          delay: index * 0.12,
          ease: "power3.out",
          scrollTrigger: { trigger: el, start: "top 88%" },
        },
      );
    },
    { scope: ref, dependencies: [reduced, index] },
  );

  return (
    <div
      ref={ref}
      className="glass relative rounded-sm p-8 shadow-[0_8px_24px_rgba(0,0,0,0.2)] md:p-12"
      style={{ zIndex: index + 1, marginTop: index === 0 ? 0 : "-1.25rem" }}
    >
      {children}
    </div>
  );
}

/* ── §4D pipeline: what happens to a request, in plain words ── */

const PIPELINE = [
  {
    numeral: "I",
    title: "Reading the feeling",
    body: "Type it in English or Hindi, or say it in English. One model reads what the words say, another listens to how you sound, and together they settle on the feeling. Not one of five moods. One of fifteen.",
  },
  {
    numeral: "II",
    title: "The matching brain",
    body: "That feeling becomes a target. 1.2 million songs are measured against it, and the closest ones surface. Not by genre, not by what other people played, by how the music itself behaves.",
  },
  {
    numeral: "III",
    title: "Reasons, shown",
    body: "Every pick can tell you why it made the cut, in plain words, drawn from the song itself. If you want the deeper readout, it is one tap away.",
  },
  {
    numeral: "IV",
    title: "Planned journeys",
    body: "Say where you are and where you want to land. Aether plans the emotional stops between, builds the route in music, then walks you through what it did.",
  },
  {
    numeral: "V",
    title: "Live drift",
    body: "While music plays, it keeps listening. When your mood moves, the next song arrives in a key and tempo that fit, mixed in the way a good DJ would.",
  },
] as const;

/* ── §4H quotes ───────────────────────────────────────────
   Written to read like real notes. When approved submissions start arriving
   through /contact, they replace this list. No moderation flow yet. */

export const QUOTES = [
  "it told me why it picked each song. didn't know i wanted that until i saw it",
  "typed how my evening felt in hindi and it just got it",
  "the mix changed before i finished saying my mood changed",
  "sad was never enough. fifteen feelings is the right amount",
  "watched it plan a route from anxious to calm. quietly beautiful",
];

/* ═════════════════════════════════════════════════════════ */

export default function Home() {
  const { appReady } = useAppState();
  const reduced = usePrefersReducedMotion();
  const heroRef = useRef<HTMLDivElement>(null);
  const introRef = useRef<HTMLElement>(null);
  const squeezeRef = useRef<HTMLElement>(null);
  const [hovered, setHovered] = useState<EmotionLabel | null>(null);
  const [mobileOpen, setMobileOpen] = useState<EmotionLabel | null>(null);
  /* §11.1: paint the type first; the hero art joins one idle beat later. */
  const artReady = useIdleMounted(260);

  /* §4H moment + feeling cards. Feelings come as words only; songs are
     never attached to another person's feeling. */
  const [feelWords, setFeelWords] = useState<string[]>([]);
  const [moments, setMoments] = useState<
    Array<{ emotion: string; topTrackTitles: string[] }>
  >([]);
  useEffect(() => {
    getFeelingFeed()
      .then((items) =>
        setFeelWords(
          items
            .map((i) => String(i.emotion ?? "").trim().toLowerCase())
            .filter(Boolean)
            .slice(0, 6),
        ),
      )
      .catch(() => undefined);
    try {
      const raw = JSON.parse(
        localStorage.getItem("aether.moments.v1") ?? "[]",
      ) as Array<{ emotion?: string; topTrackTitles?: string[] }>;
      setMoments(
        raw
          .filter((m) => m.emotion && Array.isArray(m.topTrackTitles))
          .slice(0, 4)
          .map((m) => ({
            emotion: String(m.emotion),
            topTrackTitles: (m.topTrackTitles as string[]).slice(0, 3),
          })),
      );
    } catch {
      /* quotes carry the strip */
    }
  }, []);

  /* 4A: settle entrance, gated on the preloader finishing. */
  useGSAP(
    () => {
      const els = heroRef.current?.querySelectorAll("[data-hero]");
      if (!els?.length) return;
      if (!appReady) {
        gsap.set(els, { autoAlpha: 0 });
        return;
      }
      if (reduced) {
        gsap.set(els, { autoAlpha: 1, y: 0 });
        return;
      }
      gsap.fromTo(
        els,
        { y: 56, autoAlpha: 0 },
        { y: 0, autoAlpha: 1, duration: 1.15, ease: "expo.out", stagger: 0.09 },
      );
    },
    { scope: heroRef, dependencies: [appReady, reduced] },
  );

  /* 4B: diagonal blue glare sweep, tied to scroll progress. */
  useGSAP(
    () => {
      if (reduced || !introRef.current) return;
      const glare = introRef.current.querySelector("[data-glare]");
      if (!glare) return;
      gsap.fromTo(
        glare,
        { xPercent: -180 },
        {
          xPercent: 320,
          ease: "none",
          scrollTrigger: {
            trigger: introRef.current,
            start: "top 90%",
            end: "bottom 20%",
            scrub: 0.6,
          },
        },
      );
    },
    { scope: introRef, dependencies: [reduced] },
  );

  /* 4C: the squeeze, with its label riding the same scroll progress. */
  useGSAP(
    () => {
      if (reduced || !squeezeRef.current) return;
      const words = squeezeRef.current.querySelectorAll("[data-squeeze]");
      const label = squeezeRef.current.querySelector("[data-squeeze-label]");
      const st = {
        trigger: squeezeRef.current,
        start: "top 85%",
        end: "center 45%",
        scrub: 0.7,
      };
      gsap.fromTo(
        words[0],
        { xPercent: -26, scaleX: 1.08 },
        { xPercent: 0, scaleX: 1, ease: "none", scrollTrigger: st },
      );
      gsap.fromTo(
        words[1],
        { xPercent: 26, scaleX: 1.08 },
        { xPercent: 0, scaleX: 1, ease: "none", scrollTrigger: st },
      );
      if (label) {
        gsap.fromTo(
          label,
          { autoAlpha: 0.15, letterSpacing: "0.7em" },
          { autoAlpha: 1, letterSpacing: "0.22em", ease: "none", scrollTrigger: st },
        );
      }
    },
    { scope: squeezeRef, dependencies: [reduced] },
  );

  const active = hovered ?? null;

  return (
    <>
      {/* ══ 4A — HERO ══════════════════════════════════════ */}
      <section
        ref={heroRef}
        className="relative grid min-h-screen content-center gap-10 px-6 pt-24 md:grid-cols-[1.25fr_1fr] md:items-center md:px-10"
      >
        <div className="min-w-0">
        <div data-hero>
          <ParenLabel accent>EMOTION-AWARE MUSIC INTELLIGENCE</ParenLabel>
        </div>
        <h1
          data-hero
          className="chrome-text display mt-6 text-[clamp(4.5rem,13vw,9.5rem)]"
        >
          {SITE.wordmark}
        </h1>
        <p data-hero className="serif-accent mt-4 text-2xl text-paper/75 md:text-3xl">
          {SITE.tagline}
        </p>
        <p data-hero className="mt-8 max-w-xl text-base leading-relaxed text-mist">
          Type it in English or Hindi. Say it out loud in English. Fifteen
          feelings in, an explained playlist out.
        </p>
        <div data-hero className="mt-14">
          <Link
            to="/curate"
            className="mono-meta border-b border-paper/40 pb-1 text-paper transition-colors hover:border-blue hover:text-blue"
          >
            start curating →
          </Link>
        </div>
        </div>
        {/* §4A: the fan of generated sleeves. Below the CTA at low opacity on
            mobile; beside the type with real air on desktop. */}
        <div data-hero className="mx-auto h-72 w-full max-w-md opacity-75 md:h-[32rem] md:max-w-xl md:justify-self-end md:opacity-100">
          {artReady && (
            <div className="fade-in h-full w-full">
              <SleeveFan />
            </div>
          )}
        </div>
      </section>

      {/* ══ 4B — INTRO + GLARE ═════════════════════════════ */}
      <section
        ref={introRef}
        className="relative overflow-hidden border-t hairline px-6 py-28 md:px-10 md:py-40"
      >
        <Reveal>
          <ParenLabel>IN ONE BREATH</ParenLabel>
          <p className="mt-8 max-w-4xl text-3xl leading-snug text-paper/90 md:text-4xl">
            Aether reads the feeling in your <span className="serif-accent">words</span>{" "}
            and your <span className="serif-accent">voice</span>, matches it against
            1.2 million songs, and shows its reasoning for every track it picks.
            Not a mood preset. A model of how you actually feel.
          </p>
        </Reveal>
        <div
          data-glare
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-[-30%] left-0 w-[45%] -skew-x-[18deg]"
          style={{
            background:
              "linear-gradient(90deg, transparent, rgba(46,107,255,0.14), transparent)",
          }}
        />
      </section>

      {/* ══ 4C — THE SQUEEZE ═══════════════════════════════ */}
      <section
        ref={squeezeRef}
        className="overflow-hidden border-t hairline px-6 py-32 text-center md:px-10 md:py-48"
      >
        <span data-squeeze-label className="mono-meta inline-block text-paper/45">
          (WHY AETHER)
        </span>
        <div className="mt-8 flex flex-wrap items-baseline justify-center gap-x-[0.35em]">
          <span data-squeeze className="display inline-block text-[clamp(3.5rem,11vw,8rem)] text-paper">
            Feel
          </span>
          <span data-squeeze className="display inline-block text-[clamp(3.5rem,11vw,8rem)] text-blue">
            Heard
          </span>
        </div>
      </section>

      {/* ══ 4D — THE PIPELINE ══════════════════════════════ */}
      <section className="border-t hairline px-6 py-28 md:px-10 md:py-40">
        <Reveal>
          <ParenLabel>HOW IT WORKS</ParenLabel>
          <h2 className="display mt-5 max-w-3xl text-4xl text-paper md:text-6xl">
            One request, five systems
          </h2>
        </Reveal>
        <div className="mt-8">
          <PipelineGallery steps={PIPELINE} />
        </div>
      </section>

      {/* ══ 4E — THE 15 EMOTIONS ═══════════════════ */}
      <section className="border-t hairline px-6 py-28 md:px-10 md:py-40">
        <Reveal>
          <ParenLabel>THE TAXONOMY</ParenLabel>
          <h2 className="display mt-5 text-4xl text-paper md:text-6xl">
            Fifteen emotions,
            <br />
            not two moods
          </h2>
        </Reveal>
        <div className="mt-16 grid gap-12 md:grid-cols-[1.4fr_1fr]">
          {/* the list: hover on desktop, tap-to-open inline on touch (§2) */}
          <ul className="flex flex-col" onMouseLeave={() => setHovered(null)}>
            {AETHER_EMOTIONS.map((emotion) => {
              const openInline = mobileOpen === emotion;
              return (
                <li key={emotion}>
                  <button
                    type="button"
                    onMouseEnter={() => setHovered(emotion)}
                    onFocus={() => setHovered(emotion)}
                    onClick={() =>
                      setMobileOpen((v) => (v === emotion ? null : emotion))
                    }
                    aria-expanded={openInline}
                    className={`display block w-fit py-1 text-left text-3xl transition-all duration-300 md:text-5xl ${
                      active === emotion || openInline
                        ? "glow-blue translate-x-2 text-paper"
                        : "text-paper/30 hover:text-paper"
                    }`}
                  >
                    {emotion}
                  </button>
                  {/* mobile inline panel */}
                  <div
                    className="grid transition-[grid-template-rows] duration-500 ease-out md:hidden"
                    style={{ gridTemplateRows: openInline ? "1fr" : "0fr" }}
                  >
                    <div className="overflow-hidden">
                      <div className="glass mb-3 mt-2 rounded-sm p-4">
                        <div className="h-20">
                          <EmotionSpectrum emotion={emotion} />
                        </div>
                        <p className="mono-meta mt-3 text-paper/50">
                          ({EMOTION_NOTES[emotion].toUpperCase()})
                        </p>
                        <p className="mt-2 text-sm leading-relaxed text-mist">
                          {EMOTION_LINES[emotion]}
                        </p>
                      </div>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>

          {/* the readout: centred against the list, sticky as it scrolls (§4E.1) */}
          <div className="hidden md:flex md:flex-col">
            <div className="glass-liquid sticky top-[calc(50vh-14rem)] my-auto flex min-h-[26rem] flex-col justify-between rounded-sm p-8">
              {active ? (
                <>
                  <div className="h-36">
                    <EmotionSpectrum emotion={active} />
                  </div>
                  <div className="mt-6">
                    <div className="flex items-baseline justify-between gap-4">
                      <p className="serif-accent text-4xl text-paper">{active}</p>
                      <p className="mono-meta text-paper/50">
                        ({EMOTION_NOTES[active].toUpperCase()})
                      </p>
                    </div>
                    <p className="mt-4 text-sm leading-relaxed text-mist">
                      {EMOTION_LINES[active]}
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <div className="h-36 opacity-40">
                    <EmotionSpectrum emotion="calm" />
                  </div>
                  <p className="mt-6 text-sm leading-relaxed text-mist">
                    Hover a feeling. Each one moves differently, and each one
                    is a different target the library gets measured against.
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ══ 4F — THREE WAYS IN ═════════════════════════════ */}
      <section className="border-t hairline px-6 py-28 md:px-10 md:py-40">
        <Reveal>
          <ParenLabel>THREE WAYS IN</ParenLabel>
        </Reveal>
        <div className="mt-14 flex flex-col">
          <ArcCard index={0}>
            <div className="grid items-center gap-10 md:grid-cols-2">
              <div>
                <ParenLabel accent>01 · THE CORE</ParenLabel>
                <h3 className="display mt-4 text-3xl text-paper md:text-5xl">Curate</h3>
                <p className="mt-5 max-w-md text-sm leading-relaxed text-mist">
                  Say how you feel, in your own words or your own voice. Get
                  back a sequenced playlist where every track carries its
                  reasons.
                </p>
                <Link
                  to="/curate"
                  className="mono-meta mt-6 inline-block border-b border-paper/30 pb-0.5 text-paper/80 transition-colors hover:border-blue hover:text-paper"
                >
                  open curate →
                </Link>
              </div>
              <SpectrumArt variant="curate" className="h-40 w-full md:h-56" />
            </div>
          </ArcCard>

          <ArcCard index={1}>
            <div className="grid items-center gap-10 md:grid-cols-2">
              <SpectrumArt
                variant="journey"
                className="order-last h-40 w-full md:order-first md:h-56"
              />
              <div>
                <ParenLabel accent>02 · THE ROUTE</ParenLabel>
                <h3 className="display mt-4 text-3xl text-paper md:text-5xl">Journey</h3>
                <p className="mt-5 max-w-md text-sm leading-relaxed text-mist">
                  Wind me down after a stressful day. Aether plans the
                  emotional stops, builds the route in music, then shows you
                  how it thought about it.
                </p>
                <Link
                  to="/journey"
                  className="mono-meta mt-6 inline-block border-b border-paper/30 pb-0.5 text-paper/80 transition-colors hover:border-blue hover:text-paper"
                >
                  open journey →
                </Link>
              </div>
            </div>
          </ArcCard>

          <ArcCard index={2}>
            <div className="grid items-center gap-10 md:grid-cols-2">
              <div>
                <ParenLabel accent>03 · THE PLAYER</ParenLabel>
                <h3 className="display mt-4 text-3xl text-paper md:text-5xl">Live</h3>
                <p className="mt-5 max-w-md text-sm leading-relaxed text-mist">
                  Music that notices. Keep talking while it plays, and when
                  your mood drifts, it mixes into a song that fits the new
                  one, in key and on beat.
                </p>
                <Link
                  to="/live"
                  className="mono-meta mt-6 inline-block border-b border-paper/30 pb-0.5 text-paper/80 transition-colors hover:border-blue hover:text-paper"
                >
                  open live →
                </Link>
              </div>
              <SpectrumArt variant="live" className="h-40 w-full md:h-56" />
            </div>
          </ArcCard>
        </div>
      </section>

      {/* ══ 4G — ABOUT ARNAV ═══════════════════════════════ */}
      <section className="border-t hairline px-6 py-28 md:px-10 md:py-40">
        <Reveal>
          <ParenLabel>THE BUILDER</ParenLabel>
          <div className="mt-8 grid gap-12 md:grid-cols-2 md:items-center">
            <div>
              <p className="serif-accent max-w-xl text-2xl leading-snug text-paper/85 md:text-3xl">
                Built by one student who wanted software to listen better.
              </p>
              <p className="mt-8 max-w-lg text-sm leading-relaxed text-mist">
                Arnav Shukla is a final-year computer science and data science
                student who works on machine learning and AI systems. Aether
                is his answer to a simple gap: nothing lets you describe a
                complicated feeling, anxious but hopeful, nostalgic at 1am,
                and get precisely matching music back. So he built the whole
                chain himself: models that read feeling in text and voice, a
                library of 1.2 million songs to match against, reasons for
                every pick, a planner for emotional routes, and a transition
                engine that mixes in key.
              </p>
              <div className="mt-8 flex flex-wrap gap-x-6 gap-y-2">
                <a
                  href={LINKS.github}
                  target="_blank"
                  rel="noreferrer"
                  className="mono-meta border-b border-paper/30 pb-0.5 text-paper/70 transition-colors hover:border-blue hover:text-paper"
                >
                  github ↗
                </a>
                <a
                  href={LINKS.linkedin}
                  target="_blank"
                  rel="noreferrer"
                  className="mono-meta border-b border-paper/30 pb-0.5 text-paper/70 transition-colors hover:border-blue hover:text-paper"
                >
                  linkedin ↗
                </a>
                <Link
                  to="/connect"
                  className="mono-meta border-b border-paper/30 pb-0.5 text-paper/70 transition-colors hover:border-blue hover:text-paper"
                >
                  connect →
                </Link>
              </div>
            </div>
            <div className="glass rounded-sm p-6">
              {/* fifteen nodes, one per emotion: a feeling finding its
                  neighbours (§4G). literally what the project is. */}
              <ConstellationArt className="h-72 w-full md:h-96" />
            </div>
          </div>
        </Reveal>
      </section>

      {/* ══ 4H — WHAT IT FEELS LIKE ══════════════════ */}
      <section className="border-t hairline py-24 md:py-32">
        <div className="px-6 md:px-10">
          <ParenLabel>WHAT IT FEELS LIKE</ParenLabel>
        </div>
        <div className="mt-10">
          <Marquee>
            {QUOTES.map((q, i) => (
              <span key={q} className="flex shrink-0 gap-6">
                <figure className="glass flex w-[30rem] shrink-0 flex-col justify-center rounded-sm p-9 md:w-[38rem]">
                  <blockquote className="serif-accent text-xl leading-snug text-paper/80 md:text-2xl">
                    "{q}"
                  </blockquote>
                </figure>
                {/* a real feeling from the feed: the word only, never songs
                    joined to someone else's feeling (§4H) */}
                {feelWords[i] && (
                  <figure className="glass flex w-[22rem] shrink-0 flex-col justify-between rounded-sm p-9">
                    <span className="mono-meta text-paper/40">(FELT HERE, RECENTLY)</span>
                    <span className="serif-accent mt-5 text-5xl text-paper/85">
                      {feelWords[i]}
                    </span>
                  </figure>
                )}
                {/* this browser's own history: feeling + its own top tracks */}
                {moments[i] && (
                  <figure className="glass flex w-[36rem] shrink-0 flex-col justify-between rounded-sm p-9">
                    <span className="mono-meta text-blue">
                      ({moments[i].emotion.toUpperCase()})
                    </span>
                    <span className="mt-5 text-base leading-relaxed text-paper/70">
                      {moments[i].topTrackTitles.join(" · ")}
                    </span>
                  </figure>
                )}
              </span>
            ))}
          </Marquee>
        </div>
      </section>
    </>
  );
}
