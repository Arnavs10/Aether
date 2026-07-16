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

import { useRef, useState } from "react";
import { Link } from "react-router";
import { gsap, useGSAP } from "../lib/gsap";
import { useAppState } from "../state/AppState";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { AETHER_EMOTIONS, EMOTION_NOTES, LINKS, SITE } from "../config/site";
import type { EmotionLabel } from "../config/site";
import { ParenLabel } from "../components/ui/ParenLabel";
import { Reveal } from "../components/ui/Reveal";
import { Marquee } from "../components/ui/Marquee";
import { SpectrumArt, EmotionSpectrum } from "../components/ui/SpectrumArt";

/* ── §4D pipeline: what happens to a request, in plain words ── */

const PIPELINE = [
  {
    numeral: "I",
    title: "Reading the feeling",
    body: "You write or you speak, in English or Hindi. One model reads what the words say, another listens to how you sound, and together they settle on the feeling. Not one of five moods. One of fifteen.",
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
        className="relative flex min-h-screen flex-col justify-center px-6 pt-24 md:px-10"
      >
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
          Describe it, typed or spoken, in English or Hindi. Fifteen emotions
          in, an explained playlist out.
        </p>
        <div data-hero className="mt-14">
          <Link
            to="/curate"
            className="mono-meta border-b border-paper/40 pb-1 text-paper transition-colors hover:border-blue hover:text-blue"
          >
            start curating →
          </Link>
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
        <div className="mt-16 flex flex-col">
          {PIPELINE.map((step, i) => (
            <Reveal key={step.numeral} delay={i * 0.04}>
              <div className="grid gap-6 border-t hairline py-10 md:grid-cols-[8rem_1fr] md:gap-10">
                <span className="serif-accent text-5xl text-paper/25 md:text-6xl">
                  {step.numeral}
                </span>
                <div>
                  <h3 className="display text-2xl text-paper md:text-3xl">
                    {step.title}
                  </h3>
                  <p className="mt-4 max-w-2xl text-sm leading-relaxed text-mist">
                    {step.body}
                  </p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ══ 4E — THE 15 EMOTIONS ═══════════════════════════ */}
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
          <ul className="flex flex-col" onMouseLeave={() => setHovered(null)}>
            {AETHER_EMOTIONS.map((emotion) => (
              <li key={emotion}>
                <button
                  type="button"
                  onMouseEnter={() => setHovered(emotion)}
                  onFocus={() => setHovered(emotion)}
                  className={`display block w-fit py-1 text-left text-3xl transition-all duration-300 md:text-5xl ${
                    active === emotion
                      ? "glow-blue translate-x-2 text-paper"
                      : "text-paper/30 hover:text-paper"
                  }`}
                >
                  {emotion}
                </button>
              </li>
            ))}
          </ul>
          {/* the reveal: a spectrum that feels like the emotion */}
          <div className="relative hidden md:block">
            <div className="glass sticky top-28 flex min-h-72 flex-col justify-between rounded-sm p-8">
              {active ? (
                <>
                  <div className="h-36">
                    <EmotionSpectrum emotion={active} />
                  </div>
                  <div className="mt-6 flex items-baseline justify-between gap-4">
                    <p className="serif-accent text-4xl text-paper">{active}</p>
                    <p className="mono-meta text-paper/50">
                      ({EMOTION_NOTES[active].toUpperCase()})
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
        <div className="mt-14 flex flex-col gap-24">
          <Reveal>
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
          </Reveal>

          <Reveal>
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
          </Reveal>

          <Reveal>
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
          </Reveal>
        </div>
      </section>

      {/* ══ 4G — ABOUT ARNAV ═══════════════════════════════ */}
      <section className="border-t hairline px-6 py-28 md:px-10 md:py-40">
        <Reveal>
          <ParenLabel>THE BUILDER</ParenLabel>
          <div className="mt-8 grid gap-12 md:grid-cols-2">
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
              <SpectrumArt variant="wall" className="h-64 w-full md:h-full md:min-h-72" />
            </div>
          </div>
        </Reveal>
      </section>

      {/* ══ 4H — WHAT IT FEELS LIKE ════════════════════════ */}
      <section className="border-t hairline py-24 md:py-32">
        <div className="px-6 md:px-10">
          <ParenLabel>WHAT IT FEELS LIKE</ParenLabel>
        </div>
        <div className="mt-10">
          <Marquee>
            {QUOTES.map((q) => (
              <figure key={q} className="glass w-80 shrink-0 rounded-sm p-6 md:w-96">
                <blockquote className="serif-accent text-lg leading-snug text-paper/80">
                  "{q}"
                </blockquote>
              </figure>
            ))}
          </Marquee>
        </div>
      </section>
    </>
  );
}
