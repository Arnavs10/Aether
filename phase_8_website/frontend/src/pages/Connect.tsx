/**
 * AETHER — Connect (Pass 4 §11).
 * §11.1 one form, one send, a small message/feature toggle. The two
 *       identical boxes are gone; the section is two composed columns.
 * §11.2 failures are legible: 4xx reads as "check the address", 5xx and
 *       network own the blame and hand over a live mailto. Status + body
 *       go to console.error for diagnosis.
 * §11.3 the feed panel has substance: an in-view-gated live ticker, an
 *       honest readout derived from the feed, the §4E spectrum at small
 *       size, and a conservative one-line privacy claim.
 * §11.4 the one metric on the whole site, as prose, here only.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getFeelingFeed, postContact } from "../lib/api";
import type { ContactRequest, FeelingFeedItem } from "../lib/types";
import { AETHER_EMOTIONS, LINKS, type EmotionLabel } from "../config/site";
import { PageHeader } from "../components/ui/PageScaffold";
import { ParenLabel } from "../components/ui/ParenLabel";
import { Reveal } from "../components/ui/Reveal";
import { Marquee } from "../components/ui/Marquee";
import { BarsLoader } from "../components/ui/BarsLoader";
import { EmotionSpectrum } from "../components/ui/SpectrumArt";

const SKILLS = [
  "PYTHON",
  "PYTORCH",
  "HUGGINGFACE TRANSFORMERS",
  "LANGGRAPH · LANGCHAIN",
  "RAG",
  "FASTAPI",
  "SCIKIT-LEARN",
  "SQL",
  "JAVA",
] as const;

type SendState =
  | { kind: "idle" }
  | { kind: "sending" }
  | { kind: "sent" }
  | { kind: "client-error" }
  | { kind: "server-error" };

/** Send with the kind tag; fold the tag into the body if it's rejected. */
async function sendContact(req: ContactRequest): Promise<void> {
  try {
    await postContact(req);
  } catch (err) {
    const rejectable =
      err instanceof ApiError && (err.status === 400 || err.status === 422);
    if (rejectable && req.kind) {
      await postContact({
        name: req.name,
        email: req.email,
        message: `[${req.kind} idea] ${req.message}`,
      });
      return;
    }
    throw err;
  }
}

/* ── §11.1 + §11.2 — the one form ───────────────────────── */

function ContactForm() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [asFeature, setAsFeature] = useState(false);
  const [state, setState] = useState<SendState>({ kind: "idle" });

  const valid = name.trim() && email.includes("@") && message.trim();

  const submit = async () => {
    if (!valid || state.kind === "sending") return;
    setState({ kind: "sending" });
    try {
      await sendContact({
        name: name.trim(),
        email: email.trim(),
        message: message.trim(),
        ...(asFeature ? { kind: "feature" as const } : {}),
      });
      setState({ kind: "sent" });
      setMessage("");
    } catch (err) {
      // §11.2: the next failure diagnoses itself in devtools.
      if (err instanceof ApiError) {
        console.error("[contact] send failed", err.status, err.body);
        setState(
          err.status >= 400 && err.status < 500
            ? { kind: "client-error" }
            : { kind: "server-error" },
        );
      } else {
        console.error("[contact] send failed", err);
        setState({ kind: "server-error" });
      }
    }
  };

  return (
    <div className="glass rounded-sm p-6 md:p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <ParenLabel accent>WRITE TO ARNAV</ParenLabel>
        {/* the toggle: a message / a feature idea */}
        <div className="flex border hairline" role="radiogroup" aria-label="What is this">
          {([false, true] as const).map((v) => (
            <button
              key={String(v)}
              type="button"
              role="radio"
              aria-checked={asFeature === v}
              onClick={() => setAsFeature(v)}
              className={`mono-meta px-3 py-2 transition-colors ${
                asFeature === v ? "bg-paper/10 text-paper" : "text-paper/45 hover:text-paper"
              }`}
            >
              {v ? "a feature idea" : "a message"}
            </button>
          ))}
        </div>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-mist">
        {asFeature
          ? "Something Aether should do next. Short is fine."
          : "Questions, ideas, internship conversations. It lands in his inbox."}
      </p>
      <div className="mt-5 flex flex-col gap-3">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="your name"
          aria-label="Your name"
          className="border hairline bg-transparent p-3 text-sm text-paper placeholder:text-paper/25 focus:border-paper/35 focus:outline-none"
        />
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="your email"
          type="email"
          aria-label="Your email"
          className="border hairline bg-transparent p-3 text-sm text-paper placeholder:text-paper/25 focus:border-paper/35 focus:outline-none"
        />
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={asFeature ? "it should…" : "what's on your mind…"}
          aria-label="Your message"
          rows={5}
          className="resize-none border hairline bg-transparent p-3 text-sm text-paper placeholder:text-paper/25 focus:border-paper/35 focus:outline-none"
        />
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-4" aria-live="polite">
        <button
          type="button"
          onClick={() => void submit()}
          disabled={!valid || state.kind === "sending"}
          className="mono-meta border border-blue px-5 py-2.5 text-paper transition-all hover:[box-shadow:0_0_22px_rgba(46,107,255,0.35)] disabled:border-paper/20 disabled:text-paper/30 disabled:hover:[box-shadow:none]"
        >
          send →
        </button>
        {state.kind === "sending" && <BarsLoader tone="blue" />}
        {state.kind === "sent" && (
          <span className="mono-meta text-gold/80">(SENT. HE READS EVERYTHING)</span>
        )}
        {state.kind === "client-error" && (
          <span className="text-sm text-mist">
            something in that didn't go through. check the email address?
          </span>
        )}
        {state.kind === "server-error" && (
          <span className="max-w-md text-sm text-mist">
            the mail didn't go out. it's on my side, not yours. try again in a
            minute, or just email me directly at{" "}
            <a
              href={`mailto:${LINKS.email}`}
              className="border-b border-paper/30 text-paper/75 transition-colors hover:border-blue hover:text-paper"
            >
              {LINKS.email}
            </a>
            .
          </span>
        )}
      </div>
    </div>
  );
}

/* ── §11.3 — the feed panel with substance ──────────────── */

function feedAge(item: FeelingFeedItem): number | null {
  for (const key of ["at", "ts", "time", "timestamp", "created_at"]) {
    const v = item[key];
    if (typeof v === "number" && Number.isFinite(v)) {
      const ms = v < 1e12 ? v * 1000 : v;
      return Date.now() - ms;
    }
    if (typeof v === "string") {
      const t = Date.parse(v);
      if (!Number.isNaN(t)) return Date.now() - t;
    }
  }
  return null;
}

function roughAgo(ms: number): string {
  const m = Math.round(ms / 60_000);
  if (m < 2) return "just now";
  if (m < 60) return `about ${m} minutes ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `about ${h} hour${h === 1 ? "" : "s"} ago`;
  const d = Math.round(h / 24);
  return `about ${d} day${d === 1 ? "" : "s"} ago`;
}

function FeelingFeedPanel() {
  const [items, setItems] = useState<FeelingFeedItem[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [inView, setInView] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    const io = new IntersectionObserver(([entry]) => setInView(entry?.isIntersecting ?? false));
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // Poll only while the section is actually on screen (§11.3).
  useEffect(() => {
    if (!inView) return;
    let alive = true;
    const load = async () => {
      try {
        const raw = await getFeelingFeed();
        if (alive) setItems(raw.filter((i) => String(i.emotion ?? "").trim()));
      } catch {
        /* the panel keeps its last state */
      } finally {
        if (alive) setLoaded(true);
      }
    };
    void load();
    const id = setInterval(load, 30_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [inView]);

  const words = items.map((i) => String(i.emotion).trim().toLowerCase());
  const readout = useMemo(() => {
    if (words.length === 0) return null;
    const counts = new Map<string, number>();
    for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
    let dominant = words[0];
    let max = 0;
    for (const [w, n] of counts) {
      if (n > max) {
        max = n;
        dominant = w;
      }
    }
    const newestAge = feedAge(items[0]);
    return { dominant, dominantCount: max, total: words.length, newest: words[0], newestAge };
  }, [items]); // eslint-disable-line react-hooks/exhaustive-deps

  const dominantIsLabel =
    readout && (AETHER_EMOTIONS as readonly string[]).includes(readout.dominant);

  return (
    <div ref={panelRef} className="glass-liquid flex flex-col rounded-sm p-6 md:p-8">
      <ParenLabel accent>WHAT PEOPLE ARE FEELING</ParenLabel>
      <p className="mt-3 text-sm leading-relaxed text-mist">
        Real feelings people curated to. Single words, nothing else, fully
        anonymous.
      </p>

      <div className="mt-5 min-h-12">
        {!loaded ? (
          <BarsLoader />
        ) : words.length > 0 ? (
          <Marquee duration={24}>
            {words.slice(0, 40).map((f, i) => (
              <span key={`${f}-${i}`} className="shrink-0 font-mono text-sm uppercase tracking-[0.18em] text-paper/60">
                ({f.toUpperCase()})
              </span>
            ))}
          </Marquee>
        ) : (
          <p className="mono-meta text-paper/35">
            (QUIET RIGHT NOW. CURATE SOMETHING AND YOUR FEELING SHOWS UP HERE)
          </p>
        )}
      </div>

      {/* the honest readout */}
      {readout && (
        <div className="mt-6 grid gap-5 border-t hairline pt-6 sm:grid-cols-[1fr_auto]">
          <div className="flex flex-col gap-2 text-sm text-mist">
            <p>
              right now the room leans{" "}
              <span className="serif-accent text-lg text-paper">{readout.dominant}</span>
              , holding {readout.dominantCount} of the last {readout.total} readings.
            </p>
            <p className="mono-meta text-paper/40">
              (LATEST · {readout.newest.toUpperCase()}
              {readout.newestAge !== null ? ` · ${roughAgo(readout.newestAge).toUpperCase()}` : ""})
            </p>
          </div>
          {dominantIsLabel && (
            <div className="h-16 w-32">
              <EmotionSpectrum emotion={readout.dominant as EmotionLabel} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ═════════════════════════════════════════════════════════ */

export default function Connect() {
  return (
    <>
      <PageHeader
        eyebrow="PAGE 05 · THE HUMAN"
        title="Connect"
        lede="The person behind the engine, and two ways to talk back."
      />

      {/* ── the Arnav half ─────────────────────────────── */}
      <section className="px-6 pb-24 md:px-10">
        <Reveal>
          <div className="grid gap-12 md:grid-cols-[1.2fr_1fr]">
            <div>
              <p className="serif-accent text-2xl leading-snug text-paper/85 md:text-3xl">
                Arnav Shukla, final-year computer science and data science
                student, working in machine learning, RAG and agentic AI.
              </p>
              <div className="mt-12 flex flex-col gap-10">
                <div>
                  <ParenLabel>THE GAP</ParenLabel>
                  <p className="mt-3 max-w-xl text-sm leading-relaxed text-mist">
                    No platform lets you describe a complicated emotional
                    state, anxious but hopeful, nostalgic at 1am, and get
                    precisely matching music back. Mood playlists flatten
                    feeling into five presets. Aether was built to take the
                    feeling seriously.
                  </p>
                </div>
                <div>
                  <ParenLabel>HOW IT WORKS, PLAINLY</ParenLabel>
                  <p className="mt-3 max-w-xl text-sm leading-relaxed text-mist">
                    One model reads what you write, in English or Hindi.
                    Another listens to how you sound when you speak in English.
                    Together they land on one of fifteen feelings. That feeling
                    becomes a target, and 1.2 million songs are measured
                    against it. The closest ones surface, arranged so the
                    energy moves deliberately, each with its reasons attached.
                    Ask for a language in your words, hindi, punjabi, korean,
                    and the picks follow. Keep listening and it notices when
                    your mood moves, mixing into a song that fits the new one,
                    in key and on beat. Built end to end by one person.
                  </p>
                  <p className="mt-3 max-w-xl text-sm leading-relaxed text-mist">
                    Trained on roughly 8,900 labelled clips, the voice side
                    reads tone at 87% accuracy in validation.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-10">
              <div>
                <ParenLabel>WORKS IN</ParenLabel>
                <div className="mt-4 flex flex-wrap gap-2">
                  {SKILLS.map((s) => (
                    <span
                      key={s}
                      className="mono-meta rounded-full border hairline px-3 py-1.5 text-paper/60"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <ParenLabel>REACH</ParenLabel>
                <div className="mt-4 flex flex-col gap-2">
                  <a
                    href={`mailto:${LINKS.email}`}
                    className="w-fit text-sm text-paper/75 transition-colors hover:text-paper"
                  >
                    {LINKS.email}
                  </a>
                  <a
                    href={LINKS.linkedin}
                    target="_blank"
                    rel="noreferrer"
                    className="w-fit text-sm text-paper/75 transition-colors hover:text-paper"
                  >
                    LinkedIn ↗
                  </a>
                  <a
                    href={LINKS.repo}
                    target="_blank"
                    rel="noreferrer"
                    className="w-fit text-sm text-paper/75 transition-colors hover:text-paper"
                  >
                    GitHub ↗
                  </a>
                </div>
              </div>
            </div>
          </div>
        </Reveal>
      </section>

      {/* ── the interaction half: two composed columns ───── */}
      <section id="talk-back" className="scroll-mt-28 border-t hairline px-6 py-20 md:px-10">
        <ParenLabel>TALK BACK</ParenLabel>
        <div className="mt-10 grid gap-6 lg:grid-cols-2">
          <ContactForm />
          <FeelingFeedPanel />
        </div>
      </section>
    </>
  );
}
