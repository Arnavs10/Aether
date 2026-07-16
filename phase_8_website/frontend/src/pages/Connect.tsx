/**
 * AETHER — Connect, finished (§10). Bio, skills and reach stay; the
 * (TOLD HONESTLY) and (THE REPO) blocks are gone; the how-it-works
 * paragraph is plain language. All three interactions are real:
 * contact form → POST /contact · feeling ticker → GET /feeling-feed ·
 * feature box → /contact tagged, with the tag folded into the message
 * if the server rejects the field.
 */

import { useEffect, useState } from "react";
import { ApiError, getFeelingFeed, postContact } from "../lib/api";
import type { ContactRequest } from "../lib/types";
import { LINKS } from "../config/site";
import { PageHeader } from "../components/ui/PageScaffold";
import { ParenLabel } from "../components/ui/ParenLabel";
import { Reveal } from "../components/ui/Reveal";
import { Marquee } from "../components/ui/Marquee";
import { BarsLoader } from "../components/ui/BarsLoader";

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

type SendState = "idle" | "sending" | "sent" | "error";

/** Send with the kind tag; if the server rejects the field, fold the tag in. */
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

function ContactForm() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [state, setState] = useState<SendState>("idle");

  const valid = name.trim() && email.includes("@") && message.trim();

  const submit = async () => {
    if (!valid || state === "sending") return;
    setState("sending");
    try {
      await sendContact({ name: name.trim(), email: email.trim(), message: message.trim() });
      setState("sent");
      setMessage("");
    } catch {
      setState("error");
    }
  };

  return (
    <div className="glass rounded-sm p-6 md:p-8">
      <ParenLabel accent>WRITE TO ARNAV</ParenLabel>
      <p className="mt-3 text-sm leading-relaxed text-mist">
        Questions, ideas, internship conversations. It lands in his inbox.
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
          placeholder="what's on your mind…"
          aria-label="Your message"
          rows={4}
          className="resize-none border hairline bg-transparent p-3 text-sm text-paper placeholder:text-paper/25 focus:border-paper/35 focus:outline-none"
        />
      </div>
      <div className="mt-4 flex items-center gap-4" aria-live="polite">
        <button
          type="button"
          onClick={() => void submit()}
          disabled={!valid || state === "sending"}
          className="mono-meta border border-blue px-5 py-2.5 text-paper transition-all hover:[box-shadow:0_0_22px_rgba(46,107,255,0.35)] disabled:border-paper/20 disabled:text-paper/30 disabled:hover:[box-shadow:none]"
        >
          send →
        </button>
        {state === "sending" && <BarsLoader tone="blue" />}
        {state === "sent" && (
          <span className="mono-meta text-gold/80">(SENT. HE READS EVERYTHING)</span>
        )}
        {state === "error" && (
          <span className="text-sm text-mist">it didn't send. try again</span>
        )}
      </div>
    </div>
  );
}

function FeelingTicker() {
  const [feelings, setFeelings] = useState<string[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const items = await getFeelingFeed();
        if (!alive) return;
        setFeelings(
          items
            .map((i) => String(i.emotion ?? "").trim())
            .filter(Boolean)
            .slice(0, 40),
        );
      } catch {
        /* quiet: the ticker just stays in its empty state */
      } finally {
        if (alive) setLoaded(true);
      }
    };
    void load();
    const id = setInterval(load, 45_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="glass rounded-sm p-6 md:p-8">
      <ParenLabel accent>WHAT PEOPLE ARE FEELING</ParenLabel>
      <p className="mt-3 text-sm leading-relaxed text-mist">
        Recent feelings people curated to. Single words, nothing else, fully
        anonymous.
      </p>
      <div className="mt-5 min-h-10">
        {!loaded ? (
          <BarsLoader />
        ) : feelings.length > 0 ? (
          <Marquee duration={26}>
            {feelings.map((f, i) => (
              <span key={`${f}-${i}`} className="mono-meta shrink-0 text-paper/60">
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
    </div>
  );
}

function FeatureBox() {
  const [email, setEmail] = useState("");
  const [idea, setIdea] = useState("");
  const [state, setState] = useState<SendState>("idle");

  const valid = idea.trim() && email.includes("@");

  const submit = async () => {
    if (!valid || state === "sending") return;
    setState("sending");
    try {
      await sendContact({
        name: "feature idea",
        email: email.trim(),
        message: idea.trim(),
        kind: "feature",
      });
      setState("sent");
      setIdea("");
    } catch {
      setState("error");
    }
  };

  return (
    <div className="glass rounded-sm p-6 md:p-8">
      <ParenLabel accent>RECOMMEND A FEATURE</ParenLabel>
      <p className="mt-3 text-sm leading-relaxed text-mist">
        Something Aether should do next. Short is fine.
      </p>
      <div className="mt-5 flex flex-col gap-3">
        <textarea
          value={idea}
          onChange={(e) => setIdea(e.target.value)}
          placeholder="it should…"
          aria-label="Your feature idea"
          rows={3}
          className="resize-none border hairline bg-transparent p-3 text-sm text-paper placeholder:text-paper/25 focus:border-paper/35 focus:outline-none"
        />
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="your email"
          type="email"
          aria-label="Your email"
          className="border hairline bg-transparent p-3 text-sm text-paper placeholder:text-paper/25 focus:border-paper/35 focus:outline-none"
        />
      </div>
      <div className="mt-4 flex items-center gap-4" aria-live="polite">
        <button
          type="button"
          onClick={() => void submit()}
          disabled={!valid || state === "sending"}
          className="mono-meta border hairline px-5 py-2.5 text-paper/80 transition-colors hover:border-gold hover:text-paper disabled:opacity-40"
        >
          suggest →
        </button>
        {state === "sending" && <BarsLoader tone="gold" />}
        {state === "sent" && <span className="mono-meta text-gold/80">(NOTED. THANK YOU)</span>}
        {state === "error" && (
          <span className="text-sm text-mist">it didn't send. try again</span>
        )}
      </div>
    </div>
  );
}

export default function Connect() {
  return (
    <>
      <PageHeader
        eyebrow="PAGE 05 · THE HUMAN"
        title="Connect"
        lede="The person behind the engine, and three ways to talk back."
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
                    One model reads what you write, another listens to how you
                    sound, in English or Hindi, and together they land on one
                    of fifteen feelings. That feeling becomes a target, and
                    1.2 million songs are measured against it. The closest
                    ones surface, arranged so the energy moves deliberately,
                    each with its reasons attached. Ask for a journey and it
                    plans the emotional stops between where you are and where
                    you want to be. Keep listening and it notices when your
                    mood moves, mixing into a song that fits the new one, in
                    key and on beat. Built end to end by one person.
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

      {/* ── the interaction half ─────────────────────────── */}
      <section className="border-t hairline px-6 py-20 md:px-10">
        <ParenLabel>TALK BACK</ParenLabel>
        <div className="mt-10 grid gap-6 lg:grid-cols-3">
          <ContactForm />
          <FeelingTicker />
          <FeatureBox />
        </div>
      </section>
    </>
  );
}
