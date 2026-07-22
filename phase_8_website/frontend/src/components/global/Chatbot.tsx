/**
 * AETHER — the assistant (§11). A frosted-glass circle, bottom right on
 * every page, expanding into a chat panel in the dark editorial system.
 * POST /chat with the running history on every send (that is what lets it
 * resolve "who is that artist?" against the previous turn). The `source`
 * field is internal and never shown; when it says "fallback" we render our
 * own on-brand line instead of the server's raw text (Pass 4 §12).
 */

import { useEffect, useRef, useState } from "react";
import { chat } from "../../lib/api";
import type { ChatMessage } from "../../lib/types";
import { BarsLoader } from "../ui/BarsLoader";

const GREETING: ChatMessage = {
  role: "assistant",
  content:
    "hi, i'm aetherbot. ask me anything about aether, or about music. new releases, old records, how a page works, all fine.",
};

const HISTORY_CAP = 12;

/* §12: the rare-path line, ours, never the server's internals. */
const FALLBACK_REPLY =
  "i can't reach my music brain right now, but i know this site inside out. ask me about curate, journey or live and i'll actually be useful.";

export function Chatbot() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([GREETING]);
  const [draft, setDraft] = useState("");
  const [thinking, setThinking] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Escape closes; focus the input on open.
  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  // Keep the newest message in view.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, thinking, open]);

  const send = async () => {
    const text = draft.trim();
    if (!text || thinking) return;
    setDraft("");
    const history = messages.slice(-HISTORY_CAP);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setThinking(true);
    try {
      const res = await chat(text, history);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.source === "fallback" ? FALLBACK_REPLY : res.reply,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "i didn't catch that, the engine went quiet. try once more" },
      ]);
    } finally {
      setThinking(false);
    }
  };

  return (
    <>
      {/* the circle */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "Close AetherBot" : "Open AetherBot"}
        aria-expanded={open}
        className={`glass-liquid fixed bottom-6 right-6 z-[85] flex h-14 w-14 items-center justify-center rounded-full transition-all duration-300 hover:border-paper/30 ${
          open ? "border-blue [box-shadow:0_0_24px_rgba(46,107,255,0.35)]" : ""
        }`}
      >
        <span className="flex items-end gap-[3px]" aria-hidden="true">
          <span className="h-2.5 w-[3px] bg-silver/80" />
          <span className="h-4 w-[3px] bg-blue" />
          <span className="h-2 w-[3px] bg-silver/80" />
        </span>
      </button>

      {/* the panel */}
      {open && (
        <div
          role="dialog"
          aria-label="AetherBot"
          className="glass-liquid fixed bottom-24 right-6 z-[85] flex max-h-[min(34rem,70vh)] w-[min(24rem,calc(100vw-3rem))] flex-col rounded-sm"
        >
          <div className="flex items-center justify-between border-b hairline px-5 py-4">
            <span className="mono-meta text-paper/70">(AETHERBOT)</span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="mono-meta text-paper/45 transition-colors hover:text-paper"
            >
              CLOSE
            </button>
          </div>

          <div ref={listRef} data-lenis-prevent className="flex-1 overflow-y-auto overscroll-contain px-5 py-4">
            <div className="flex flex-col gap-3">
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={`max-w-[85%] rounded-sm px-3.5 py-2.5 text-sm leading-relaxed ${
                    m.role === "user"
                      ? "self-end border border-blue/50 bg-blue/10 text-paper"
                      : "self-start border hairline bg-paper/[0.03] text-paper/85"
                  }`}
                >
                  {m.content}
                </div>
              ))}
              {thinking && (
                <div className="self-start border hairline bg-paper/[0.03] px-3.5 py-2.5">
                  <BarsLoader tone="blue" />
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 border-t hairline p-3">
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void send();
              }}
              placeholder="ask aetherbot about aether, or about music…"
              aria-label="Message AetherBot"
              className="min-w-0 flex-1 bg-transparent px-2 py-2 text-sm text-paper placeholder:text-paper/25 focus:outline-none"
            />
            <button
              type="button"
              onClick={() => void send()}
              disabled={!draft.trim() || thinking}
              className="mono-meta border border-blue px-4 py-2 text-paper transition-all hover:[box-shadow:0_0_18px_rgba(46,107,255,0.35)] disabled:border-paper/20 disabled:text-paper/30 disabled:hover:[box-shadow:none]"
            >
              send
            </button>
          </div>
        </div>
      )}
    </>
  );
}
