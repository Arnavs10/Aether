/**
 * AETHER — the mic, rebuilt from scratch (Pass 6 §1).
 *
 * The state machine is four states and NOTHING is terminal:
 *   idle → (click) listening → (stop) processing → idle
 * Every await path catches back to idle with one calm line. The mic is
 * COMPLETELY decoupled from warmup: warmup is a single fire-and-forget
 * request per page load (module-level guard — never polled, never
 * re-called on render), and no flag from it ever gates the button. If
 * the engine is still cold at submit time, we retry that same clip once
 * after a short beat, then hand back a usable idle button either way.
 */

import { useEffect, useRef, useState } from "react";
import { ApiError, voiceEmotion, voiceWarmup } from "../../lib/api";
import type { VoiceEmotionResponse } from "../../lib/types";
import { BarsLoader } from "../ui/BarsLoader";

type MicPhase = "idle" | "listening" | "processing";

interface Props {
  onResult: (r: VoiceEmotionResponse) => void;
  disabled?: boolean;
}

const MAX_RECORD_MS = 12_000;
const WARM_RETRY_MS = 2_800;

const supported =
  typeof navigator !== "undefined" &&
  !!navigator.mediaDevices?.getUserMedia &&
  typeof MediaRecorder !== "undefined";

/* §1: ONCE per page load, fire-and-forget, result ignored. */
let warmupFired = false;
function fireWarmupOnce(): void {
  if (warmupFired || !supported) return;
  warmupFired = true;
  voiceWarmup().catch(() => undefined);
}

export function VoiceMic({ onResult, disabled = false }: Props) {
  const [phase, setPhase] = useState<MicPhase>("idle");
  const [note, setNote] = useState<string | null>(null);

  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const stopTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    fireWarmupOnce();
    return () => {
      alive.current = false;
      if (stopTimer.current) clearTimeout(stopTimer.current);
      if (retryTimer.current) clearTimeout(retryTimer.current);
      try {
        recorder.current?.stream.getTracks().forEach((t) => t.stop());
      } catch {
        /* already stopped */
      }
    };
  }, []);

  const toIdle = (message: string | null) => {
    if (!alive.current) return;
    setPhase("idle");
    setNote(message);
  };

  /* ── listening ─────────────────────────────────────────── */
  const beginRecording = async () => {
    setNote(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!alive.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : undefined;
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      recorder.current = rec;
      chunks.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        void submit(new Blob(chunks.current, { type: rec.mimeType || "audio/webm" }), true);
      };
      rec.onerror = () => {
        stream.getTracks().forEach((t) => t.stop());
        toIdle("the mic hit a snag. tap and try again");
      };
      rec.start();
      setPhase("listening");
      stopTimer.current = setTimeout(() => stopRecording(), MAX_RECORD_MS);
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      toIdle(
        name === "NotAllowedError" || name === "SecurityError"
          ? "the mic needs permission. allow it in the browser and tap again"
          : name === "NotFoundError"
            ? "no microphone found on this device"
            : "the mic couldn't start. try again",
      );
    }
  };

  const stopRecording = () => {
    if (stopTimer.current) clearTimeout(stopTimer.current);
    try {
      if (recorder.current?.state === "recording") recorder.current.stop();
      else toIdle(null);
    } catch {
      toIdle("the mic hit a snag. tap and try again");
    }
  };

  /* ── processing (one bounded cold-start retry, then idle) ── */
  const submit = async (blob: Blob, allowRetry: boolean) => {
    if (!alive.current) return;
    setPhase("processing");
    try {
      const r = await voiceEmotion(blob);
      if (!alive.current) return;
      setPhase("idle");
      setNote(null);
      onResult(r);
    } catch (err) {
      if (!alive.current) return;
      if (err instanceof ApiError && err.isWarming && allowRetry) {
        setNote("the voice engine is waking. one more try in a second…");
        retryTimer.current = setTimeout(() => {
          if (alive.current) void submit(blob, false);
        }, WARM_RETRY_MS);
        return;
      }
      toIdle(
        err instanceof ApiError && err.isWarming
          ? "the voice engine is still waking. tap and speak once more"
          : "the voice engine didn't answer. try again",
      );
    }
  };

  const onTap = () => {
    if (disabled || phase === "processing" || !supported) return;
    if (phase === "listening") {
      stopRecording();
      return;
    }
    void beginRecording();
  };

  const busy = phase === "listening" || phase === "processing";
  const label =
    phase === "listening"
      ? "listening… tap to finish"
      : phase === "processing"
        ? "reading the feeling…"
        : !supported
          ? "voice input isn't supported in this browser"
          : null;

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={onTap}
        disabled={disabled || !supported}
        aria-label={phase === "listening" ? "Stop recording" : "Speak how you feel"}
        className={`flex h-12 w-12 items-center justify-center border transition-all duration-200 disabled:opacity-40 ${
          phase === "listening"
            ? "border-blue [box-shadow:0_0_18px_rgba(46,107,255,0.4)]"
            : "hairline hover:border-paper/35"
        }`}
      >
        {busy ? (
          <BarsLoader tone={phase === "listening" ? "blue" : "silver"} />
        ) : (
          <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
            <rect x="9" y="3" width="6" height="11" rx="1" fill="var(--color-paper)" opacity="0.75" />
            <path d="M6 11a6 6 0 0 0 12 0" fill="none" stroke="var(--color-paper)" strokeOpacity="0.75" strokeWidth="1.6" />
            <line x1="12" y1="17" x2="12" y2="21" stroke="var(--color-paper)" strokeOpacity="0.75" strokeWidth="1.6" />
          </svg>
        )}
      </button>
      {(label || note) && (
        <span className={`mono-meta ${note ? "text-ember/90" : "text-paper/50"}`} aria-live="polite">
          ({(note ?? label ?? "").toUpperCase()})
        </span>
      )}
    </div>
  );
}
