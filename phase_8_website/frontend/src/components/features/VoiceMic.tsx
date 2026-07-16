/**
 * AETHER — the mic experience (§12). Warmup starts the moment the component
 * mounts so the models load while the user reads; tapping before ready shows
 * the stepped-bar warming state (the preloader's language in miniature) and
 * starts recording the moment the engine reports ready. Recording goes to
 * POST /voice-emotion as multipart audio; the parent receives the full
 * response (emotion, text, distribution) and decides what to do with it.
 */

import { useEffect, useRef, useState } from "react";
import { ApiError, voiceEmotion, voiceWarmup } from "../../lib/api";
import type { VoiceEmotionResponse, WarmupState } from "../../lib/types";
import { BarsLoader } from "../ui/BarsLoader";

type MicPhase = "idle" | "warming" | "recording" | "processing";

interface Props {
  onResult: (r: VoiceEmotionResponse) => void;
  disabled?: boolean;
}

const MAX_RECORD_MS = 12_000;

export function VoiceMic({ onResult, disabled = false }: Props) {
  const [warm, setWarm] = useState<WarmupState>("cold");
  const [phase, setPhase] = useState<MicPhase>("idle");
  const [note, setNote] = useState<string | null>(null);

  const wantsRecord = useRef(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const stopTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const alive = useRef(true);

  /* Warm on entry, poll lightly until ready or error. */
  useEffect(() => {
    alive.current = true;
    let timer: ReturnType<typeof setTimeout>;
    let polls = 0;

    const poll = async () => {
      if (!alive.current || polls++ > 40) return;
      try {
        const r = await voiceWarmup();
        if (!alive.current) return;
        setWarm(r.status);
        if (r.status === "ready") {
          if (wantsRecord.current) {
            wantsRecord.current = false;
            void beginRecording();
          }
          return;
        }
        if (r.status === "error") return;
        timer = setTimeout(poll, wantsRecord.current ? 2_000 : 4_000);
      } catch {
        if (!alive.current) return;
        timer = setTimeout(poll, 5_000);
      }
    };
    void poll();

    return () => {
      alive.current = false;
      clearTimeout(timer);
      if (stopTimer.current) clearTimeout(stopTimer.current);
      recorder.current?.stream.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
        void submit(new Blob(chunks.current, { type: rec.mimeType || "audio/webm" }));
      };
      rec.start();
      setPhase("recording");
      stopTimer.current = setTimeout(() => stopRecording(), MAX_RECORD_MS);
    } catch {
      setPhase("idle");
      setNote("the mic was blocked. allow microphone access and try again");
    }
  };

  const stopRecording = () => {
    if (stopTimer.current) clearTimeout(stopTimer.current);
    if (recorder.current?.state === "recording") recorder.current.stop();
  };

  const submit = async (blob: Blob) => {
    if (!alive.current) return;
    setPhase("processing");
    try {
      const r = await voiceEmotion(blob);
      if (!alive.current) return;
      setPhase("idle");
      onResult(r);
    } catch (err) {
      if (!alive.current) return;
      setPhase("idle");
      if (err instanceof ApiError && err.isWarming) {
        setWarm("loading");
        setNote("the voice engine is still waking. give it a moment and tap again");
      } else {
        setNote("the voice engine didn't answer. try again");
      }
    }
  };

  const onTap = () => {
    if (disabled || phase === "processing") return;
    setNote(null);
    if (phase === "recording") {
      stopRecording();
      return;
    }
    if (warm !== "ready") {
      wantsRecord.current = true;
      setPhase("warming");
      return;
    }
    void beginRecording();
  };

  const showBars = phase === "warming" || phase === "recording" || phase === "processing";
  const tone = phase === "recording" ? "blue" : phase === "warming" ? "gold" : "silver";
  const label =
    phase === "recording"
      ? "listening… tap to finish"
      : phase === "processing"
        ? "reading the feeling…"
        : phase === "warming"
          ? "waking the voice engine…"
          : warm === "error"
            ? "voice is unavailable right now"
            : null;

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={onTap}
        disabled={disabled || warm === "error"}
        aria-label={phase === "recording" ? "Stop recording" : "Speak how you feel"}
        className={`flex h-12 w-12 items-center justify-center border transition-all duration-200 disabled:opacity-40 ${
          phase === "recording"
            ? "border-blue [box-shadow:0_0_18px_rgba(46,107,255,0.4)]"
            : "hairline hover:border-paper/35"
        }`}
      >
        {showBars ? (
          <BarsLoader tone={tone} />
        ) : (
          <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
            <rect x="9" y="3" width="6" height="11" rx="1" fill="var(--color-paper)" opacity="0.75" />
            <path d="M6 11a6 6 0 0 0 12 0" fill="none" stroke="var(--color-paper)" strokeOpacity="0.75" strokeWidth="1.6" />
            <line x1="12" y1="17" x2="12" y2="21" stroke="var(--color-paper)" strokeOpacity="0.75" strokeWidth="1.6" />
          </svg>
        )}
      </button>
      {(label || note) && (
        <span className={`mono-meta ${note ? "text-ember/90" : "text-paper/50"}`}>
          ({(note ?? label ?? "").toUpperCase()})
        </span>
      )}
    </div>
  );
}
