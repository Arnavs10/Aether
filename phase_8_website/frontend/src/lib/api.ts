/**
 * AETHER — the single typed API client (§10: "keep all API calls in one
 * typed client module"). Every network touch to the FastAPI service goes
 * through here; components never build URLs themselves.
 *
 * Base URL: `VITE_API_BASE` env var, defaulting to the local dev service.
 * Never hardcoded in components; no secrets in the frontend (§11).
 *
 * Phase-8.5 note: keep this module the ONLY seam — adding Google Sign-In +
 * `/me/history` later means new functions here, zero component rewrites.
 */

import { AETHER_EMOTIONS } from "../config/site";
import type {
  CatalogTrack,
  ChatMessage,
  ChatResponse,
  ContactRequest,
  CurateRequest,
  CurateResponse,
  FeelingFeedItem,
  HealthResponse,
  JourneyRequest,
  JourneyResponse,
  LiveObserveRequest,
  LiveObserveResponse,
  LiveStartRequest,
  LiveStartResponse,
  VoiceEmotionResponse,
  VoiceWarmupResponse,
} from "./types";

/** Resolved once; trailing slashes stripped so paths join predictably. */
export const API_BASE: string = (
  import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

/** Structured API failure — carries HTTP status + parsed body when present. */
export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(message: string, status: number, body: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }

  /** True for the §7 "voice models still loading" case. */
  get isWarming(): boolean {
    return this.status === 503;
  }
}

interface RequestOptions {
  method?: "GET" | "POST";
  /** JSON body (mutually exclusive with `form`). */
  json?: unknown;
  /** Multipart body (used by /voice-emotion). */
  form?: FormData;
  /** Per-call timeout; generous defaults because /curate hits a 1.2M store. */
  timeoutMs?: number;
  signal?: AbortSignal;
}

/**
 * Core fetch wrapper: timeout via AbortController, JSON parsing, and a
 * human-readable ApiError extracted from FastAPI's `{detail: …}` shape.
 */
async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { method = "GET", json, form, timeoutMs = 30_000, signal } = opts;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  // Merge an external abort signal (e.g. component unmount) with the timeout.
  signal?.addEventListener("abort", () => controller.abort(), { once: true });

  const headers: Record<string, string> = {};
  let body: BodyInit | undefined;
  if (form) {
    body = form; // browser sets the multipart boundary itself
  } else if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body,
      signal: controller.signal,
    });
  } catch (err) {
    // Network-level failure: engine down, CORS, or timeout abort.
    const aborted = controller.signal.aborted && !signal?.aborted;
    throw new ApiError(
      aborted
        ? `The engine didn't answer in time (${Math.round(timeoutMs / 1000)}s).`
        : "Couldn't reach the Aether engine.",
      0,
      err,
    );
  } finally {
    clearTimeout(timer);
  }

  const text = await res.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }

  if (!res.ok) {
    const detail =
      typeof parsed === "object" && parsed !== null && "detail" in parsed
        ? String((parsed as { detail: unknown }).detail)
        : `Request failed (${res.status}).`;
    throw new ApiError(detail, res.status, parsed);
  }
  return parsed as T;
}

/* ════════════════════════════════════════════════════════
   Endpoints — one function per route (§1 list, complete)
   ════════════════════════════════════════════════════════ */

/** GET /health — used by the cold-start-aware preloader poll (§3). */
export function getHealth(timeoutMs = 2_500): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { timeoutMs });
}

/**
 * GET /emotions — the live 15-label list (§1: fetch, don't hardcode).
 * Normalizes bare-array / {emotions} / {labels} wrappers; falls back to the
 * canonical constant so chips never render empty.
 */
export async function getEmotions(): Promise<string[]> {
  try {
    const raw = await request<unknown>("/emotions", { timeoutMs: 6_000 });
    if (Array.isArray(raw) && raw.every((x) => typeof x === "string")) {
      return raw;
    }
    if (raw && typeof raw === "object") {
      const o = raw as Record<string, unknown>;
      for (const key of ["emotions", "labels"]) {
        const v = o[key];
        if (Array.isArray(v) && v.every((x) => typeof x === "string")) {
          return v as string[];
        }
      }
    }
  } catch {
    /* fall through to the canonical fallback */
  }
  return [...AETHER_EMOTIONS];
}

/** POST /curate — the main feature. Long timeout: it works a 1.2M store. */
export function curate(
  req: CurateRequest,
  signal?: AbortSignal,
): Promise<CurateResponse> {
  return request<CurateResponse>("/curate", {
    method: "POST",
    json: req,
    timeoutMs: 120_000,
    signal,
  });
}

/** POST /journey — the planned-route feature. */
export function journey(
  req: JourneyRequest,
  signal?: AbortSignal,
): Promise<JourneyResponse> {
  return request<JourneyResponse>("/journey", {
    method: "POST",
    json: req,
    timeoutMs: 180_000,
    signal,
  });
}

/** POST /live/start — step 1 of the stateful loop (§4L). */
export function liveStart(req: LiveStartRequest): Promise<LiveStartResponse> {
  return request<LiveStartResponse>("/live/start", {
    method: "POST",
    json: req,
    timeoutMs: 30_000,
  });
}

/** POST /live/observe — step 2; first call = baseline, drift ⇒ crossfade. */
export function liveObserve(
  req: LiveObserveRequest,
): Promise<LiveObserveResponse> {
  return request<LiveObserveResponse>("/live/observe", {
    method: "POST",
    json: req,
    timeoutMs: 30_000,
  });
}

/** POST /voice-emotion — multipart field `audio` (§7). Backend transcodes. */
export function voiceEmotion(
  audio: Blob,
  filename = "aether-voice.webm",
): Promise<VoiceEmotionResponse> {
  const form = new FormData();
  form.append("audio", audio, filename);
  return request<VoiceEmotionResponse>("/voice-emotion", {
    method: "POST",
    form,
    timeoutMs: 90_000,
  });
}

/** GET /voice/warmup — kick + poll background model loading (§7.3). */
export function voiceWarmup(): Promise<VoiceWarmupResponse> {
  return request<VoiceWarmupResponse>("/voice/warmup", { timeoutMs: 8_000 });
}

/** POST /contact — Connect form + feature ideas (tagged via `kind`). */
export function postContact(req: ContactRequest): Promise<unknown> {
  return request<unknown>("/contact", {
    method: "POST",
    json: req,
    timeoutMs: 20_000,
  });
}

/** GET /feeling-feed — recent anonymized emotions (§4M ticker). */
export async function getFeelingFeed(): Promise<FeelingFeedItem[]> {
  const raw = await request<unknown>("/feeling-feed", { timeoutMs: 10_000 });
  if (Array.isArray(raw)) return raw as FeelingFeedItem[];
  if (raw && typeof raw === "object") {
    const v = (raw as Record<string, unknown>)["items"] ??
      (raw as Record<string, unknown>)["feed"];
    if (Array.isArray(v)) return v as FeelingFeedItem[];
  }
  return [];
}

/** POST /feeling-feed — contribute an anonymized emotion after a curate. */
export function postFeelingFeed(emotion: string): Promise<unknown> {
  return request<unknown>("/feeling-feed", {
    method: "POST",
    json: { emotion },
    timeoutMs: 10_000,
  });
}

/** GET /tracks — 50 real randomly sampled seeds with camelot + bpm (§2.3). */
export async function getTracks(): Promise<CatalogTrack[]> {
  const raw = await request<unknown>("/tracks", { timeoutMs: 15_000 });
  if (Array.isArray(raw)) return raw as CatalogTrack[];
  if (raw && typeof raw === "object") {
    const v = (raw as Record<string, unknown>)["tracks"];
    if (Array.isArray(v)) return v as CatalogTrack[];
  }
  return [];
}

/** POST /chat — the assistant (§2.8). ALWAYS pass history for follow-ups. */
export function chat(
  message: string,
  history: ChatMessage[],
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    json: { message, history },
    timeoutMs: 60_000,
  });
}

/**
 * Build a 15-float distribution for a chip blend (§2.4): equal weights at
 * the selected indices, index-aligned to the live GET /emotions order.
 * Returns null unless the result is exactly 15 values with a selection.
 */
export function distributionFor(
  selected: string[],
  order: string[],
): number[] | null {
  if (order.length !== 15 || selected.length === 0) return null;
  const w = 1 / selected.length;
  return order.map((label) => (selected.includes(label) ? w : 0));
}
