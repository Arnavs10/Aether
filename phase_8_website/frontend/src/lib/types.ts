/**
 * AETHER — API contract types.
 * Every shape below is copied from VERIFIED live responses against the real
 * 1.204M-track service (continuation prompt §2). No inferred fields remain.
 */

import type { EmotionLabel } from "../config/site";
export type { EmotionLabel };

/* ── shared: TrackOut (§2.4) ────────────────────────────── */

/** One recommended track, exactly as /curate and /journey return it. */
export interface Track {
  rank: number;
  track_id: string;
  title: string;
  artist: string;
  source_emotion: string;
  /** 0–1 normalized store features (tempo is NOT bpm). */
  energy: number;
  valence: number;
  tempo: number;
  match_score: number;
  why: string | null;
  /** Verified: can arrive as an empty string. Fold in only when non-empty. */
  why_technical: string | null;
  /**
   * CLIENT-SIDE ONLY. The API never returns this; the frontend attaches it
   * after the iTunes lookup (preview_url, link, artwork, duration) so the
   * playlist exporter can write playable URLs.
   */
  provider_ref?: Record<string, unknown>;
}

/* ── GET /health (§2.1) ─────────────────────────────────── */
export interface HealthResponse {
  status: string;
  tracks: number;
  llm: string;
}

/* ── GET /tracks (§2.3) — 50 real random samples ────────── */
export interface CatalogTrack {
  track_id: string;
  /** Note: this endpoint uses `name`, unlike TrackOut's `title`. */
  name: string;
  artist: string;
  camelot: string;
  bpm: number;
}

/* ── POST /curate (§2.4) ────────────────────────────────── */
/** Server resolution order: distribution → emotion → text → default calm. */
export interface CurateRequest {
  /** ONE label, e.g. "calm". */
  emotion?: string;
  /** EXACTLY 15 floats, index-aligned to GET /emotions order. Never []. */
  distribution?: number[];
  /** Free text, EN or HI. */
  text?: string;
  /** Playlist length, default 12. */
  length?: number;
  /** Attach per-track reasoning, default true. */
  explain?: boolean;
}

export interface CurateResponse {
  mood: string;
  intensity_label: string;
  arc_shape: string;
  reason: string;
  size: number;
  tracks: Track[];
}

/* ── POST /journey (§2.5) — FLAT response ───────────────── */
export interface JourneyRequest {
  text: string;
  length?: number;
}

export interface JourneyResponse {
  request: string;
  start: string;
  target: string;
  waypoints: string[];
  direction: string;
  summary: string;
  size: number;
  /** Plain strings, e.g. ["perceive", "plan(mw=3)", "act", "reflect(ok)", "explain"]. */
  trace: string[];
  tracks: Track[];
}

/* ── POST /live/start + /live/observe (§2.6–2.7) ────────── */
export interface LiveStartRequest {
  track_id: string;
}
export interface LiveStartResponse {
  session_id: string;
  track_id: string;
}

export interface LiveObserveRequest {
  session_id: string;
  emotion?: string;
  /** Exactly 15 floats if present. Never []. */
  distribution?: number[];
}

/** Always present on every observe response. */
export interface DriftInfo {
  drifted: boolean;
  distance: number;
  from: string;
  to: string;
}

export interface LiveNextTrack {
  camelot: string;
  bpm: number;
  harmonic: number;
  combined: number;
  /** Track identity fields as the backend attaches them. */
  [key: string]: unknown;
}

export interface CrossfadePlan {
  out_track_id: string;
  in_track_id: string;
  duration_s: number;
  curve: string;
  beats: number;
  [key: string]: unknown;
}

export interface LiveObserveResponse {
  /** First observe = baseline (false). A real drift = true. */
  triggered: boolean;
  reason: string;
  drift: DriftInfo;
  next: LiveNextTrack | null;
  crossfade: CrossfadePlan | null;
}

/* ── POST /chat (§2.8) ──────────────────────────────────── */
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
export interface ChatRequest {
  message: string;
  history?: ChatMessage[];
}
export interface ChatResponse {
  reply: string;
  /** Internal only. Never surfaced to the user. */
  source: "groq" | "fallback";
}

/* ── POST /voice-emotion · GET /voice/warmup (§2.9) ─────── */
export interface VoiceEmotionResponse {
  emotion: string;
  distribution: number[]; // 15 floats, index-aligned to GET /emotions
  text: string;
  confidence: number;
  labels: string[];
}

export type WarmupState = "cold" | "loading" | "ready" | "error";
export interface VoiceWarmupResponse {
  status: WarmupState;
  [key: string]: unknown;
}

/* ── POST /contact · /feeling-feed (§2.10) ──────────────── */
export interface ContactRequest {
  name: string;
  email: string;
  message: string;
  /** Feature-idea tag. If the server rejects it, resend with the tag folded
   *  into the message body instead (handled at the call site). */
  kind?: "suggestion" | "feature";
}

export interface FeelingFeedItem {
  emotion: string;
  [key: string]: unknown;
}
