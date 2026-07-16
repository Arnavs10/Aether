/**
 * AETHER — site-wide configuration.
 * Single place for nav order, creator links, feature flags, and honest
 * platform facts used across sections. No secrets live here (§11).
 */

/** Canonical 15-emotion taxonomy — FALLBACK ONLY.
 *  Pages fetch the live list from `GET /emotions` (§1); this constant exists
 *  so the UI never breaks if that call fails, and to type `EmotionLabel`. */
export const AETHER_EMOTIONS = [
  "happy",
  "sad",
  "angry",
  "calm",
  "anxious",
  "energetic",
  "focused",
  "nostalgic",
  "romantic",
  "melancholic",
  "confident",
  "hopeful",
  "frustrated",
  "lonely",
  "dreamy",
] as const;

export type EmotionLabel = (typeof AETHER_EMOTIONS)[number];

/** Short, honest per-emotion descriptors (from the taxonomy definition). */
export const EMOTION_NOTES: Record<EmotionLabel, string> = {
  happy: "upbeat · feel-good",
  sad: "slow ballads · minor key",
  angry: "heavy · aggressive beats",
  calm: "ambient · lo-fi · soft",
  anxious: "tense · building",
  energetic: "dance · high tempo",
  focused: "minimal · study beats",
  nostalgic: "retro · warm acoustic",
  romantic: "slow jams · tender",
  melancholic: "dark · layered depth",
  confident: "bass-heavy anthems",
  hopeful: "uplifting · major key",
  frustrated: "hard rock · dissonant",
  lonely: "sparse · echo-heavy",
  dreamy: "synth · atmospheric",
};

export const NAV = [
  { label: "Home", to: "/" },
  { label: "Curate", to: "/curate" },
  { label: "Journey", to: "/journey" },
  { label: "Live", to: "/live" },
  { label: "Connect", to: "/connect" },
] as const;

export const LINKS = {
  email: "arnavshuklaforbusiness@gmail.com",
  linkedin: "https://www.linkedin.com/in/arnav-shukla10/",
  github: "https://github.com/Arnavs10",
  repo: "https://github.com/Arnavs10/Aether",
} as const;

export const FLAGS = {
  /** §14 — OAuth is coded server-side but not configured. While false the
   *  nav renders nothing at all; flipping this is the only change needed. */
  spotifyLogin: false,
} as const;

export const SITE = {
  wordmark: "AETHER",
  tagline: "music that knows how you feel.",
  location: "Indore, India",
  timezone: "Asia/Kolkata",
  year: new Date().getFullYear(),
} as const;
