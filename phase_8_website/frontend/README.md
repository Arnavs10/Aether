# AETHER — Phase 8 website (`phase_8_website/frontend/`)

Frontend for the Aether emotion-aware music-intelligence platform. It fronts
the finished FastAPI service in `api/` at the repo root: the 1.2M-track
matching brain, text + voice emotion reading, explained picks, the journey
planner, live drift with harmonic crossfades, and the assistant.

**Stack:** Vite · React 19 · TypeScript · Tailwind v4 · GSAP + ScrollTrigger ·
Lenis. Client-only SPA by design.

**Language truth (everywhere in the UI):** you can **type in English or
Hindi** and **speak in English**. Spoken Hindi is not supported. In free
text you can also **ask for a language** ("some sad hindi songs", "kpop for
studying") and the catalogue follows: Hindi, Punjabi, Tamil, Telugu, Korean,
Japanese, Spanish, French, English.

---

## Run it (two terminals)

```bash
# terminal 1 — the engine, from the repo root:
cd api
export AETHER_STORE=/path/to/music_store.npz
export GROQ_API_KEY=...
export GROQ_MODEL=...
uvicorn app:app --reload            # → http://127.0.0.1:8000

# terminal 2 — the site, from phase_8_website/frontend/:
cp .env.example .env                # points at http://127.0.0.1:8000
npm install
npm run dev                         # → http://localhost:5173
```

| Script              | Does                                          |
| ------------------- | --------------------------------------------- |
| `npm run dev`       | Vite dev server on :5173                      |
| `npm run build`     | typecheck (`tsc --noEmit`) + production build |
| `npm run preview`   | serve the production build locally            |
| `npm run typecheck` | typecheck only                                |

**Env var:** `VITE_API_BASE` only. No secrets ever live in the frontend.

---

## Two kinds of track (the freshness layer)

A playlist blends **store picks** (measured features, match %, live-mixable)
with **fresh picks** sourced live from Apple's current catalogue
(`track_id` starts with `itunes:`; features and match are `null`; always
playable; never live-mixable). The single predicate lives in
`src/lib/tracks.ts → isFreshPick()`. Cards render the two differently on
purpose: fresh picks show `(FRESH)`, never a percentage, never meters,
never a live-mix action.

Tracks arrive already playable (`preview_url`, `cover`, `link`); the
client-side iTunes resolver (`src/lib/itunes.ts`, priority queue with a
burst-then-throttle schedule and viewport boosting) only handles the rare
track that ships without them.

## Playback

One shared player (`src/lib/audio.ts` + `src/components/player/`) serves
all three feature pages: one track at a time app-wide, visible states,
calm unavailable handling. Journey has "play the route" with auto-advance
and jump-to-card. Live runs a **real crossfade** on a drift trigger: two
audio elements, equal-power volume ramp over the engine's duration, capped
against the audio remaining, visuals synced to the actual fade, with a
visual-only fallback when no preview exists.

## Where things live

```
src/
├── config/site.ts        nav · links · flags · canonical 15-emotion fallback
├── lib/                  api client · types · tracks predicate · itunes
│                         resolver · shared audio player · export · gsap
├── state/AppState.tsx    engine status · appReady gate · shared Lenis
├── components/
│   ├── global/           Preloader · SmoothScroll · AmbientWaves · Grain ·
│   │                     ScrollProgress · Nav · Footer · Faq · GoopField ·
│   │                     Chatbot
│   ├── ui/               ParenLabel · Reveal · Marquee · PageScaffold ·
│   │                     SpectrumArt (+EmotionSpectrum · Constellation ·
│   │                     wall) · SleeveFan · VinylDisc · PipelineGallery ·
│   │                     BarsLoader
│   ├── player/           PlayButton · useTrackDelivery
│   └── features/         EmotionChips · VoiceMic · TrackCard · Downloads
└── pages/                Home · Curate · Journey · Live · Connect · 404
```

Notes that matter when editing:

- **Send rule (Curate):** chips → voice reading → text, surfaced live in
  the status line; the textarea dims when chips drive. Never send
  `distribution: []`.
- **`GET /tracks`** field is `name`, not `title`; it powers the Live seed
  picker (shuffle accumulates, filter covers what's loaded only).
- **Chat** always sends history (capped 12); `source` is never rendered;
  a fallback reply is replaced with our own line.
- The footer bottom bar shows the **visitor's** city and live local time:
  one keyless IP lookup (display-only, cached per session), timezone
  fallback, time-only last. Never `navigator.geolocation`.
- Glass (`.glass-liquid`) is applied to exactly three surface groups
  (chatbot, feature control panels + player, the 4E/feed panels) with an
  `@supports` fallback. Keep it off full-width and canvas-adjacent
  surfaces.
- `FLAGS.spotifyLogin` is false and renders nothing.

## State of the build

Pass 4 shipped in full: the §15 order was executed top to bottom with no
cut line. Language sweep, fresh/store rendering, playback on all three
pages with the real Live crossfade, the Curate input truth + mic append +
resolver speed + language surfacing, the merged Connect form with legible
failures, mobile down to 360px, the seed picker, the visitor clock, the
per-emotion spectrum system with its fifteen lines, the constellation, the
denser field, the hero sleeve fan, the FAQ disc, the real marquee, the
designed footer wall, the pinned pipeline gallery, the arc-wheel stack,
and liquid glass. Phase 8.5 (Google Sign-In + history) remains future
scope; `lib/api.ts` stays the single seam for it.
