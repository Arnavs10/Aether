# AETHER — Phase 8 website (`phase_8_website/frontend/`)

Frontend for the Aether emotion-aware music-intelligence platform. It fronts
the finished FastAPI service in `api/` at the repo root: the 1.2M-track
matching brain, text + voice emotion detection (EN + HI), explained picks,
the journey planner, live drift with harmonic crossfades, and the assistant.

**Stack:** Vite · React 19 · TypeScript · Tailwind v4 · GSAP + ScrollTrigger ·
Lenis. Client-only SPA by design: the heavy elements are all client-side
canvas/scroll work where SSR buys nothing.

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

**Env var:** `VITE_API_BASE`, the API base URL. Never hardcoded in
components; no secrets ever live in the frontend (Groq key, email creds and
any streaming secrets are backend env vars). In production, set
`VITE_API_BASE` in the host's environment settings at build time.

The preloader polls `GET /health` every 2s and enters the moment the engine
answers; cold starts stay honest on screen. If the API stays quiet for 25s a
calm "look around while it wakes" escape appears. Browsing degraded is
silent: a failed feature call shows one quiet line inside that panel, with a
retry.

---

## Where things live

```
src/
├── config/site.ts        nav · links · flags · canonical 15-emotion fallback
├── lib/
│   ├── api.ts            THE single typed client, every endpoint + helpers
│   ├── types.ts          API contract, copied from verified live responses
│   ├── itunes.ts         iTunes resolver: cache + ~19/min queue + JSONP
│   ├── exportPlaylist.ts .m3u8/.json export + client-side provider_ref
│   ├── audio.ts          the one shared preview player
│   └── gsap.ts           central plugin registration
├── state/AppState.tsx    engine status · appReady gate · shared Lenis
├── components/
│   ├── global/           Preloader · SmoothScroll · AmbientWaves · Grain ·
│   │                     ScrollProgress · Nav · Footer · Faq · GoopField ·
│   │                     Chatbot
│   ├── ui/               ParenLabel · Reveal · Marquee · PageScaffold ·
│   │                     SpectrumArt (+EmotionSpectrum) · BarsLoader
│   └── features/         EmotionChips · VoiceMic · TrackCard · Downloads
└── pages/                Home · Curate · Journey · Live · Connect · 404
```

Notes that matter when editing:

- **Request building (Curate/Live):** one chip sends `{emotion}`; a blend
  sends a 15-float `{distribution}` aligned to the live `GET /emotions`
  order; voice passes its own `distribution` straight through; free text
  sends `{text}`. Never send `distribution: []`.
- **`GET /tracks`** returns 50 real sampled seeds (field is `name`, not
  `title`) and powers the Live quick picker.
- **`why_technical`** can be an empty string; the reasoning reveal is built
  from the always-present fields and folds it in only when non-empty.
- **Chat** always sends the running history (capped at 12 turns); the
  `source` field is internal and never rendered.
- `FLAGS.spotifyLogin` is false and the nav renders nothing for it; flipping
  the flag is the only change when credentials exist.

---

## State of the build

**Done and wired to the live engine:** the full design system and global
motion shell (grain, ambient field, scroll bar, Lenis, reduced-motion), the
cold-start-aware preloader (bar-spectrum frame, self-drawing border, scan
sweep), Home 4A–4H including the glare sweep and the per-emotion spectrum
reveal, the footer with the equalizer wall and rewritten FAQ, the rebuilt
interactive bottom field (gather / stiffen / spread physics), and all five
pages built in full: Curate, Journey, Live, Connect, plus the assistant on
every page. Voice input with warm-on-entry runs on every feature page.
Downloads (.m3u8/.json) work anonymously. The hero wordmark treatment is
locked as final.

**Remaining flourishes (one focused pass):** the 4D catalogue's sticky
left visual gallery, and the 4F arc-wheel scroll + stacked-cards reveal.
Everything else from the continuation prompt is in.

**Phase 8.5 (not built, room left):** Google Sign-In + `/me/history`.
`lib/api.ts` stays the single seam, so adding it is new functions there and
zero component rewrites.

**One banked number:** the 87% voice-model accuracy is confirmed by Arnav
but currently unsurfaced (the hero stats and engine column were removed by
design). Say the word if you want it back somewhere, Connect would suit it.
