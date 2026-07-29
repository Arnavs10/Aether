# Aether — Web App (`phase_8_website/frontend/`)

The frontend for **Aether**, an emotion-aware music-intelligence platform. This is a client-only single-page app that fronts the Aether backend (the FastAPI service in [`api/`](../../api) at the repo root): the 1.2M-track matching brain, text and voice emotion reading, explained picks, the Journey planner, live drift with harmonic crossfades, and the in-app assistant.

For the full system overview, models, and architecture, see the [root README](../../README.md). This document covers the web app only.

**Live:** https://aether-emotion-music.vercel.app

---

## Stack

| Concern | Choice |
|---------|--------|
| Build tool | Vite |
| Framework | React 19 + TypeScript |
| Styling | Tailwind CSS v4 (CSS-first `@theme`) |
| Motion | GSAP + ScrollTrigger, Lenis (smooth scroll) |
| Routing | React Router (client-side SPA) |
| Hosting | Vercel (SPA rewrite via `vercel.json`) |

Client-only SPA by design: all rendering happens in the browser, and the app talks to the backend over the REST API.

---

## Pages

- **Home** — the landing experience and product story.
- **Curate** — the core flow: describe how you feel (type in English or Hindi, or speak in English) and get an explained, sequenced playlist.
- **Journey** — hand a starting and target mood to the agent and it plans a multi-stage emotional arc.
- **Live** — a continuously adapting player that senses mood drift and glides between harmonically compatible tracks.
- **Connect** — links and ways to reach the project.
- **AetherBot** — an in-app assistant for questions about the music and the app.

---

## Getting started

### Prerequisites

- Node.js 18+
- The Aether backend running and reachable (see [`api/`](../../api) and the root README)

### Setup

```bash
# from the repo root
cd phase_8_website/frontend

npm install

# configure the API endpoint the app talks to
cp .env.example .env
# then open .env and fill in the values it lists

npm run dev
```

### Build

```bash
npm run build      # production build → dist/
npm run preview    # preview the production build locally
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the values it lists (the base URL of the Aether backend the app calls). The example file is the source of truth for which variables are needed.

---

## Deployment

Deployed on **Vercel** as a static SPA build. `vercel.json` holds the single-page rewrite so client-side routes resolve correctly on refresh and deep links. Pushing to `main` triggers a Vercel build of this folder.

---

## Design

The interface follows a clean, understated, premium design language: a fixed dark palette, a single display typeface, glass surfaces, and hairline detailing, so the focus stays on the music and the emotion behind it. Tailwind v4's CSS-first `@theme` holds the design tokens.

---

## Notes

The web app is fully deployed and works end to end on desktop. A refined, fully optimized mobile experience (in particular the Live and Connect pages) is in progress and is the current top priority on the roadmap.

---

*Part of [Aether](../../README.md) · emotion-aware music intelligence.*
