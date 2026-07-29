# Aether — Website (Phase 8)

The website layer of **Aether**, an emotion-aware music-intelligence platform. Phase 8 is where the finished machine-learning system (Phases 1 to 7) becomes a usable product: a **FastAPI backend** that serves the pipeline over REST, and a **React web app** that people actually interact with.

This README covers the whole website phase, both halves and how they fit together. For the models, the ML pipeline, and the full system architecture, see the [root README](../README.md).

**Live:** https://aether-emotion-music.vercel.app

---

## The two halves

Phase 8 has two parts, which live in two places in the repo:

| Part | Location | What it is |
|------|----------|------------|
| **Backend / API** | [`api/`](../api) (repo root) | FastAPI service that wraps the recommender, RAG, agent, and live engine and exposes them as REST endpoints. |
| **Frontend / Web app** | [`frontend/`](./frontend) (this folder) | React + Vite single-page app that consumes the API and presents the whole experience. |

The frontend is a client-only SPA; all the intelligence lives behind the API, and the two communicate over HTTP.

```
Browser (React / Vite SPA on Vercel)
        │  REST over HTTPS
        ▼
FastAPI service (api/, on an Azure VM)
        │  in-process calls
        ▼
Aether pipeline (Phases 1 to 7: emotion models → fusion → matching → RAG → agent → live engine)
```

---

## Stack

**Backend**

| Concern | Choice |
|---------|--------|
| Framework | FastAPI (Python) |
| Serves | Recommender (Phase 4), RAG (Phase 5), Journey agent (Phase 6), live engine (Phase 7) |
| Hosting | Azure VM |

**Frontend**

| Concern | Choice |
|---------|--------|
| Build tool | Vite |
| Framework | React 19 + TypeScript |
| Styling | Tailwind CSS v4 (CSS-first `@theme`) |
| Motion | GSAP + ScrollTrigger, Lenis (smooth scroll) |
| Routing | React Router (client-side SPA) |
| Hosting | Vercel (SPA rewrite via `frontend/vercel.json`) |

---

## Pages

- **Home** — the landing experience and product story.
- **Curate** — the core flow: describe how you feel (type in English or Hindi, or speak in English) and get an explained, sequenced playlist.
- **Journey** — hand a starting and target mood to the agent and it plans a multi-stage emotional arc.
- **Live** — a continuously adapting player that senses mood drift and glides between harmonically compatible tracks.
- **Connect** — links and ways to reach the project.
- **AetherBot** — an in-app assistant for questions about the music and the app.

---

## Running the website locally

You run the two halves separately: the API first, then the web app that talks to it.

### 1. Backend (API)

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# the backend needs GROQ_API_KEY set (see the root README for all backend env vars)
uvicorn api.service:app --reload
```

### 2. Frontend (web app)

```bash
cd phase_8_website/frontend
npm install

# point the app at your running API
cp .env.example .env
# then open .env and fill in the values it lists (the API base URL)

npm run dev
```

Open the dev URL Vite prints, and the app will talk to your local API.

---

## Configuration

- **Backend** reads its keys and toggles from environment variables (`GROQ_API_KEY`, the feature-store and freshness settings, and so on). The full list is in the [root README](../README.md).
- **Frontend** reads its settings from `frontend/.env` (copied from `frontend/.env.example`), which holds the base URL of the backend the app calls. That example file is the source of truth for the variables it needs.

---

## Deployment

- **Frontend** is deployed on **Vercel** as a static SPA build. `frontend/vercel.json` holds the single-page rewrite so client-side routes resolve correctly on refresh and deep links. Pushing to `main` triggers a Vercel build of the `frontend/` folder.
- **Backend** runs on an **Azure VM**, serving the FastAPI app the frontend calls.

The two deploy independently: the browser app on Vercel, the API on Azure, talking over HTTPS.

---

## Design

The interface follows a clean, understated, premium design language: a fixed dark palette, a single display typeface, glass surfaces, and hairline detailing, so the focus stays on the music and the emotion behind it. Tailwind v4's CSS-first `@theme` holds the design tokens.

---

## Notes

The website is fully deployed and works end to end on desktop. A refined, fully optimized mobile experience (in particular the Live and Connect pages) is in progress and is the current top priority on the roadmap.

---

*Part of [Aether](../README.md) · emotion-aware music intelligence.*
