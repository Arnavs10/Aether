"""
═══════════════════════════════════════════════════════════════════
AETHER — API · FastAPI App
═══════════════════════════════════════════════════════════════════
The HTTP surface for Aether. Thin by design: every endpoint validates input
(via schemas), calls one AetherService method, and serializes the result.

Run it:
    cd api && uvicorn app:app --reload
    → interactive docs at http://127.0.0.1:8000/docs

Endpoints
    GET  /health                 service + catalog + LLM status
    GET  /emotions               the 15 Aether emotions
    GET  /tracks                 sample catalog (track_id, key, bpm)
    POST /curate                 mood → explained playlist        (Main Feature)
    POST /journey                "anxious→calm" → agentic arc      (Agent)
    POST /live/start             open a live session on a track    (Fun Feature)
    POST /live/observe           feed a mood → hold / crossfade    (Fun Feature)

By default it builds over the in-memory sample catalog. Point at the real 1.2M
store by swapping the constructor in `_build_service()`.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path         

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi import FastAPI, HTTPException                    # noqa: E402

from config import AETHER_EMOTIONS                            # noqa: E402
from service import AetherService                             # noqa: E402
from schemas import (                                         # noqa: E402
    CurateRequest, CurateResponse, JourneyRequest, JourneyResponse,
    LiveStartRequest, LiveStartResponse, LiveObserveRequest,
    LiveObserveResponse, TrackOut, TrackListItem, HealthResponse,
)
from voice_routes import router as voice_router
from fastapi.middleware.cors import CORSMiddleware   # noqa: E402
from web_routes import router as web_router          # noqa: E402
from chat_routes import router as chat_router     # noqa: E40


def _build_service() -> AetherService:
    # Point at the real 1.2M store by setting AETHER_STORE to its .npz path:
    #   export AETHER_STORE=phase_2_music_data/store/music_store.npz
    # Otherwise falls back to the in-memory sample catalog.
    import os
    store_path = os.getenv("AETHER_STORE")
    if store_path:
        return AetherService.from_store_path(store_path)
    return AetherService.from_sample()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.svc = _build_service()      # built once at startup
    # Start loading the voice models now, in the background, so the ~30s cold
    # start happens while the server boots instead of on the user's first mic
    # tap. Failure here is not fatal: the engine records its own error and the
    # voice endpoint reports it, so the rest of the API still comes up.
    try:
        from voice_emotion import get_engine
        get_engine().warmup_async()
        print("[voice] warmup started at server startup")
    except Exception as exc:  # noqa: BLE001
        print(f"[voice] could not start warmup: {type(exc).__name__}: {exc}")
    yield


app = FastAPI(
    title="Aether API",
    version="1.0.0",
    description="Emotion-aware music intelligence — curate, journey, live.",
    lifespan=lifespan,
)
app.include_router(voice_router)
app.include_router(web_router)# ← new
app.include_router(chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # dev: any origin. tighten to your deployed URL later.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _svc() -> AetherService:
    return app.state.svc


# ── serializers ──
def _track_out(t) -> TrackOut:
    """Serialize a Phase 4 Track for the wire.

    `provider_ref` is filled by the delivery layer (iTunes) during enrich() and
    discover(), so by the time a track reaches here the artwork, preview and
    store link have usually already been fetched. Flatten them out rather than
    dropping them: otherwise the client re-searches iTunes for a song the server
    just resolved, which is slow, wastes a rate-limited quota, and can resolve to
    a different recording than the one we picked.
    """
    ref = t.provider_ref or {}
    extra = t.extra or {}
    return TrackOut(
        rank=t.rank, track_id=t.track_id, title=t.title, artist=t.artist,
        source_emotion=t.source_emotion, energy=t.energy, valence=t.valence,
        tempo=t.tempo, match_score=t.match_score,
        why=extra.get("why"),
        why_technical=extra.get("why_technical"),
        year=t.year,
        preview_url=ref.get("preview_url"),
        cover=ref.get("cover"),
        link=ref.get("link"),
        album=ref.get("album"),
    )


# ── meta ──
@app.get("/health", response_model=HealthResponse)
def health():
    svc = _svc()
    return HealthResponse(status="ok", tracks=len(svc.store),
                          llm="on" if svc.llm_fn else "off")


@app.get("/emotions", response_model=list[str])
def emotions():
    return AETHER_EMOTIONS


@app.get("/tracks", response_model=list[TrackListItem])
def tracks():
    return [TrackListItem(**t) for t in _svc().list_tracks()]


# ── MAIN FEATURE ──
@app.post("/curate", response_model=CurateResponse)
def curate(req: CurateRequest):
    try:
        rec = _svc().curate(emotion=req.emotion, distribution=req.distribution,
                            text=req.text, length=req.length, explain=req.explain)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    mood = rec.dominant_emotions[0][0] if rec.dominant_emotions else "unknown"
    return CurateResponse(
        mood=mood, intensity_label=rec.intensity_label, arc_shape=rec.arc_shape,
        reason=rec.reason, size=rec.size,
        tracks=[_track_out(t) for t in rec.tracks],
    )


# ── AGENT ──
@app.post("/journey", response_model=JourneyResponse)
def journey(req: JourneyRequest):
    try:
        res = _svc().journey(req.text, length=req.length)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JourneyResponse(
        request=res.request, start=res.perceived.start,
        target=res.perceived.target, waypoints=res.plan.waypoints,
        direction=res.plan.direction, summary=res.explanation.summary,
        size=res.playlist.size, trace=res.trace,
        tracks=[_track_out(t) for t in res.playlist.tracks],
    )


# ── FUN FEATURE ──
@app.post("/live/start", response_model=LiveStartResponse)
def live_start(req: LiveStartRequest):
    try:
        sid = _svc().live_start(req.track_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return LiveStartResponse(session_id=sid, track_id=req.track_id)


@app.post("/live/observe", response_model=LiveObserveResponse)
def live_observe(req: LiveObserveRequest):
    try:
        d = _svc().live_observe(req.session_id, emotion=req.emotion,
                               distribution=req.distribution)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return LiveObserveResponse(**d.as_dict())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
