"""
Aether — Voice API routes.

Wire into your existing FastAPI app (api/app.py) with two lines:

    from voice_routes import router as voice_router
    app.include_router(voice_router)

Endpoints:
    GET  /voice/warmup     -> starts/reports background model loading (call on page load)
    POST /voice-emotion    -> audio clip -> {emotion, distribution[15], text}
                              feed the result into /curate, /journey, or /live/observe
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from voice_emotion import get_engine

router = APIRouter(tags=["voice"])


class WarmupOut(BaseModel):
    status: str                      # cold | loading | ready | error
    detail: Optional[str] = None


class VoiceEmotionOut(BaseModel):
    emotion: str
    distribution: list[float]
    text: str
    confidence: float
    labels: list[str]


@router.get("/voice/warmup", response_model=WarmupOut)
def voice_warmup() -> WarmupOut:
    """Kick off (or report) background loading of the voice models.

    The frontend calls this once on first page load so the ~30s model load
    happens invisibly while the user browses. Returns 'ready' once warm.
    """
    eng = get_engine()
    status = eng.warmup_async()
    return WarmupOut(status=status, detail=eng.error)


@router.post("/voice-emotion", response_model=VoiceEmotionOut)
async def voice_emotion(audio: UploadFile = File(...)) -> VoiceEmotionOut:
    """Audio -> emotion. Frontend records a clip, POSTs it here, then passes
    the returned `emotion` (+ `text`) to /curate | /journey | /live/observe."""
    eng = get_engine()
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await audio.read())
        tmp.close()
        try:
            out = eng.predict(tmp.name)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"voice inference failed: {e}")
        return VoiceEmotionOut(**out)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
