"""
Aether — Voice API routes.

Wire into your existing FastAPI app (api/app.py) with two lines:

    from voice_routes import router as voice_router
    app.include_router(voice_router)

Endpoints:
    GET  /voice/warmup     -> starts/reports background model loading (call on page load)
    POST /voice-emotion    -> audio clip -> {emotion, distribution[15], text}
                              feed the result into /curate, /journey, or /live/observe

Error visibility
----------------
`predict()` failures used to collapse into a bare 503 with the cause only in the
response body, which meant a broken mic looked identical to a cold model and the
real reason never reached the server log. Every failure here now prints a full
traceback plus the upload's filename, size and content type, so the cause is
visible in the terminal running uvicorn.
"""
from __future__ import annotations

import os
import tempfile
import traceback
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
    if eng.error:
        # A load failure here is the usual reason /voice-emotion later 503s.
        # Surfacing it at warmup time means the log shows the cause before the
        # user ever taps the mic.
        print(f"[voice] warmup status={status!r} error={eng.error!r}")
    return WarmupOut(status=status, detail=eng.error)


@router.post("/voice-emotion", response_model=VoiceEmotionOut)
async def voice_emotion(audio: UploadFile = File(...)) -> VoiceEmotionOut:
    """Audio -> emotion. Frontend records a clip, POSTs it here, then passes
    the returned `emotion` (+ `text`) to /curate | /journey | /live/observe."""
    eng = get_engine()

    # Report engine state up front: "model never loaded" and "model loaded but
    # inference failed" are different bugs and were previously indistinguishable.
    status = getattr(eng, "status", None)
    if eng.error:
        print(f"[voice] engine reports status={status!r} error={eng.error!r}")

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        raw = await audio.read()
        tmp.write(raw)
        tmp.close()
        print(f"[voice] received {audio.filename!r} "
              f"({audio.content_type}, {len(raw)} bytes) -> {tmp.name}")

        try:
            out = eng.predict(tmp.name)
        except Exception as e:  # noqa: BLE001
            # Print the full traceback: the exception type and the frame it came
            # from are what identify the failure (a decode error, a missing
            # model, a bad tensor shape), and none of that survives the 503.
            traceback.print_exc()
            print(f"[voice] predict FAILED on {tmp.name!r} "
                  f"(engine status={status!r}): {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"voice inference failed: {type(e).__name__}: {e}",
            )

        print(f"[voice] ok -> emotion={out.get('emotion')!r} "
              f"text={str(out.get('text'))[:60]!r}")
        return VoiceEmotionOut(**out)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
