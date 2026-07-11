"""
═══════════════════════════════════════════════════════════════════
AETHER — API · Schemas
═══════════════════════════════════════════════════════════════════
Pydantic request/response models — the typed contract for every endpoint.
Keeping these here (separate from app.py) means the HTTP layer stays thin and
the shapes are validated + self-documenting in the auto-generated OpenAPI docs.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── shared ──
class TrackOut(BaseModel):
    rank: int
    track_id: str
    title: str
    artist: str
    source_emotion: Optional[str] = None
    energy: Optional[float] = None
    valence: Optional[float] = None
    tempo: Optional[float] = None
    match_score: Optional[float] = None
    why: Optional[str] = None
    why_technical: Optional[str] = None


# ── curate (main feature) ──
class CurateRequest(BaseModel):
    emotion: Optional[str] = Field(None, description="one of the 15 Aether emotions")
    distribution: Optional[list[float]] = Field(None, description="15-dim emotion vector")
    text: Optional[str] = Field(None, description="free-text mood, e.g. 'i feel low'")
    length: int = Field(12, ge=1, le=50)
    explain: bool = True


class CurateResponse(BaseModel):
    mood: str
    intensity_label: str
    arc_shape: str
    reason: str
    size: int
    tracks: list[TrackOut]


# ── journey (agent) ──
class JourneyRequest(BaseModel):
    text: str = Field(..., description="e.g. 'take me from anxious to calm'")
    length: int = Field(12, ge=1, le=50)


class JourneyResponse(BaseModel):
    request: str
    start: str
    target: str
    waypoints: list[str]
    direction: str
    summary: str
    size: int
    trace: list[str]
    tracks: list[TrackOut]


# ── live player (fun feature) ──
class LiveStartRequest(BaseModel):
    track_id: str


class LiveStartResponse(BaseModel):
    session_id: str
    track_id: str


class LiveObserveRequest(BaseModel):
    session_id: str
    emotion: Optional[str] = None
    distribution: Optional[list[float]] = None


class DriftOut(BaseModel):
    drifted: bool
    distance: float
    from_emotion: str = Field(..., alias="from")
    to_emotion: str = Field(..., alias="to")

    model_config = {"populate_by_name": True}


class CrossfadeOut(BaseModel):
    out_track_id: str
    in_track_id: str
    duration_s: float
    curve: str
    beats: float


class NextTrackOut(BaseModel):
    track_id: str
    name: str
    artist: str
    camelot: Optional[str] = None
    bpm: float
    emotion: float
    harmonic: float
    combined: float


class LiveObserveResponse(BaseModel):
    triggered: bool
    reason: str
    drift: DriftOut
    next: Optional[NextTrackOut] = None
    crossfade: Optional[CrossfadeOut] = None


# ── catalog / meta ──
class TrackListItem(BaseModel):
    track_id: str
    name: str
    artist: str
    camelot: Optional[str] = None
    bpm: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    tracks: int
    llm: str
