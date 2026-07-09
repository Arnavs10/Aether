"""
Aether — API endpoint tests (FastAPI TestClient).

Run from the repo root:  pytest -q
Boots the app over the sample catalog and exercises every endpoint end to end,
including the multi-step live-player flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _name in ("", "api"):
    _p = _ROOT / _name if _name else _ROOT
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi.testclient import TestClient
from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # triggers lifespan → builds the service once
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["tracks"] > 0


def test_emotions(client):
    r = client.get("/emotions")
    assert r.status_code == 200 and len(r.json()) == 15


def test_tracks(client):
    r = client.get("/tracks")
    assert r.status_code == 200
    items = r.json()
    assert len(items) > 0 and "camelot" in items[0]


def test_curate_by_emotion(client):
    r = client.post("/curate", json={"emotion": "sad", "length": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["size"] == 4
    assert all(t["source_emotion"] == "sad" for t in body["tracks"])
    assert body["tracks"][0]["why"]        # RAG annotation present


def test_curate_by_text(client):
    r = client.post("/curate", json={"text": "i feel really low tonight", "length": 3})
    assert r.status_code == 200 and r.json()["size"] == 3


def test_curate_rejects_bad_emotion(client):
    r = client.post("/curate", json={"emotion": "zzz", "length": 3})
    assert r.status_code == 400


def test_journey(client):
    r = client.post("/journey", json={"text": "from anxious to calm", "length": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["start"] == "anxious" and body["target"] == "calm"
    assert body["waypoints"][0] == "anxious" and body["waypoints"][-1] == "calm"
    assert body["size"] >= 1 and body["summary"]


def test_live_flow(client):
    started = client.post("/live/start", json={"track_id": "sad-1"})
    assert started.status_code == 200
    sid = started.json()["session_id"]

    # baseline reading — warm-up, holds
    r1 = client.post("/live/observe", json={"session_id": sid, "emotion": "sad"})
    assert r1.status_code == 200 and r1.json()["triggered"] is False

    # mood change — should trigger a transition + crossfade
    r2 = client.post("/live/observe", json={"session_id": sid, "emotion": "energetic"})
    assert r2.status_code == 200
    body = r2.json()
    assert body["triggered"] is True
    assert body["next"] is not None
    assert body["crossfade"]["duration_s"] >= 3.0


def test_live_unknown_track(client):
    r = client.post("/live/start", json={"track_id": "does-not-exist"})
    assert r.status_code == 404


def test_live_unknown_session(client):
    r = client.post("/live/observe", json={"session_id": "nope", "emotion": "sad"})
    assert r.status_code == 404
