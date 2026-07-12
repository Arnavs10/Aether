"""
Aether — extra web endpoints for the Phase 8 site.

  POST /contact               -> email a suggestion / feature idea to Arnav
  POST /feeling-feed          -> record an anonymized emotion (call after a curate)
  GET  /feeling-feed          -> recent anonymized emotions for the live feed
  GET  /auth/spotify/login    -> redirect to Spotify authorization
  GET  /auth/spotify/callback -> exchange code, detect premium|free tier

Wire into api/app.py (one import + one line):
    from web_routes import router as web_router
    app.include_router(web_router)

Env vars (server-side ONLY — never in the frontend):
  /contact:  GMAIL_ADDRESS, GMAIL_APP_PASSWORD   (+ optional CONTACT_TO)
  Spotify:   SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET,
             SPOTIFY_REDIRECT_URI, FRONTEND_URL

/feeling-feed works with no config. /contact needs the Gmail vars to actually
send (it still saves a local backup if email isn't configured). Spotify needs a
free Developer app before it will run.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import smtplib
import ssl
import threading
import time
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["web"])

_HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────── /contact ──
CONTACT_TO = os.environ.get("CONTACT_TO", "arnavshuklaforbusiness@gmail.com")
_CONTACT_LOG = _HERE / "contact_submissions.jsonl"


class ContactIn(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    email: Optional[str] = Field(default=None, max_length=200)
    name: Optional[str] = Field(default=None, max_length=120)
    kind: str = Field(default="suggestion")        # "suggestion" | "feature" | ...


class ContactOut(BaseModel):
    ok: bool
    detail: str


def _send_email(subject: str, body: str, reply_to: Optional[str]) -> None:
    addr = os.environ.get("GMAIL_ADDRESS")
    pw = os.environ.get("GMAIL_APP_PASSWORD")
    if not addr or not pw:
        raise RuntimeError("email not configured (set GMAIL_ADDRESS + GMAIL_APP_PASSWORD)")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = addr
    msg["To"] = CONTACT_TO
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content(body)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as s:
        s.login(addr, pw)
        s.send_message(msg)


@router.post("/contact", response_model=ContactOut)
def contact(req: ContactIn):
    # Always keep a local backup so a submission is never lost, even if mail fails.
    entry = {
        "ts": time.time(), "kind": req.kind, "name": req.name,
        "email": req.email, "message": req.message,
    }
    try:
        with _CONTACT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass

    subject = f"[Aether · {req.kind}] " + (req.name or req.email or "anonymous")
    body = (
        f"Kind: {req.kind}\nName: {req.name or '-'}\nEmail: {req.email or '-'}\n\n"
        f"{req.message}\n"
    )
    try:
        _send_email(subject, body, reply_to=req.email)
    except Exception as e:  # noqa: BLE001
        # Saved locally; report a soft failure so the UI can still say 'received'.
        raise HTTPException(status_code=503, detail=f"saved locally, email failed: {e}")
    return ContactOut(ok=True, detail="sent")


# ─────────────────────────────────────────────────────── /feeling-feed ──
_FEED_PATH = _HERE / "feeling_feed.json"
_FEED_MAX = 200
_FEED_LOCK = threading.Lock()

try:
    from config import AETHER_EMOTIONS
except Exception:  # noqa: BLE001
    AETHER_EMOTIONS = []


class FeedIn(BaseModel):
    emotion: str


def _feed_load() -> list:
    try:
        return json.loads(_FEED_PATH.read_text())
    except Exception:  # noqa: BLE001
        return []


def _feed_save(items: list) -> None:
    try:
        _FEED_PATH.write_text(json.dumps(items[-_FEED_MAX:]))
    except OSError:
        pass


@router.post("/feeling-feed")
def feeling_feed_add(req: FeedIn):
    """Record one anonymized emotion. The frontend calls this after a successful
    curate. Stores ONLY the emotion label + timestamp — nothing identifying."""
    emo = req.emotion.strip().lower()
    if AETHER_EMOTIONS and emo not in AETHER_EMOTIONS:
        raise HTTPException(status_code=400, detail=f"unknown emotion {req.emotion!r}")
    with _FEED_LOCK:
        items = _feed_load()
        items.append({"emotion": emo, "ts": time.time()})
        _feed_save(items)
    return {"ok": True}


@router.get("/feeling-feed")
def feeling_feed_get(limit: int = 30):
    """Recent anonymized emotions (newest first) for the live 'what people are
    feeling' ticker."""
    limit = max(1, min(limit, _FEED_MAX))
    with _FEED_LOCK:
        items = _feed_load()[-limit:]
    now = time.time()
    out = [
        {"emotion": it["emotion"], "seconds_ago": int(now - it["ts"])}
        for it in reversed(items)
    ]
    return {"count": len(out), "feed": out}


# ─────────────────────────────────────────────────────── Spotify OAuth ──
_SPOTIFY_AUTH = "https://accounts.spotify.com/authorize"
_SPOTIFY_TOKEN = "https://accounts.spotify.com/api/token"  # noqa: S105
_SPOTIFY_ME = "https://api.spotify.com/v1/me"
_SCOPES = "user-read-private user-read-email"


def _spotify_cfg():
    cid = os.environ.get("SPOTIFY_CLIENT_ID")
    sec = os.environ.get("SPOTIFY_CLIENT_SECRET")
    redirect = os.environ.get(
        "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/auth/spotify/callback"
    )
    if not cid or not sec:
        raise RuntimeError("Spotify not configured (set SPOTIFY_CLIENT_ID + SECRET)")
    return cid, sec, redirect


@router.get("/auth/spotify/login")
def spotify_login():
    try:
        cid, _, redirect = _spotify_cfg()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    state = secrets.token_urlsafe(16)
    params = urllib.parse.urlencode({
        "client_id": cid, "response_type": "code", "redirect_uri": redirect,
        "scope": _SCOPES, "state": state,
    })
    return RedirectResponse(f"{_SPOTIFY_AUTH}?{params}")


@router.get("/auth/spotify/callback")
def spotify_callback(code: Optional[str] = None, error: Optional[str] = None,
                     state: Optional[str] = None):
    frontend = os.environ.get("FRONTEND_URL", "http://localhost:5173")
    if error:
        return RedirectResponse(f"{frontend}/?spotify=error")
    if not code:
        raise HTTPException(status_code=400, detail="missing code")
    try:
        cid, sec, redirect = _spotify_cfg()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    data = urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": code, "redirect_uri": redirect,
    }).encode()
    basic = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    token_req = urllib.request.Request(_SPOTIFY_TOKEN, data=data, headers={
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    })
    try:
        with urllib.request.urlopen(token_req, timeout=15) as r:  # noqa: S310
            tok = json.loads(r.read())
        access = tok["access_token"]
        me_req = urllib.request.Request(
            _SPOTIFY_ME, headers={"Authorization": f"Bearer {access}"})
        with urllib.request.urlopen(me_req, timeout=15) as r:  # noqa: S310
            me = json.loads(r.read())
        product = me.get("product", "free")        # 'premium' | 'free'
    except Exception:  # noqa: BLE001
        return RedirectResponse(f"{frontend}/?spotify=error")

    # Tier detection only — tokens are NOT persisted here. Full in-app playback
    # would add secure per-session token storage later (same swap-a-tier logic).
    return RedirectResponse(f"{frontend}/?spotify=ok&tier={product}")
