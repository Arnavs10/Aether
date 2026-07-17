"""
Aether — Chatbot endpoint (POST /chat).

The glass-circle assistant's brain. It:
  • explains Aether + navigates the site (from a small knowledge string),
  • answers general + music questions (Groq LLM),
  • knows the LATEST music via free iTunes RSS (new releases / top songs),
  • grounds factual artist/album questions via MusicBrainz (free, no key).

Wire into api/app.py:
    from chat_routes import router as chat_router
    app.include_router(chat_router)

Env: GROQ_API_KEY (already set for the rest of Aether). No other keys needed.
If Groq isn't configured, it still answers Aether/nav questions from the
knowledge base so the widget is never dead.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["chat"])

# ── Aether knowledge the bot always has (kept short + honest) ─────────────
AETHER_KNOWLEDGE = """
AETHER is an emotion-aware music-intelligence platform by Arnav Shukla (final-year
CS/Data-Science student). It detects a listener's emotion (text or voice, English or
Hindi) across 15 emotions, matches it against a 1.2M-song feature store, and returns
explained playlists.
Pages: Home (overview), Curate (main: mood -> explained playlist), Journey (a LangGraph
agent plans an emotional arc from A to B), Live (starts from a track, detects mood drift,
mixes into a harmonically-compatible next song), Connect (about Arnav + contact + a live
'what people are feeling' feed).
Login is NOT required — everything works anonymously; playback is 30s previews + open-on-
Apple/Spotify/YouTube links. Voice uses a speech-emotion model + Whisper. Explanations come
from a RAG layer grounding each pick in real audio features.
Contact: arnavshuklaforbusiness@gmail.com · GitHub github.com/Arnavs10 · LinkedIn
linkedin.com/in/arnav-shukla10. Keep answers concise, warm, and never invent Aether features.
""".strip()

_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 900  # 15 min


def _cached_get(url: str, headers: Optional[dict] = None, ttl: int = _CACHE_TTL) -> str:
    now = time.time()
    hit = _CACHE.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=8) as r:  # noqa: S310
        body = r.read().decode("utf-8", errors="ignore")
    _CACHE[url] = (now, body)
    return body


# ── free music context sources ───────────────────────────────────────────
def _itunes_latest(limit: int = 8) -> str:
    """Latest / most-popular songs via Apple's free RSS (no key)."""
    try:
        url = f"https://itunes.apple.com/us/rss/topsongs/limit={limit}/json"
        data = json.loads(_cached_get(url))
        entries = data.get("feed", {}).get("entry", [])
        rows = []
        for e in entries:
            name = e.get("im:name", {}).get("label", "")
            artist = e.get("im:artist", {}).get("label", "")
            if name and artist:
                rows.append(f"- {name} — {artist}")
        if rows:
            return "CURRENT TOP SONGS (Apple):\n" + "\n".join(rows)
        print("[chat] iTunes RSS: empty feed.")
        return ""
    except urllib.error.HTTPError as e:
        print(f"[chat] iTunes RSS HTTP {e.code}.")
        return ""
    except Exception as e:  # noqa: BLE001
        print(f"[chat] iTunes RSS failed: {e!r}")
        return ""


_FACT_STOPWORDS = (
    "who is", "who's", "who are", "what is", "what's", "tell me about",
    "tell me", "when did", "info on", "information about",
    "the band", "the artist", "the group", "artist", "band", "group", "about",
)


def _clean_entity(text: str) -> str:
    """Strip interrogative filler so MusicBrainz gets a clean artist name."""
    t = text.strip().rstrip("?.!").strip().lower()
    changed = True
    while changed:
        changed = False
        for sw in _FACT_STOPWORDS:
            if t.startswith(sw + " "):
                t = t[len(sw):].strip()
                changed = True
                break
    return t or text.strip().rstrip("?.!").strip()


def _musicbrainz(query: str) -> str:
    """All close matches as DISTINCT candidates — same-name artists disambiguated, never merged."""
    entity = _clean_entity(query)
    try:
        q = urllib.parse.quote(entity[:120])
        url = f"https://musicbrainz.org/ws/2/artist?query={q}&fmt=json&limit=5"
        data = json.loads(_cached_get(
            url, headers={"User-Agent": "Aether/1.0 (aether.official1010@gmail.com)"}))
        artists = data.get("artists", [])
        rows = []
        for a in artists:
            name = a.get("name", "")
            if not name:
                continue
            if a.get("score", 0) < 70 and rows:   # keep strong matches; always keep top one
                continue
            atype = a.get("type", "")
            began_label = "born" if atype == "Person" else "formed"
            area = (a.get("area") or {}).get("name", "")
            began = (a.get("life-span") or {}).get("begin", "")
            disamb = a.get("disambiguation", "")
            bits = [name]
            if atype:
                bits.append(atype)
            if area:
                bits.append(area)
            if began:
                bits.append(f"{began_label} {began}")
            if disamb:
                bits.append(disamb)
            rows.append("- " + " · ".join(bits))
            if len(rows) >= 3:
                break
        if rows:
            return ("MUSICBRAINZ ARTIST MATCHES "
                    "(each line is a DIFFERENT artist — do not merge):\n" + "\n".join(rows))
        print(f"[chat] MusicBrainz: no artist match for {entity!r}")
        return ""
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:200]
        print(f"[chat] MusicBrainz HTTP {e.code} for {entity!r}: {body}")
        return ""
    except Exception as e:  # noqa: BLE001
        print(f"[chat] MusicBrainz call failed for {entity!r}: {e!r}")
        return ""

def _needs_latest(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("latest", "new", "recent", "top", "chart", "trending", "this week", "2026"))


def _needs_facts(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in (
        "who is", "who's", "who are", "tell me about", "album", "discography",
        "band", "released", "artist", "when did", "songs by", "song by", "genre",
    ))

_REFERENTIAL = (
    "that artist", "this artist", "the artist", "that singer", "this singer",
    "the singer", "that band", "the band", "that group", "same artist",
    "that song's artist", "who sang", "who sings", "that guy", "that girl",
    " she ", " he ", " they ", " her ", " his ", " their ",
)


def _resolve_entity_from_history(message: str, history: list[dict]) -> Optional[str]:
    """If the message refers to an artist by pronoun/reference, resolve the actual
    name from the conversation via one small LLM call. Fail-open: returns None."""
    m = f" {message.lower()} "
    if not any(r in m for r in _REFERENTIAL) or not history:
        return None
    convo = "\n".join(f"{h.get('role')}: {h.get('content')}" for h in history[-6:])
    prompt = (
        "Conversation:\n" + convo + f"\nuser: {message}\n\n"
        "Which music artist is the user asking about in their LAST message? "
        "Resolve references like 'that artist' using the conversation above. "
        "Reply with ONLY the artist's name, nothing else. If unclear, reply NONE."
    )
    name = _groq_reply(
        "You identify which music artist the user means. Reply with only a name or NONE.",
        [], prompt,
    )
    if not name:
        return None
    name = name.strip().strip('"').strip(".")
    if not name or name.upper() == "NONE" or len(name) > 60:
        return None
    return name

# ── Groq call ─────────────────────────────────────────────────────────────
def _groq_reply(system: str, history: list[dict], user: str) -> Optional[str]:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("[chat] GROQ_API_KEY not set — using fallback.")
        return None

    # Env-overridable so a Groq deprecation is a one-line fix, not a code edit.
    # llama-3.3-70b-versatile was deprecated by Groq (June 2026) -> gpt-oss-120b.
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    try:
        msgs = [{"role": "system", "content": system}]
        for h in history[-6:]:
            role = "assistant" if h.get("role") == "assistant" else "user"
            msgs.append({"role": role, "content": str(h.get("content", ""))[:1500]})
        msgs.append({"role": "user", "content": user})

        # The default model is a REASONING model: it spends completion tokens
        # thinking before it writes, and `content` only appears once that ends.
        # So the budget must cover reasoning AND the answer, and reasoning_effort
        # keeps it from over-thinking a chat reply. Sized too small, the API
        # returns 200 with an EMPTY content and finish_reason "length" — no error,
        # just silence, which then looks like the whole chatbot is dead.
        payload = json.dumps({
            "model": model,
            "messages": msgs,
            "temperature": 0.6,
            "max_completion_tokens": 2048,   # max_tokens is deprecated
            "reasoning_effort": "low",       # ignored by non-reasoning models
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "Aether/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            data = json.loads(r.read())

        choice = data["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        if not content:
            finish = choice.get("finish_reason", "?")
            reasoning = choice["message"].get("reasoning") or ""
            print(f"[chat] Groq returned empty content (model={model}, "
                  f"finish_reason={finish}, reasoning_chars={len(reasoning)}). "
                  f"finish_reason='length' means the budget went on reasoning.")
            return None
        return content

    except urllib.error.HTTPError as e:
        # Groq's JSON body names the exact problem (bad model / auth / rate limit).
        body = e.read().decode("utf-8", "replace")
        print(f"[chat] Groq HTTP {e.code} (model={model}): {body}")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"[chat] Groq call failed (model={model}): {e!r}")
        return None
# ── schema + route ────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    source: str  # "groq" | "fallback"


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Build live music context only when the question calls for it (keeps it fast).
    history = [{"role": m.role, "content": m.content} for m in req.history]

    context_bits = []
    if _needs_latest(req.message):
        context_bits.append(_itunes_latest())

    # Resolve referential follow-ups ("who is that artist?") to a real name using history.
    resolved = _resolve_entity_from_history(req.message, history)
    if resolved or _needs_facts(req.message):
        context_bits.append(_musicbrainz(resolved or req.message))

    context = "\n\n".join(b for b in context_bits if b)

    system = AETHER_KNOWLEDGE
    if context:
        system += (
            "\n\nThe live data below was fetched just now and is MORE CURRENT than your "
            "training data — trust it as authoritative. If MUSICBRAINZ ARTIST MATCHES lists "
            "several artists, they are DIFFERENT people who share a name: never merge their "
            "facts. Use the conversation to pick the one the user means; if it's genuinely "
            "ambiguous, briefly list the distinct options and ask which. Weave it in "
            "naturally; don't dump the raw list:\n" + context
        )


    reply = _groq_reply(system, history, req.message)
    if reply:
        return ChatResponse(reply=reply, source="groq")

    # Fallback: no LLM key / call failed — still helpful for Aether/nav questions.
    fallback = (
        "I'm Aether's assistant. I can explain the site — Curate turns your mood into an "
        "explained playlist, Journey plans an emotional arc, and Live mixes songs as your "
        "mood drifts. No login needed. (My music-knowledge brain is offline right now, but "
        "the features all work — try the Curate page.)"
    )
    return ChatResponse(reply=fallback, source="fallback")
