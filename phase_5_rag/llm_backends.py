"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 5 · LLM Backends (the real "G" in RAG)
═══════════════════════════════════════════════════════════════════
`explainer.LLMExplainer` takes an injected `llm_fn(prompt) -> str`. This module
provides CONCRETE, runnable implementations of that function, so Aether's RAG
actually generates with a language model conditioned on retrieved context —
rather than only stitching retrieved sentences with a template.

Wiring it up is one line:

    from rag import AetherExplainer
    from llm_backends import make_llm
    rag = AetherExplainer.default(llm_fn=make_llm("anthropic"))   # or "local"
    rag.annotate(recommendation)      # each track.why is now LLM-generated

Backends behind one signature:

  • anthropic — Claude via the `anthropic` SDK (needs ANTHROPIC_API_KEY, paid).
  • openai    — GPT via the `openai` SDK (needs OPENAI_API_KEY, paid).
  • groq      — Llama etc. via Groq's FREE API (needs GROQ_API_KEY). Recommended
                no-cost cloud option: fast, OpenAI-compatible, good quality.
  • local     — a small local generative model (FLAN-T5) via `transformers`.
                No API key, fully offline after a one-time download, so a fresh
                clone can produce REAL LLM generation with zero credentials.

Every backend is grounding-constrained by the prompt LLMExplainer builds (it
passes retrieved docs + measured features and says "use ONLY this context").
Any backend failure is caught upstream by LLMExplainer, which falls back to the
deterministic grounded template — so the pipeline never breaks.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
from typing import Callable

LLMFn = Callable[[str], str]

# System instruction shared by the API backends — reinforces grounding.
_SYSTEM = (
    "You are Aether's music explainer. Explain why a song or playlist fits a "
    "listener's mood using ONLY the provided context and measurements. Never "
    "invent facts about the song, its lyrics, or its artist. Be warm, concrete, "
    "and concise."
)


# ──────────────────────────────────────────────
# Anthropic (Claude) — recommended
# ──────────────────────────────────────────────
def anthropic_llm(
    model: str | None = None,          # defaults below; override via arg or ANTHROPIC_MODEL
    max_tokens: int = 256,
    temperature: float = 0.4,
    api_key: str | None = None,
) -> LLMFn:
    """
    Build an `llm_fn` backed by Anthropic's Messages API.

    Model resolution order: explicit `model` arg → $ANTHROPIC_MODEL → a current
    default. Model names change over time, so if you get a 404, set
    ANTHROPIC_MODEL to a model your account has.

    Raises at construction (fail fast) if the SDK or key is missing, so wiring
    problems surface immediately rather than mid-request.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic_llm needs the `anthropic` SDK: pip install anthropic"
        ) from exc

    model = model or os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5-20251001"

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it, or pass api_key=…, or use "
            "make_llm('local') for a no-key offline model."
        )

    client = anthropic.Anthropic(api_key=key)

    def _call(prompt: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        ).strip()

    return _call


# ──────────────────────────────────────────────
# OpenAI (GPT) — optional
# ──────────────────────────────────────────────
def openai_llm(
    model: str | None = None,          # override via arg or OPENAI_MODEL
    max_tokens: int = 256,
    temperature: float = 0.4,
    api_key: str | None = None,
) -> LLMFn:
    """Build an `llm_fn` backed by OpenAI's Chat Completions API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai_llm needs the `openai` SDK: pip install openai"
        ) from exc

    model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it, or pass api_key=…, or use "
            "make_llm('local') for a no-key offline model."
        )

    client = OpenAI(api_key=key)

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    return _call


# ──────────────────────────────────────────────
# Groq (free tier) — recommended no-cost cloud option
# ──────────────────────────────────────────────
def groq_llm(
    model: str | None = None,          # override via arg or GROQ_MODEL
    max_tokens: int = 2048,
    temperature: float = 0.4,
    api_key: str | None = None,
    reasoning_effort: str | None = "low",
) -> LLMFn:
    """
    Build an `llm_fn` backed by Groq's free API (OpenAI-compatible).

    Groq offers a genuine free tier with fast open models. Get a key at
    https://console.groq.com/keys and export GROQ_API_KEY. If the default model
    is retired, set GROQ_MODEL (see console.groq.com/docs/models).

    On the token budget
    -------------------
    The default model is a REASONING model: it spends completion tokens thinking
    before it writes anything, and `content` only appears once that finishes.
    The budget therefore has to cover reasoning AND the answer. A budget sized
    for the answer alone (the old 256) is silently consumed by reasoning on any
    non-trivial prompt, and the API returns HTTP 200 with an EMPTY `content` and
    `finish_reason: "length"`. No error, no exception, just nothing — which then
    surfaces downstream as a blank explanation rather than a failure.

    So: a budget with headroom, `reasoning_effort="low"` to stop the model
    over-thinking a one-sentence job, and an explicit raise when content is
    empty so callers can fall back deliberately instead of rendering "".

    Args:
        model: Groq model id. Falls back to $GROQ_MODEL, then a current default.
        max_tokens: completion budget, covering reasoning + answer.
        temperature: sampling temperature.
        api_key: overrides $GROQ_API_KEY.
        reasoning_effort: "low" | "medium" | "high", or None to omit. Ignored by
            non-reasoning models, so it is safe to leave on.

    Raises (at call time):
        RuntimeError: the model returned no usable content. Callers treat this
            as a failure and fall back; it is never rendered to a user.
    """
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError(
            "groq_llm needs the `groq` SDK: pip install groq"
        ) from exc

    model = model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys, then export GROQ_API_KEY=…"
        )

    client = Groq(api_key=key)

    def _call(prompt: str) -> str:
        kwargs: dict = {
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,   # max_tokens is deprecated
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": prompt}],
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            # An older SDK may not know reasoning_effort / max_completion_tokens.
            kwargs.pop("reasoning_effort", None)
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens", max_tokens)
            resp = client.chat.completions.create(**kwargs)

        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if content:
            return content

        # Empty content is a real failure, not an empty answer. Say so loudly and
        # raise, so the caller's fallback fires instead of rendering a blank.
        finish = getattr(choice, "finish_reason", "?")
        reasoning = getattr(choice.message, "reasoning", None) or ""
        print(f"[llm] groq returned empty content "
              f"(model={model}, finish_reason={finish}, "
              f"reasoning_chars={len(reasoning)}, budget={max_tokens}). "
              f"If finish_reason is 'length', the budget was spent on reasoning.")
        raise RuntimeError(f"groq returned no content (finish_reason={finish})")

    return _call
def local_llm(
    model: str = "google/flan-t5-base",       # or 'google/flan-t5-small' (~300MB)
    max_new_tokens: int = 128,
) -> LLMFn:
    """
    Build an `llm_fn` backed by a small local seq2seq LM via `transformers`.

    Real generative model, no API key. The first call downloads the weights
    (~1GB for flan-t5-base; use flan-t5-small for ~300MB), then runs offline on
    CPU. Lets a fresh clone demonstrate genuine LLM generation with no
    credentials.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "local_llm needs `transformers` (and torch): "
            "pip install transformers torch"
        ) from exc

    pipe = pipeline("text2text-generation", model=model)

    def _call(prompt: str) -> str:
        # FLAN-T5 has no system role, so prepend the grounding instruction.
        full = f"{_SYSTEM}\n\n{prompt}"
        out = pipe(full, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"].strip()

    return _call


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────
_BACKENDS: dict[str, Callable[..., LLMFn]] = {
    "anthropic": anthropic_llm,
    "claude": anthropic_llm,
    "openai": openai_llm,
    "gpt": openai_llm,
    "groq": groq_llm,
    "local": local_llm,
    "flan": local_llm,
}


def make_llm(provider: str = "anthropic", **kwargs) -> LLMFn:
    """
    Build an `llm_fn` for the named provider ("anthropic" | "openai" | "local").

    Extra kwargs pass through to the specific backend (model, temperature, …).
    """
    key = provider.lower().strip()
    if key not in _BACKENDS:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. "
            f"Choose from: {sorted(set(_BACKENDS))}."
        )
    return _BACKENDS[key](**kwargs)


# ─────────────────────────────────────────────────────────────
# Self-test — offline-safe (no key, no download). Verifies wiring + guards,
# and proves the FULL RAG chain generates via an injected llm_fn.
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    import sys
    from pathlib import Path
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    print("LLM backends self-test")
    print("-" * 55)

    # 1. Unknown provider → clear error.
    try:
        make_llm("nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("  unknown provider → ValueError ✓")

    # 2. Missing key fails fast with an actionable message (if SDK present) or
    #    ImportError (if SDK absent) — either way, no silent misconfiguration.
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        make_llm("anthropic")
        raised = None
    except (RuntimeError, ImportError) as exc:
        raised = exc
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
    assert raised is not None, "anthropic without key should raise"
    print(f"  anthropic w/o key → {type(raised).__name__} ✓")

    # 3. The whole RAG chain generates via an injected llm_fn (the real proof:
    #    output is the LLM's text, NOT the template).
    from rag import AetherExplainer
    from models import Track, Recommendation

    def stub_llm(prompt: str) -> str:
        assert "ONLY" in prompt and "CONTEXT" in prompt.upper()  # grounded prompt
        return "LLM: this low-energy track mirrors your reflective mood."

    rec = Recommendation(
        tracks=[Track(title="Rain", artist="D", track_id="s1",
                      energy=0.19, valence=0.14, tempo=0.12,
                      match_score=0.94, source_emotion="sad", rank=1)],
        request_text="down", intent_mode="single", intensity_level=3,
        intensity_label="intense", dominant_emotions=[("sad", 1.0)],
        arc_shape="descending", reason="An intense 'sad' playlist.",
    )
    rag = AetherExplainer.default(prefer_chroma=False, llm_fn=stub_llm)
    expl = rag.explain(rec)
    assert expl.tracks[0].why == "LLM: this low-energy track mirrors your reflective mood.", \
        expl.tracks[0].why
    assert expl.tracks[0].citations, "LLM output should still carry retrieval citations"
    print(f"  RAG via llm_fn → \"{expl.tracks[0].why}\"")
    print(f"  grounded citations preserved → {expl.tracks[0].citations}")

    # 4. Backends are all constructible names.
    assert set(_BACKENDS) >= {"anthropic", "openai", "groq", "local"}
    print(f"  providers available → {sorted(set(_BACKENDS))}")

    print("-" * 55)
    print("✅ All LLM-backend self-tests passed.")


if __name__ == "__main__":
    _selftest()
