"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 6 · Agent Demo
═══════════════════════════════════════════════════════════════════
Watch the agent turn a sentence into an explained emotional journey:

    python phase_6_agentic/demo_agent.py
    python phase_6_agentic/demo_agent.py --request "lift me out of feeling sad"
    python phase_6_agentic/demo_agent.py --request "from anxious to calm" --llm anthropic
    python phase_6_agentic/demo_agent.py --store data/feature_store --length 15

Without --store it uses a tiny built-in library (no data files needed).
--llm (anthropic|openai|local) turns on LLM perception + LLM-generated "why".
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
_P5 = _ROOT / "phase_5_rag"
for _p in (_ROOT, _HERE, _P5):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent import AetherAgent, _build_tiny_tools             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Aether Phase 6 agent demo")
    ap.add_argument("--request", default="help me go from anxious to calm")
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--llm", choices=["anthropic", "openai", "groq", "local"], default=None)
    ap.add_argument("--store", default=None, help="path to a real feature store")
    args = ap.parse_args()

    llm_fn = None
    if args.llm:
        from llm_backends import make_llm
        try:
            llm_fn = make_llm(args.llm)
            print(f"[demo] LLM: {args.llm} (perception + generation)\n")
        except Exception as exc:
            print(f"[demo] could not init '{args.llm}' ({exc}); using offline paths.\n")

    if args.store:
        agent = AetherAgent.default(store_path=args.store, llm_fn=llm_fn)
    else:
        print("[demo] using tiny built-in library (pass --store for the real one)\n")
        agent = AetherAgent(_build_tiny_tools(), llm_fn=llm_fn)

    res = agent.run(args.request, length=args.length)

    print("=" * 68)
    print(f"REQUEST : {args.request!r}")
    print(f"PERCEIVE: {res.perceived.start} → {res.perceived.target} "
          f"({res.perceived.source}; {', '.join(res.perceived.notes)})")
    print(f"PLAN    : {res.plan.describe()}  [{res.plan.direction}]")
    print(f"TRACE   : {' · '.join(res.trace)}")
    print("=" * 68)
    print(f"\n{res.explanation.summary}\n")
    print("THE JOURNEY")
    for i, t in enumerate(res.playlist.tracks, 1):
        why = t.extra.get("why", "")
        print(f"  {i:2}. [{t.source_emotion:11}] {t.title} — {t.artist} "
              f"(energy {t.energy:.2f})")
        if why:
            print(f"      ↳ {why}")


if __name__ == "__main__":
    main()
