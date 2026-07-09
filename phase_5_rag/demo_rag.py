"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 5 · RAG Demo
═══════════════════════════════════════════════════════════════════
See the full RAG pipeline end to end: a real recommendation, then a "why these
songs" explanation. Choose the generator:

    python phase_5_rag/demo_rag.py                 # offline grounded template
    python phase_5_rag/demo_rag.py --llm anthropic # real RAG via Claude  (ANTHROPIC_API_KEY)
    python phase_5_rag/demo_rag.py --llm openai    # real RAG via GPT      (OPENAI_API_KEY)
    python phase_5_rag/demo_rag.py --llm local     # real RAG via local FLAN-T5 (no key)

With --llm, retrieval feeds a language model that GENERATES the explanation
(true RAG). Without it, explanations are composed deterministically from the
same retrieved knowledge (offline, zero-setup).
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Path bootstrap: root + sibling phases + local modules ──
_ROOT = Path(__file__).resolve().parent.parent
_P2 = _ROOT / "phase_2_music_data"
_P3 = _ROOT / "phase_3_emotion_music_mapping"
_P4 = _ROOT / "phase_4_recommendation"
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _P2, _P3, _P4, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                          # noqa: E402
from schema import Song                                     # noqa: E402
from feature_store import FeatureStore                      # noqa: E402
from playlist import PlaylistGenerator                      # noqa: E402
from recommender import AetherRecommender                   # noqa: E402
from config import AETHER_EMOTIONS                          # noqa: E402
from rag import AetherExplainer                             # noqa: E402


def _tiny_recommender() -> AetherRecommender:
    """A tiny in-memory recommender (no CSV/network) for the demo."""
    def mk(tid, name, artist, feats, year=2000):
        base = {f: 0.5 for f in
                ["danceability", "energy", "key", "loudness", "mode",
                 "speechiness", "acousticness", "instrumentalness",
                 "liveness", "valence", "tempo"]}
        base.update(feats)
        return Song(tid, name, [artist], year, f"{year}-01-01", base)

    songs = [
        mk("s1", "Rain", "Nael", {"tempo": 68, "energy": 0.20, "valence": 0.15, "danceability": 0.30, "acousticness": 0.70, "instrumentalness": 0.20}),
        mk("s2", "Lonely Road", "Vera", {"tempo": 70, "energy": 0.22, "valence": 0.18, "danceability": 0.32, "acousticness": 0.68, "instrumentalness": 0.25}),
        mk("s3", "Grey Skies", "Orin", {"tempo": 65, "energy": 0.18, "valence": 0.12, "danceability": 0.28, "acousticness": 0.72, "instrumentalness": 0.22}),
    ]
    store = FeatureStore().build_from_songs(songs)
    return AetherRecommender(PlaylistGenerator(store, max_per_artist=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aether Phase 5 RAG demo")
    ap.add_argument("--llm", choices=["anthropic", "openai", "groq", "local"], default=None,
                    help="LLM backend for generation (omit for offline template)")
    ap.add_argument("--text", default="i feel really low and heavy tonight",
                    help="the listener's request text")
    args = ap.parse_args()

    # Build the LLM generator if requested — and PROBE it, so we never claim
    # "true RAG" when the call is actually failing and falling back.
    llm_fn = None
    if args.llm:
        from llm_backends import make_llm
        try:
            candidate = make_llm(args.llm)
            probe = candidate("Reply with exactly one word: PONG")
            llm_fn = candidate
            print(f"[demo] generator: {args.llm} LLM — probe OK "
                  f"({probe.strip()[:40]!r}) → TRUE RAG active\n")
        except Exception as exc:
            print(f"[demo] '{args.llm}' LLM probe FAILED → {type(exc).__name__}: {exc}")
            print("[demo] falling back to offline grounded template "
                  "(NOT true RAG — fix the error above)\n")
            llm_fn = None
    else:
        print("[demo] generator: offline grounded template "
              "(pass --llm anthropic|openai|local for true RAG)\n")

    # 1. Recommend (Phases 1C→4).
    rec = _tiny_recommender()
    dist = np.zeros(len(AETHER_EMOTIONS)); dist[AETHER_EMOTIONS.index("sad")] = 0.9
    recommendation = rec.recommend(dist, args.text, length=3)

    # 2. Explain (Phase 5 RAG) and fold the 'why' back in.
    rag = AetherExplainer.default(llm_fn=llm_fn)
    expl = rag.explain(recommendation)

    # 3. Show it.
    print("=" * 68)
    print(f"REQUEST: {args.text!r}")
    print(f"MOOD:    {recommendation.intensity_label} "
          f"'{recommendation.dominant_emotions[0][0]}'   "
          f"ARC: {recommendation.arc_shape}   TRACKS: {recommendation.size}")
    print("=" * 68)
    print(f"\nSUMMARY\n  {expl.summary}\n")
    print("WHY EACH TRACK")
    for i, (t, te) in enumerate(zip(recommendation.tracks, expl.tracks), 1):
        print(f"  {i}. {t.title} — {t.artist}")
        print(f"     {te.why}")
        print(f"     grounded in: {te.citations}\n")
    print("CITATIONS (retrieved knowledge)")
    for c in expl.citations:
        print(f"  • {c['id']}  —  {c['title']}")


if __name__ == "__main__":
    main()
