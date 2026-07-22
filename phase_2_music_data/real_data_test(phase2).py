"""
═══════════════════════════════════════════════════════════════════
AETHER — Real-Data Integration Smoke Test
═══════════════════════════════════════════════════════════════════
Run ONCE against the real 1.2M store to verify curate + journey work at scale,
and to time them. This does NOT touch the live player (that needs the harmonic
index, handled in a separate step).

    cd ~/Desktop/Aether
    export GROQ_API_KEY=gsk_...            # optional (LLM explanations)
    python3 real_data_test.py

It auto-finds your store under phase_2_music_data/store/. If your .npz has a
different name, pass it:  python3 real_data_test.py path/to/store.npz
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name in ("", "phase_2_music_data", "phase_3_emotion_music_mapping",
             "phase_4_recommendation", "phase_5_rag", "api"):
    p = ROOT / name if name else ROOT
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def find_store() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    store_dir = ROOT / "phase_2_music_data" / "store"
    cands = list(store_dir.glob("*.npz")) if store_dir.exists() else []
    if not cands:
        cands = list((ROOT / "phase_2_music_data").glob("**/*.npz"))
    if not cands:
        print("❌ No .npz store found under phase_2_music_data/.")
        print("   Pass the path explicitly: python3 real_data_test.py PATH/store.npz")
        sys.exit(1)
    return str(cands[0])


def main() -> None:
    from feature_store import FeatureStore

    store_path = find_store()
    print("=" * 60)
    print("AETHER · REAL-DATA SMOKE TEST")
    print("=" * 60)
    print(f"store file: {store_path}")

    # ── 1. Load the real store ──
    t0 = time.time()
    store = FeatureStore.load(store_path)
    load_s = time.time() - t0
    n = len(store)
    print(f"\n[1] LOAD  → {n:,} songs in {load_s:.2f}s")
    if n < 1000:
        print("    ⚠️  fewer songs than expected — is this the real store?")
    # peek at a few real track ids/names
    sample = [(str(store.track_ids[i]), str(store.names[i]), str(store.artists[i]))
              for i in range(min(3, n))]
    for tid, name, art in sample:
        print(f"      e.g. {tid[:22]:22} | {name[:30]:30} | {art[:24]}")

    # ── 2. Build the service over the real store ──
    from service import AetherService

    # AetherService.__init__ does NOT auto-load the LLM (only the from_sample /
    # from_store_path constructors do), so load Groq explicitly here and surface
    # any error rather than silently falling back.
    llm_fn = None
    if os.getenv("GROQ_API_KEY"):
        try:
            from llm_backends import make_llm
            llm_fn = make_llm("groq")
            probe = llm_fn("Reply with exactly one word: OK")
            print(f"    LLM probe → {probe.strip()[:24]!r}  (Groq/Llama live)")
        except Exception as exc:
            print(f"    ⚠️ LLM init FAILED → {type(exc).__name__}: {exc}")
            llm_fn = None
    else:
        print("    (no GROQ_API_KEY in env — RAG will use templates)")

    t0 = time.time()
    svc = AetherService(store, llm_fn=llm_fn)
    build_s = time.time() - t0
    print(f"\n[2] SERVICE  → built in {build_s:.2f}s | llm={'on' if svc.llm_fn else 'off'}")

    # ── 3. Curate on real data (timed) ──
    print("\n[3] CURATE  (emotion='sad', length=8)")
    t0 = time.time()
    rec = svc.curate(emotion="sad", length=8, explain=False)  # explain off = pure match speed
    curate_s = time.time() - t0
    print(f"    matched {rec.size} real songs in {curate_s:.2f}s")
    for t in rec.tracks[:5]:
        print(f"      {t.rank}. {t.title[:34]:34} — {t.artist[:22]:22} "
              f"(score {t.match_score:.3f})")

    # ── 4. Curate WITH RAG explanation (timed) ──
    print("\n[4] CURATE + RAG explanation (length=4)")
    t0 = time.time()
    rec2 = svc.curate(emotion="calm", length=4, explain=True)
    rag_s = time.time() - t0
    print(f"    curate+explain in {rag_s:.2f}s")
    if rec2.tracks:
        why = (rec2.tracks[0].extra or {}).get("why", "")
        print(f"      why[0]: {why[:120]}")

    # ── 5. Journey on real data (timed) ──
    print("\n[5] JOURNEY  ('from anxious to calm', length=8)")
    t0 = time.time()
    res = svc.journey("from anxious to calm", length=8)
    journey_s = time.time() - t0
    print(f"    {res.plan.describe()} | {res.playlist.size} songs in {journey_s:.2f}s")
    for t in res.playlist.tracks[:6]:
        print(f"      [{t.source_emotion:10}] {t.title[:30]:30} — {t.artist[:20]}")

    # ── summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  songs loaded     : {n:,}")
    print(f"  load time        : {load_s:.2f}s")
    print(f"  curate (match)   : {curate_s:.2f}s")
    print(f"  curate + RAG     : {rag_s:.2f}s")
    print(f"  journey          : {journey_s:.2f}s")
    print("  ✅ curate + journey work on REAL data" if rec.size and res.playlist.size
          else "  ⚠️ something returned empty — see above")
    print("=" * 60)
    print("\nNote: the Live Player (Camelot transitions) is tested separately —")
    print("it needs a harmonic index rebuilt from tracks_features.csv (key/mode).")


if __name__ == "__main__":
    main()
