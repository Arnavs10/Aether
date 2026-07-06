"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 2: Build the Music Feature Store
═══════════════════════════════════════════════════════════════════

One-shot script: reads the 1.2M-song CSV, cleans it (loader.py), normalizes
into the 6-dim match space (feature_store.py), and saves a compact .npz store
that Phase 3+ loads instantly (no re-parsing the 1.2 GB CSV).

Runs memory-safe: songs stream in chunks, then build once at the end. On a
laptop this takes a couple of minutes and a few GB of peak RAM for ~1.2M
lightweight Song objects; use --limit to build a smaller store for testing.

Usage:
    python build_store.py
    python build_store.py --csv tracks_features.csv --out store/music_store.npz
    python build_store.py --limit 50000        # quick test build
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loader import iter_song_chunks
from feature_store import FeatureStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Aether music feature store.")
    parser.add_argument("--csv", default="tracks_features.csv",
                        help="Path to the source Spotify CSV.")
    parser.add_argument("--out", default="store/music_store.npz",
                        help="Output .npz path for the built store.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max songs to include (0 = all ~1.2M).")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}\n"
              f"   Download rodolfofigueroa/spotify-12m-songs and place it here.")
        return 1

    print("═" * 60)
    print("  Building Aether Music Feature Store")
    print("═" * 60)
    print(f"  source : {csv_path}")
    print(f"  output : {args.out}")
    print(f"  limit  : {args.limit or 'ALL'}\n")

    t0 = time.time()

    # Stream + accumulate Song objects (lightweight — just metadata + features).
    all_songs = []
    for chunk in iter_song_chunks(str(csv_path), chunk_size=args.chunk_size,
                                  limit=args.limit, log=True):
        all_songs.extend(chunk)
        if args.limit and len(all_songs) >= args.limit:
            all_songs = all_songs[:args.limit]
            break

    if not all_songs:
        print("❌ No valid songs loaded — check the CSV.")
        return 1

    print(f"\n🔧 Normalizing {len(all_songs):,} songs into match space…")
    store = FeatureStore().build_from_songs(all_songs)

    print(f"💾 Saving store ({len(store):,} songs)…")
    store.save(args.out)

    dt = time.time() - t0
    print(f"\n✅ Done in {dt:.0f}s. Store → {args.out}")
    print(f"   (+ sidecar {Path(args.out).with_suffix('.meta.json')})")

    # Quick sanity: search each emotion and show the top match.
    print("\n🔎 Sanity check — top match per emotion:")
    try:
        from config import EMOTION_MUSIC_TARGETS
        for emo in list(EMOTION_MUSIC_TARGETS)[:5]:
            res = store.search_emotion(emo, k=1)
            if res:
                r = res[0]
                print(f"   {emo:<11} → {r.name[:35]:<35} "
                      f"({r.artist[:20]}) score={r.score:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"   (skipped: {exc})")

    print("\nNext: Phase 3 — emotion→music matching engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
