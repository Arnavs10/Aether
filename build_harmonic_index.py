"""
═══════════════════════════════════════════════════════════════════
AETHER — Build Harmonic Index (real 1.2M songs)
═══════════════════════════════════════════════════════════════════
The Live Player needs each track's Camelot key + BPM for harmonic mixing, but
the saved feature store (.npz) dropped `key` and `mode`. This one-time script
reads them straight from tracks_features.csv and writes a compact index to disk:

    track_id → (camelot_code, bpm, energy)

Run once (takes a few seconds over 1.2M rows):

    cd ~/Desktop/Aether
    python3 build_harmonic_index.py

It writes phase_7_drift_crossfade/harmonic_index.json.gz, which the API loads at
startup. CSV columns are matched to your real file: id, key, mode, tempo, energy.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_P7 = ROOT / "phase_7_drift_crossfade"
for p in (ROOT, _P7):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from camelot import to_camelot                              # Phase 7  # noqa: E402

CSV_PATH = ROOT / "phase_2_music_data" / "tracks_features.csv"
OUT_PATH = _P7 / "harmonic_index.json.gz"


def _to_int(x: str, default: int = -1) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _to_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("AETHER · BUILD HARMONIC INDEX")
    print("=" * 60)
    print(f"reading: {csv_path}")

    t0 = time.time()
    # Compact record per track: [camelot, bpm, energy]. Only store rows whose
    # key is known (Camelot needs it); unknown-key tracks simply won't be
    # transition candidates, which is correct.
    index: dict[str, list] = {}
    total = 0
    with_key = 0

    # Raise the CSV field size limit — some rows have long text fields.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        needed = {"id", "key", "mode", "tempo", "energy"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            print(f"❌ CSV missing expected columns: {missing}")
            print(f"   found: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            total += 1
            tid = (row.get("id") or "").strip()
            if not tid:
                continue
            camelot = to_camelot(_to_int(row.get("key", -1)),
                                 _to_int(row.get("mode", -1)))
            if camelot is None:
                continue
            index[tid] = [
                camelot,
                round(_to_float(row.get("tempo", 0.0)), 2),   # real BPM
                round(_to_float(row.get("energy", 0.0)), 4),
            ]
            with_key += 1
            if total % 200_000 == 0:
                print(f"  … {total:,} rows scanned, {with_key:,} indexed")

    build_s = time.time() - t0
    print(f"\nscanned {total:,} rows in {build_s:.1f}s")
    print(f"indexed {with_key:,} tracks with a known key "
          f"({with_key / max(1,total) * 100:.1f}%)")

    # Save (gzipped JSON keeps it small + fast to load).
    t0 = time.time()
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(index, f)
    save_s = time.time() - t0
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nsaved → {OUT_PATH}")
    print(f"  {size_mb:.1f} MB, written in {save_s:.1f}s")

    # Peek.
    for tid in list(index)[:3]:
        cam, bpm, en = index[tid]
        print(f"  e.g. {tid[:22]:22} → {cam:>3}  {bpm:.0f} BPM  energy {en:.2f}")

    print("\n✅ harmonic index built. The API will load it automatically.")


if __name__ == "__main__":
    main()
