"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 2: Music Data Loader
═══════════════════════════════════════════════════════════════════

Loads the source Spotify CSV (rodolfofigueroa/spotify-12m-songs, ~1.2M rows,
~1.2 GB) into clean `Song` objects.

Reads in CHUNKS so the whole file never sits in memory at once — the same
memory-safety principle used in Phase 1C. Along the way it:

  • keeps only the columns we need (metadata + 11 raw features)
  • drops rows with missing/invalid audio features
  • drops rows with empty id or name
  • de-duplicates by track id (first occurrence wins)
  • parses the stringified `artists` field into a real list

Usage:
    from loader import load_songs, iter_song_chunks

    # Stream (memory-safe) — recommended for the full 1.2M file:
    for chunk in iter_song_chunks("tracks_features.csv"):
        for song in chunk:
            ...

    # Or load everything into a list (fine on 16GB+ machines):
    songs = load_songs("tracks_features.csv", limit=0)
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from schema import (
    Song, RAW_FEATURES, META_COLUMNS, SOURCE_COLUMNS,
    parse_artists, safe_year,
)

DEFAULT_CHUNK_SIZE = 100_000


@dataclass
class LoadStats:
    """Bookkeeping across the load."""
    rows_read: int = 0
    kept: int = 0
    dropped_missing_features: int = 0
    dropped_missing_meta: int = 0
    dropped_duplicate: int = 0

    def summary(self) -> str:
        return (
            f"read={self.rows_read:,} | kept={self.kept:,} | "
            f"dropped[feat={self.dropped_missing_features:,}, "
            f"meta={self.dropped_missing_meta:,}, "
            f"dup={self.dropped_duplicate:,}]"
        )


def _clean_chunk(
    df: pd.DataFrame,
    seen_ids: set[str],
    stats: LoadStats,
) -> list[Song]:
    """Validate/clean one dataframe chunk into a list of Song objects.

    Args:
        df: raw chunk from pd.read_csv.
        seen_ids: set of track ids already kept (for cross-chunk dedup).
        stats: LoadStats to update in place.

    Returns:
        List of clean Song objects from this chunk.
    """
    songs: list[Song] = []

    # Ensure required columns exist (fail loud if the schema drifted).
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Source CSV is missing expected columns: {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    for row in df.itertuples(index=False):
        stats.rows_read += 1
        rd = row._asdict()

        raw_id = rd.get("id")
        raw_name = rd.get("name")
        # pandas turns empty CSV cells into NaN (float), which str()-ifies to
        # "nan". Guard against that so blank ids/names are correctly dropped.
        track_id = "" if pd.isna(raw_id) else str(raw_id).strip()
        name = "" if pd.isna(raw_name) else str(raw_name).strip()

        # Drop rows without a usable id or name.
        if not track_id or not name or track_id.lower() == "nan" or name.lower() == "nan":
            stats.dropped_missing_meta += 1
            continue

        # Cross-chunk de-duplication by track id.
        if track_id in seen_ids:
            stats.dropped_duplicate += 1
            continue

        # Extract + validate the 11 raw features (all must be present & numeric).
        raw_features: dict[str, float] = {}
        bad = False
        for feat in RAW_FEATURES:
            val = rd.get(feat)
            try:
                fval = float(val)
            except (TypeError, ValueError):
                bad = True
                break
            if pd.isna(fval):
                bad = True
                break
            raw_features[feat] = fval
        if bad:
            stats.dropped_missing_features += 1
            continue

        song = Song(
            track_id=track_id,
            name=name,
            artists=parse_artists(rd.get("artists")),
            year=safe_year(rd.get("year")),
            release_date=(str(rd.get("release_date")).strip()
                          if rd.get("release_date") is not None else None),
            raw_features=raw_features,
        )
        songs.append(song)
        seen_ids.add(track_id)
        stats.kept += 1

    return songs


def iter_song_chunks(
    csv_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int = 0,
    log: bool = True,
) -> Iterator[list[Song]]:
    """Stream clean Song objects from the CSV, one chunk at a time.

    Memory stays flat regardless of file size. De-duplication is maintained
    across chunks via a shared `seen_ids` set.

    Args:
        csv_path: path to the source CSV.
        chunk_size: rows per read chunk.
        limit: if > 0, stop after yielding roughly this many kept songs.
        log: print progress per chunk.

    Yields:
        Lists of Song objects (one list per chunk).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Download rodolfofigueroa/"
            f"spotify-12m-songs and place it here."
        )

    seen_ids: set[str] = set()
    stats = LoadStats()

    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        usecols=lambda c: c in SOURCE_COLUMNS,  # read only needed columns
        low_memory=False,
    )

    for chunk_df in reader:
        songs = _clean_chunk(chunk_df, seen_ids, stats)
        if log:
            print(f"   {stats.summary()}", flush=True)
        if songs:
            yield songs
        if limit and stats.kept >= limit:
            break

    if log:
        print(f"\n✅ Load complete: {stats.summary()}")


def load_songs(
    csv_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int = 0,
    log: bool = True,
) -> list[Song]:
    """Load all (or up to `limit`) clean Song objects into a single list.

    Convenience wrapper around iter_song_chunks. On the full 1.2M file this
    holds ~1M lightweight Song objects in memory (fine on 16GB machines); use
    iter_song_chunks directly if you want to stream instead.
    """
    out: list[Song] = []
    for chunk in iter_song_chunks(csv_path, chunk_size, limit, log):
        out.extend(chunk)
        if limit and len(out) >= limit:
            return out[:limit] if limit else out
    return out


if __name__ == "__main__":
    # Quick smoke test on the first chunk only.
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "tracks_features.csv"
    print(f"Smoke test on {csv} (first 5 kept songs)…")
    songs = load_songs(csv, limit=5)
    for s in songs:
        print(f"  {s.name[:40]:<40} | {s.primary_artist()[:25]:<25} "
              f"| {s.year} | energy={s.raw_features['energy']:.2f} "
              f"valence={s.raw_features['valence']:.2f}")
    print(f"\nLoaded {len(songs)} songs ✓")
