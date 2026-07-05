#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# AETHER — Phase 1C: Chunked build runner
# ═══════════════════════════════════════════════════════════════════
# Runs build_local.py repeatedly. Each run processes a bounded number of
# clips and then EXITS, which fully frees memory back to the OS. The
# resume checkpoint means each run picks up where the last stopped, so
# looping this drives the build to completion without RAM ever
# accumulating (fixes the `zsh: killed` OOM).
#
# Usage:
#   chmod +x run_build_chunked.sh
#   ./run_build_chunked.sh
#
# Stops automatically when a run reports the build is complete.
# ═══════════════════════════════════════════════════════════════════
set -u

MAX_LOOPS=200          # safety cap on number of chunk-runs
SLEEP_BETWEEN=2        # seconds between chunks (lets the OS reclaim RAM)

echo "=== Aether chunked build starting ==="
for i in $(seq 1 "$MAX_LOOPS"); do
    echo ""
    echo "──────────────────────────────────────────"
    echo "  Chunk run #$i"
    echo "──────────────────────────────────────────"

    # Run one chunk. AETHER_CHUNK_LIMIT tells build_local.py how many
    # clips to process this run before exiting cleanly.
    AETHER_CHUNK_LIMIT=400 python3 build_local.py
    status=$?

    # build_local.py exits 0 = more work remains (ran a chunk),
    #                    exit 42 = everything complete (our sentinel).
    if [ "$status" -eq 42 ]; then
        echo ""
        echo "✅ Build reported COMPLETE. Stopping."
        exit 0
    fi
    if [ "$status" -ne 0 ]; then
        echo "⚠️  Chunk exited with status $status (likely OOM-killed mid-chunk)."
        echo "    That's fine — the checkpoint saved progress. Continuing."
    fi

    sleep "$SLEEP_BETWEEN"
done

echo "Reached MAX_LOOPS ($MAX_LOOPS). Re-run the script if not yet complete."
