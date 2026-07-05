"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 1C: Local Fusion Dataset Build (terminal, CPU, resumable)
═══════════════════════════════════════════════════════════════════

Runs your Phase 1A (text) + 1B (voice) models over MELD on your own machine
and streams the fusion training tuples to disk. No Colab, no GPU quota, no
runtime reclamation. If it stops for any reason, just run it again — it
resumes from where it left off.

Run it from your Aether project folder (the one containing fusion_data.py):

    python3 build_local.py

Configure the paths in the CONFIG block below if your layout differs.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
import time

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these if your folders are named differently
# ─────────────────────────────────────────────────────────────
# Folder you downloaded from Google Drive (contains inference.py,
# voice_inference.py, text_emotion/, voice_emotion/).
MODELS_ROOT = "./Aether_models"

TEXT_MODEL_DIR = f"{MODELS_ROOT}/text_emotion/final"     # text model weights
VOICE_MODEL_DIR = f"{MODELS_ROOT}/voice_emotion"          # voice head + config

# Where the finished dataset (.npz) and streaming files go.
OUT_DIR = "./data/fusion"
STREAM_DIR = "./data/fusion_stream"

# Optional: cap the number of TRAIN clips (0 = use all ~10k).
# The fusion head is tiny, so a few thousand is plenty if you want it faster.
# Leave at 0 for the full build. Validation/test always build in full.
TRAIN_LIMIT = 0

# Keep CPU threads reasonable so the machine stays usable while it runs.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make your inference.py / voice_inference.py importable.
sys.path.insert(0, MODELS_ROOT)


def _preflight() -> None:
    """Fail early with a clear message if something required is missing."""
    problems = []
    if not os.path.isdir(MODELS_ROOT):
        problems.append(f"MODELS_ROOT not found: {MODELS_ROOT} "
                        "(download the Aether_models folder from Drive).")
    for f in ("inference.py", "voice_inference.py"):
        if not os.path.isfile(f"{MODELS_ROOT}/{f}"):
            problems.append(f"Missing {MODELS_ROOT}/{f}")
    if not os.path.isdir(TEXT_MODEL_DIR):
        problems.append(f"Missing text model dir: {TEXT_MODEL_DIR}")
    else:
        # The folder can exist while the big weights file failed to download
        # (Drive often skips large files). Check the weights explicitly.
        has_weights = any(
            os.path.isfile(f"{TEXT_MODEL_DIR}/{w}")
            for w in ("model.safetensors", "pytorch_model.bin")
        )
        if not has_weights:
            problems.append(
                f"Text model weights missing in {TEXT_MODEL_DIR} "
                "(need model.safetensors ~313MB or pytorch_model.bin). "
                "Re-download this file from Drive — large files often fail "
                "to download inside a zipped folder."
            )
    if not os.path.isfile(f"{VOICE_MODEL_DIR}/final/voice_emotion_head.pt"):
        problems.append(f"Missing {VOICE_MODEL_DIR}/final/voice_emotion_head.pt")
    if not os.path.isfile("fusion_data.py"):
        problems.append("fusion_data.py not found in the current folder. "
                        "Run this script from your Aether project folder.")
    if problems:
        print("❌ Preflight failed:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    print("✅ Preflight OK — models, code, and paths all found.")


def main() -> int:
    _preflight()

    import fusion_data as fd

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STREAM_DIR, exist_ok=True)

    # Chunk limit: if AETHER_CHUNK_LIMIT is set (by the chunked runner), each
    # split processes at most that many clips THIS run, then we exit so the OS
    # reclaims memory. The resume checkpoint makes the next run continue.
    chunk_env = os.environ.get("AETHER_CHUNK_LIMIT")
    chunk_limit = int(chunk_env) if chunk_env and chunk_env.isdigit() else 0

    print("\n🧠 Loading Phase 1A + 1B models (first run downloads emotion2vec ~1GB)…")
    t0 = time.time()
    models = fd.ModelBundle(
        text_model_dir=TEXT_MODEL_DIR,
        voice_model_dir=VOICE_MODEL_DIR,
        load_whisper=False,   # MELD already has transcripts
    )
    print(f"✅ Models ready in {time.time() - t0:.0f}s.\n")

    # Total rows per split (for completion detection).
    split_totals = {"train": 9988, "validation": 1108, "test": 2610}

    splits = [
        ("train", "meld_train", TRAIN_LIMIT),
        ("validation", "meld_val", 0),
        ("test", "meld_test", 0),
    ]

    did_any_work = False
    all_complete = True

    for split_name, tag, limit in splits:
        jsonl = f"{STREAM_DIR}/{tag}.jsonl"
        ckpt = f"{STREAM_DIR}/{tag}.ckpt"

        # Resume: skip rows already processed in a prior run.
        skip = fd.read_checkpoint(ckpt) if os.path.exists(ckpt) else 0

        total = split_totals.get(split_name, 0)
        if total and skip >= total:
            # This split is already fully processed.
            continue

        all_complete = False  # at least one split still has work

        # Apply the per-run chunk limit (bounds THIS run so memory stays flat).
        effective_limit = limit
        if chunk_limit:
            effective_limit = chunk_limit if limit == 0 else min(limit, chunk_limit)

        note = f" (resuming — skipping first {skip} rows)" if skip else ""
        cap = f" [chunk: up to {effective_limit} this run]" if chunk_limit else ""
        print(f"=== Building split: {split_name}{note}{cap} ===")

        samples = fd.iter_meld_samples(split=split_name, limit=effective_limit, skip_rows=skip)
        stats = fd.build_fusion_dataset_streaming(
            samples, models,
            jsonl_path=jsonl,
            checkpoint_path=ckpt,
            use_whisper_transcript_fallback=False,
            log_every=100,
            flush_every=50,
            gc_every=200,
        )
        print(f"[{split_name}] this run: seen={stats.seen} kept={stats.kept} "
              f"voice_fail={stats.skipped_voice_fail} no_gold={stats.skipped_no_gold}\n")
        if stats.seen > 0:
            did_any_work = True

        # Under a chunk limit, stop after doing one split's chunk so memory is
        # freed promptly; the runner re-launches us for the next chunk.
        if chunk_limit and did_any_work:
            print("↩️  Chunk done — exiting so the OS frees memory. "
                  "The runner will continue automatically.")
            return 0

    if all_complete:
        print("=== All splits already complete — converting streamed JSONL → .npz ===")
        for split_name, tag, _ in splits:
            jsonl = f"{STREAM_DIR}/{tag}.jsonl"
            if os.path.exists(jsonl):
                fd.jsonl_to_npz(
                    jsonl_path=jsonl,
                    npz_path=f"{OUT_DIR}/{tag}.npz",
                    source="MELD", split=split_name,
                )
        print(f"\n🎉 Done. Fusion dataset is in: {OUT_DIR}")
        print("   Files: meld_train.npz, meld_val.npz, meld_test.npz (+ .manifest.json)")
        return 42  # sentinel: tells the chunked runner to stop

    return 0


if __name__ == "__main__":
    sys.exit(main())
