"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 1C: Fusion Dataset Builder
═══════════════════════════════════════════════════════════════════

Builds the training dataset for the fusion layer.

The fusion layer does NOT see raw audio or raw text. It operates in
*distribution-space*: its inputs are the 15-dim probability outputs of
the Phase 1A (text) and Phase 1B (voice) models. This module runs both
models over a real multimodal dataset (MELD) and collects, per utterance:

    text_probs[15]        — Phase 1A output on the transcript
    voice_probs[15]       — Phase 1B acoustic output on the audio
    confidence_features   — entropy / max-prob / agreement signals
    modality_mask         — which modalities are present (text, voice)
    target                — gold Aether emotion id (0..14)

Design principles
-----------------
1. WRAP, don't reimplement. We import the exact Phase 1A/1B inference
   classes so the probabilities here are byte-identical to production.
2. LABEL UNIVERSE consistency. MELD gold labels are mapped to Aether
   using the SAME convention Phase 1B was trained on (surprise→dreamy,
   disgust→frustrated, fear→anxious, ...). Mixing conventions would put
   text and voice in different label spaces and silently corrupt the
   entire ablation.
3. MOSEI-READY. MELD-specific logic lives behind a small adapter
   (`iter_meld_samples`). A future `iter_mosei_samples` emitting the same
   RawSample tuples plugs in with no change to the core builder.
4. DEFENSIVE. emotion2vec extraction can fail on short/corrupt clips
   (Phase 1B returns zeros(15) on failure). We detect and drop those
   rather than feeding a fake all-zero "distribution" into training.

Output
------
A compressed .npz with aligned arrays:
    text_probs      (N, 15) float32
    voice_probs     (N, 15) float32
    conf_features   (N, F)  float32
    modality_mask   (N, 2)  float32   [text_present, voice_present]
    targets         (N,)    int64
plus a sidecar .json manifest describing how it was built.

Usage
-----
    # Smoke-test the dataset loader only (fast, no model inference):
    python fusion_data.py --smoke-test

    # Build the full fusion dataset from MELD:
    python fusion_data.py --split train --limit 0 --out ./data/fusion/meld_train.npz

    # Quick end-to-end build on 50 samples (sanity before the full run):
    python fusion_data.py --split train --limit 50 --out ./data/fusion/meld_smoke.npz
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────
# Emotion system — single source of truth.
# Must match Phase 1A / 1B / config.py EXACTLY (order matters:
# the probability vectors are positional).
# ─────────────────────────────────────────────────────────────
AETHER_EMOTIONS = [
    "happy", "sad", "angry", "calm", "anxious",
    "energetic", "focused", "nostalgic", "romantic",
    "melancholic", "confident", "hopeful", "frustrated",
    "lonely", "dreamy",
]
NUM_EMOTIONS = len(AETHER_EMOTIONS)  # 15
EMOTION_TO_ID = {e: i for i, e in enumerate(AETHER_EMOTIONS)}
ID_TO_EMOTION = {i: e for i, e in enumerate(AETHER_EMOTIONS)}

# ─────────────────────────────────────────────────────────────
# MELD gold-label → Aether mapping.
#
# CRITICAL: this uses the SAME convention Phase 1B was trained on,
# so text-model targets and voice-model outputs live in one label
# universe. MELD's 7 labels are: anger, disgust, fear, joy,
# neutral, sadness, surprise.
#
#   anger    → angry       (voice-supported)
#   disgust  → frustrated  (Phase 1B: disgust→frustrated)
#   fear     → anxious     (Phase 1B: fearful→anxious)
#   joy      → happy       (voice-supported)
#   neutral  → calm        (Phase 1B: neutral→calm)
#   sadness  → sad         (voice-supported)
#   surprise → dreamy      (Phase 1B: surprised→dreamy)
#
# Result: MELD gold targets span exactly 7 of the 15 Aether emotions.
# The other 8 (energetic, focused, nostalgic, romantic, melancholic,
# confident, hopeful, lonely) receive NO gold supervision from MELD —
# this is expected and handled downstream by the distillation prior +
# modality masking in the ML fuser.
# ─────────────────────────────────────────────────────────────
MELD_TO_AETHER: dict[str, str] = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "frustrated",
    "fear": "anxious",
    "fearful": "anxious",
    "joy": "happy",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "calm",
    "sadness": "sad",
    "sad": "sad",
    "surprise": "dreamy",
    "surprised": "dreamy",
}

# The subset of Aether emotions MELD can actually supervise.
MELD_SUPERVISED_EMOTIONS = sorted(
    {EMOTION_TO_ID[v] for v in MELD_TO_AETHER.values()}
)

# HuggingFace repo for MELD with pre-extracted 16 kHz mono WAV audio.
MELD_HF_REPO = "ajyy/MELD_audio"
MELD_HF_CONFIG = "MELD_Audio"

TARGET_SAMPLE_RATE = 16000
_EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Data containers
# ═══════════════════════════════════════════════════════════════════
@dataclass
class RawSample:
    """One multimodal utterance, dataset-agnostic.

    This is the seam between dataset-specific loaders (MELD, MOSEI, ...)
    and the model-inference core. Any dataset adapter must yield these.

    Attributes:
        transcript: The utterance text (may be empty → text modality absent).
        audio_path: Path to a 16 kHz mono WAV (may be None → voice absent).
        gold_emotion_id: Aether emotion id in [0, 14], or None if unmappable.
        source: Dataset name, for the manifest (e.g. "MELD").
        uid: Stable identifier for debugging/caching.
    """
    transcript: str
    audio_path: Optional[str]
    gold_emotion_id: Optional[int]
    source: str = "MELD"
    uid: str = ""


@dataclass
class FusionExample:
    """One fully-featurized fusion training example (distribution-space)."""
    text_probs: np.ndarray      # (15,) float32
    voice_probs: np.ndarray     # (15,) float32
    conf_features: np.ndarray   # (F,)  float32
    modality_mask: np.ndarray   # (2,)  float32  [text_present, voice_present]
    target: int                 # gold Aether id


@dataclass
class BuildStats:
    """Bookkeeping for the manifest and console summary."""
    seen: int = 0
    kept: int = 0
    skipped_no_gold: int = 0
    skipped_voice_fail: int = 0
    skipped_text_empty: int = 0
    skipped_both_missing: int = 0
    errors: int = 0
    per_target: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Confidence / agreement featurizer
# ═══════════════════════════════════════════════════════════════════
# These features are the entire justification for a learnable fuser
# beating a fixed 0.6/0.4 rule: they let the model discount a modality
# when that modality is uncertain (high entropy / low peak) or absent,
# and react to how much the two modalities (dis)agree. This is where a
# text/voice conflict becomes something the network can actually see.

# Feature layout (documented so downstream code and the interview
# writeup can reference exact indices):
CONF_FEATURE_NAMES = [
    "text_entropy_norm",    # 0: normalized entropy of text distribution [0,1]
    "voice_entropy_norm",   # 1: normalized entropy of voice distribution [0,1]
    "text_max_prob",        # 2: peak probability of text distribution
    "voice_max_prob",       # 3: peak probability of voice distribution
    "text_top2_gap",        # 4: margin between top-1 and top-2 (text)
    "voice_top2_gap",       # 5: margin between top-1 and top-2 (voice)
    "agreement_argmax",     # 6: 1.0 if both argmax agree else 0.0
    "prob_cosine",          # 7: cosine similarity of the two prob vectors
    "js_divergence",        # 8: Jensen-Shannon divergence (symmetric, [0,1])
    "text_present",         # 9: modality mask (text)
    "voice_present",        # 10: modality mask (voice)
]
NUM_CONF_FEATURES = len(CONF_FEATURE_NAMES)


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector (natural log)."""
    p = np.clip(p, _EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def _normalized_entropy(p: np.ndarray) -> float:
    """Entropy scaled to [0, 1] by the max entropy log(K)."""
    if p.sum() <= _EPS:
        return 0.0
    return _entropy(p) / float(np.log(len(p)))


def _top2_gap(p: np.ndarray) -> float:
    """Margin between the two highest probabilities (peakedness proxy)."""
    if p.size < 2:
        return float(p.max()) if p.size else 0.0
    s = np.sort(p)[::-1]
    return float(s[0] - s[1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in [0, 1] (log base 2)."""
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(js / np.log(2))  # normalize to [0,1]


def compute_conf_features(
    text_probs: np.ndarray,
    voice_probs: np.ndarray,
    text_present: bool,
    voice_present: bool,
) -> np.ndarray:
    """Build the confidence/agreement feature vector for one example.

    Absent modalities contribute neutral values (zeros) so the mask,
    not spurious feature values, tells the model what's missing.

    Args:
        text_probs: (15,) text distribution (all-zero if text absent).
        voice_probs: (15,) voice distribution (all-zero if voice absent).
        text_present: whether the text modality is available.
        voice_present: whether the voice modality is available.

    Returns:
        (NUM_CONF_FEATURES,) float32 vector.
    """
    t_present = 1.0 if text_present else 0.0
    v_present = 1.0 if voice_present else 0.0

    t_entropy = _normalized_entropy(text_probs) if text_present else 0.0
    v_entropy = _normalized_entropy(voice_probs) if voice_present else 0.0
    t_max = float(text_probs.max()) if text_present else 0.0
    v_max = float(voice_probs.max()) if voice_present else 0.0
    t_gap = _top2_gap(text_probs) if text_present else 0.0
    v_gap = _top2_gap(voice_probs) if voice_present else 0.0

    if text_present and voice_present:
        agree = 1.0 if int(np.argmax(text_probs)) == int(np.argmax(voice_probs)) else 0.0
        cos = _cosine(text_probs, voice_probs)
        js = _js_divergence(text_probs, voice_probs)
    else:
        agree, cos, js = 0.0, 0.0, 0.0

    return np.array(
        [t_entropy, v_entropy, t_max, v_max, t_gap, v_gap,
         agree, cos, js, t_present, v_present],
        dtype=np.float32,
    )


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: Model wrappers (lazy — only imported when actually building)
# ═══════════════════════════════════════════════════════════════════
# We import the REAL Phase 1A / 1B inference classes. Their locations
# differ between the Colab/Drive layout and a local checkout, so we try
# a few import paths and fail with an actionable message.

class ModelBundle:
    """Holds the loaded Phase 1A + 1B detectors and exposes the two
    calls fusion needs: text→probs and audio→(probs, transcript).

    Loading is expensive (emotion2vec+, Whisper), so we build this once
    and reuse it across the whole dataset.
    """

    def __init__(
        self,
        text_model_dir: Optional[str] = None,
        voice_model_dir: Optional[str] = None,
        load_whisper: bool = False,
        voice_timeout_sec: float = 60.0,
    ):
        self.text_detector = self._load_text_detector(text_model_dir)
        self.voice_detector = self._load_voice_detector(voice_model_dir, load_whisper)
        self._load_whisper = load_whisper
        # A single pathological clip (very long/corrupt audio) can make
        # emotion2vec hang for minutes on CPU, blocking the whole build.
        # If one clip exceeds this many seconds, we skip it (treated as a
        # voice failure → the clip is dropped). A handful of skips out of
        # thousands is negligible and keeps the build unblockable.
        self.voice_timeout_sec = float(voice_timeout_sec)

    # ── Phase 1A ──
    @staticmethod
    def _load_text_detector(model_dir: Optional[str]):
        TextEmotionDetector = _import_symbol(
            candidates=[
                ("phase_1a_text_emotion.inference", "TextEmotionDetector"),
                ("inference", "TextEmotionDetector"),
                ("text_inference", "TextEmotionDetector"),
            ],
            what="Phase 1A TextEmotionDetector",
        )
        try:
            return TextEmotionDetector(model_path=model_dir) if model_dir else TextEmotionDetector()
        except TypeError:
            # Older signature without keyword
            return TextEmotionDetector(model_dir) if model_dir else TextEmotionDetector()

    # ── Phase 1B ──
    @staticmethod
    def _load_voice_detector(model_dir: Optional[str], load_whisper: bool):
        VoiceEmotionDetector = _import_symbol(
            candidates=[
                ("phase_1b_voice_emotion.inference", "VoiceEmotionDetector"),
                ("voice_inference", "VoiceEmotionDetector"),
                ("inference", "VoiceEmotionDetector"),
            ],
            what="Phase 1B VoiceEmotionDetector",
        )
        kwargs = {"load_whisper": load_whisper}
        if model_dir:
            kwargs["model_dir"] = model_dir
        try:
            return VoiceEmotionDetector(**kwargs)
        except TypeError:
            # Fall back to positional / minimal signature
            return VoiceEmotionDetector(model_dir, load_whisper) if model_dir else VoiceEmotionDetector()

    # ── The two calls fusion actually needs ──
    def text_vector(self, text: str) -> np.ndarray:
        """15-dim text probability vector in AETHER_EMOTIONS order."""
        vec = self.text_detector.get_emotion_vector(text)
        return _as_prob_vector(vec)

    def voice_vector_and_transcript(self, audio_path: str) -> tuple[np.ndarray, Optional[str]]:
        """Return (15-dim voice probs, transcript-or-None).

        Uses predict() once so we get the acoustic distribution AND the
        Whisper transcript from a single emotion2vec+/Whisper pass.
        Phase 1B returns zeros(15) on extraction failure — the caller
        must treat an all-zero vector as a failed voice sample.

        The predict() call is run under a hard timeout (voice_timeout_sec):
        if a single clip takes too long (very long/corrupt audio hanging
        emotion2vec on CPU), we abandon it and return zeros so the build
        keeps moving instead of stalling indefinitely.
        """
        result = self._predict_with_timeout(audio_path)
        if not isinstance(result, dict) or "error" in result:
            return np.zeros(NUM_EMOTIONS, dtype=np.float32), None

        probs_dict = result.get("probabilities", {})
        vec = np.array(
            [probs_dict.get(e, 0.0) for e in AETHER_EMOTIONS],
            dtype=np.float32,
        )
        transcript = result.get("transcription")
        if isinstance(transcript, str):
            transcript = transcript.strip() or None
        else:
            transcript = None
        return _as_prob_vector(vec), transcript

    def _predict_with_timeout(self, audio_path: str):
        """Run voice_detector.predict() but give up after voice_timeout_sec.

        Uses a background thread rather than signals so it works reliably on
        macOS and inside notebooks. If the timeout fires, the worker thread is
        left to finish/die on its own (daemon) and we move on; the clip is
        treated as a voice failure by the caller.

        Returns:
            The predict() result dict, or None if it timed out or errored.
        """
        import threading

        holder: dict = {}

        def _worker():
            try:
                holder["result"] = self.voice_detector.predict(audio_path)
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(self.voice_timeout_sec)

        if t.is_alive():
            # Timed out — abandon this clip.
            print(f"   [timeout>{self.voice_timeout_sec:.0f}s] skipping slow clip: "
                  f"{os.path.basename(str(audio_path))}", flush=True)
            return None
        if "error" in holder:
            return None
        return holder.get("result")


def _import_symbol(candidates, what: str):
    """Try several (module, attribute) pairs; return the first that imports.

    Raises a single actionable ImportError if none succeed.
    """
    errors = []
    for module_name, attr in candidates:
        try:
            module = __import__(module_name, fromlist=[attr])
            return getattr(module, attr)
        except Exception as exc:  # ImportError, AttributeError, etc.
            errors.append(f"    {module_name}.{attr}: {exc}")
    raise ImportError(
        f"Could not import {what}. Tried:\n" + "\n".join(errors) +
        "\n\nMake sure the Phase 1A/1B inference.py files are on the Python "
        "path (e.g. copy them next to this script, or add their folders to "
        "PYTHONPATH). On Colab, they live under the mounted Drive at "
        "Aether_models/."
    )


def _as_prob_vector(vec) -> np.ndarray:
    """Coerce a model output to a clean (15,) float32 vector.

    Does NOT renormalize a genuine all-zero vector (that signals a failed
    voice sample and must stay all-zero so the caller can detect it).
    """
    arr = np.asarray(vec, dtype=np.float32).flatten()
    if arr.shape[0] != NUM_EMOTIONS:
        raise ValueError(
            f"Expected a {NUM_EMOTIONS}-dim emotion vector, got shape {arr.shape}. "
            "This usually means the model's label order does not match "
            "AETHER_EMOTIONS — stop and fix before building, or the fusion "
            "targets will be misaligned."
        )
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    return arr


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: MELD adapter  (dataset-specific — the MOSEI seam mirrors this)
# ═══════════════════════════════════════════════════════════════════
def _map_meld_label(raw_label) -> Optional[int]:
    """Map a MELD emotion label (str or int-indexed) to an Aether id."""
    if raw_label is None:
        return None
    if isinstance(raw_label, str):
        key = raw_label.lower().strip()
        aether = MELD_TO_AETHER.get(key)
        return EMOTION_TO_ID[aether] if aether else None
    return None


def _diagnose_meld_error(err: Optional[BaseException]) -> str:
    """Map a MELD load failure to the SPECIFIC fix for that failure.

    Earlier this function always blamed 'datasets >= 4.0', which was only
    right for one of several possible failures. Now we read the actual
    error text and point at the true cause.
    """
    text = f"{type(err).__name__}: {err}".lower() if err else ""

    if "soundfile" in text:
        return (
            "CAUSE: the `soundfile` package is missing — `datasets` needs it "
            "to decode the audio.\n"
            "FIX:   pip3 install soundfile\n"
            "       (also recommended: pip3 install librosa)"
        )
    if "datasets scripts are no longer supported" in text or "loading script" in text:
        return (
            "CAUSE: your `datasets` version (>= 4.0) refuses to run this "
            "repo's loading script.\n"
            "FIX:   pip install 'datasets<4.0'\n"
            "       (fallback: download MELD.Raw.tar.gz, extract 16 kHz mono "
            "WAVs, and load locally.)"
        )
    if "trust_remote_code" in text:
        return (
            "CAUSE: this `datasets` version no longer accepts trust_remote_code "
            "for script datasets.\n"
            "FIX:   pip install 'datasets<4.0'"
        )
    if "connection" in text or "timeout" in text or "hf" in text:
        return (
            "CAUSE: looks network-related (HuggingFace Hub unreachable / rate "
            "limited).\n"
            "FIX:   check your connection and retry; optionally set an HF token "
            "for higher limits."
        )
    # Unknown — hand back the generic fallback menu.
    return (
        "CAUSE: unrecognized. General fallbacks:\n"
        "  (a) pin an older loader:  pip install 'datasets<4.0'\n"
        "  (b) ensure audio deps:    pip3 install soundfile librosa\n"
        "  (c) download MELD.Raw.tar.gz and extract 16 kHz mono WAVs locally.\n"
        "Re-run `python fusion_data.py --smoke-test` after each attempt."
    )


def iter_meld_samples(
    split: str = "train",
    limit: int = 0,
    cache_dir: Optional[str] = None,
    skip_rows: int = 0,
) -> Iterator[RawSample]:
    """Yield RawSample objects from the MELD dataset.

    Loading MELD via `ajyy/MELD_audio` relies on a HuggingFace *loading
    script*, which newer `datasets` (>=4.0) may refuse to run. If that
    happens this raises a clear error pointing at the fallback, rather
    than dying with an opaque stack trace.

    Args:
        split: "train", "validation"/"dev", or "test".
        limit: if > 0, stop after yielding this many mappable samples
               (for smoke tests).
        cache_dir: optional HF datasets cache directory.

    Yields:
        RawSample per utterance, with a temp WAV written for the audio.
    """
    try:
        from datasets import load_dataset, Audio
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "The `datasets` library is required to load MELD. "
            "Install with: pip install datasets soundfile"
        ) from exc

    split_aliases = {
        "train": ["train"],
        "validation": ["validation", "dev", "valid"],
        "dev": ["validation", "dev", "valid"],
        "test": ["test"],
    }
    wanted = split_aliases.get(split, [split])

    ds = None
    load_err = None
    for split_name in wanted:
        try:
            ds = load_dataset(
                MELD_HF_REPO, MELD_HF_CONFIG,
                split=split_name, cache_dir=cache_dir,
                trust_remote_code=True,
            )
            break
        except Exception as exc:  # noqa: BLE001
            load_err = exc
            continue

    if ds is None:
        raise RuntimeError(
            f"Failed to load MELD ({MELD_HF_REPO}, config={MELD_HF_CONFIG}, "
            f"split={split}).\n\n"
            f"REAL underlying error:\n    {type(load_err).__name__}: {load_err}\n\n"
            + _diagnose_meld_error(load_err)
        )

    # Ensure audio is decoded at the target sample rate.
    audio_col = "audio" if "audio" in ds.column_names else None
    if audio_col is not None:
        try:
            ds = ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SAMPLE_RATE))
        except Exception:
            pass  # some builds already provide arrays at 16k

    # Detect label / text column names defensively.
    label_col = _first_present(ds.column_names,
                               ["emotion", "Emotion", "label", "labels"])
    text_col = _first_present(ds.column_names,
                              ["text", "utterance", "Utterance", "sentence", "transcript"])

    if label_col is None:
        raise KeyError(
            f"Could not find an emotion label column in MELD. "
            f"Columns present: {ds.column_names}"
        )

    import tempfile
    import soundfile as sf

    yielded = 0
    for idx, row in enumerate(ds):
        # Resume support: skip rows already processed in a prior run.
        # This happens BEFORE any audio is decoded/materialized, so resuming
        # is cheap (no wasted work on already-done clips).
        if skip_rows and idx < skip_rows:
            continue

        raw_label = row.get(label_col)
        gold = _map_meld_label(raw_label)
        # We still yield unmapped rows (gold=None) so the builder can count
        # them; it will skip them. But to honor `limit` on *usable* samples
        # we only increment yielded for mappable ones.

        transcript = ""
        if text_col is not None:
            t = row.get(text_col, "")
            transcript = t.strip() if isinstance(t, str) else ""

        audio_path = None
        if audio_col is not None:
            audio = row.get(audio_col)
            audio_path = _materialize_wav(audio, tempfile, sf)

        uid = f"MELD-{split}-{idx}"
        yield RawSample(
            transcript=transcript,
            audio_path=audio_path,
            gold_emotion_id=gold,
            source="MELD",
            uid=uid,
        )

        if gold is not None:
            yielded += 1
            if limit and yielded >= limit:
                return


def _materialize_wav(audio, tempfile_mod, sf_mod) -> Optional[str]:
    """Turn a HF Audio value into a real 16 kHz mono WAV path on disk.

    Phase 1B's predict() takes a file path, so decoded arrays must be
    written out. Returns None if audio can't be materialized.
    """
    try:
        if audio is None:
            return None
        # Case 1: HF decoded dict {"array": np.ndarray, "sampling_rate": int}
        if isinstance(audio, dict) and "array" in audio:
            array = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate", TARGET_SAMPLE_RATE))
            if array.ndim > 1:  # stereo → mono
                array = array.mean(axis=1)
            with tempfile_mod.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf_mod.write(tmp.name, array, sr)
                return tmp.name
        # Case 2: already a path
        if isinstance(audio, str) and Path(audio).exists():
            return audio
        if isinstance(audio, dict) and audio.get("path") and Path(audio["path"]).exists():
            return audio["path"]
    except Exception:
        return None
    return None


def _first_present(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: Core builder  (dataset-agnostic)
# ═══════════════════════════════════════════════════════════════════
def build_fusion_dataset(
    samples: Iterable[RawSample],
    models: ModelBundle,
    use_whisper_transcript_fallback: bool = False,
    log_every: int = 100,
) -> tuple[list[FusionExample], BuildStats]:
    """Featurize an iterable of RawSamples into FusionExamples.

    For each sample:
      * text present  ⟺ non-empty transcript  → run Phase 1A
      * voice present ⟺ audio yields a non-zero Phase 1B vector
      * drop samples with no gold label or with neither modality usable

    Args:
        samples: iterable of RawSample (from any dataset adapter).
        models: loaded ModelBundle.
        use_whisper_transcript_fallback: if a sample has no transcript but
            Whisper produced one during voice inference, use it as the text
            input. Only meaningful if the voice detector loaded Whisper.
        log_every: progress print cadence.

    Returns:
        (list[FusionExample], BuildStats)
    """
    out: list[FusionExample] = []
    stats = BuildStats()
    t0 = time.time()

    for sample in samples:
        stats.seen += 1

        if sample.gold_emotion_id is None:
            stats.skipped_no_gold += 1
            _maybe_log(stats, t0, log_every)
            continue

        # ── Voice modality ──
        voice_probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        voice_present = False
        whisper_text: Optional[str] = None
        if sample.audio_path:
            try:
                voice_probs, whisper_text = models.voice_vector_and_transcript(sample.audio_path)
                # zeros(15) is Phase 1B's failure sentinel
                voice_present = bool(voice_probs.sum() > _EPS)
                if not voice_present:
                    stats.skipped_voice_fail += 1
            except Exception:
                stats.errors += 1
                stats.skipped_voice_fail += 1
                voice_probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
                voice_present = False

        # ── Text modality ──
        transcript = sample.transcript
        if (not transcript) and use_whisper_transcript_fallback and whisper_text:
            transcript = whisper_text

        text_probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
        text_present = False
        if transcript:
            try:
                text_probs = models.text_vector(transcript)
                text_present = bool(text_probs.sum() > _EPS)
                if not text_present:
                    stats.skipped_text_empty += 1
            except Exception:
                stats.errors += 1
                text_probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
                text_present = False

        # ── Must have at least one usable modality ──
        if not text_present and not voice_present:
            stats.skipped_both_missing += 1
            _maybe_log(stats, t0, log_every)
            continue

        conf = compute_conf_features(text_probs, voice_probs, text_present, voice_present)
        mask = np.array(
            [1.0 if text_present else 0.0, 1.0 if voice_present else 0.0],
            dtype=np.float32,
        )

        out.append(FusionExample(
            text_probs=text_probs,
            voice_probs=voice_probs,
            conf_features=conf,
            modality_mask=mask,
            target=int(sample.gold_emotion_id),
        ))
        stats.kept += 1
        stats.per_target[sample.gold_emotion_id] = (
            stats.per_target.get(sample.gold_emotion_id, 0) + 1
        )
        _maybe_log(stats, t0, log_every)

    return out, stats


def _maybe_log(stats: BuildStats, t0: float, log_every: int) -> None:
    if log_every and stats.seen % log_every == 0:
        rate = stats.seen / max(time.time() - t0, _EPS)
        print(f"   … seen {stats.seen:>6} | kept {stats.kept:>6} "
              f"| {rate:5.1f}/s", flush=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: Serialization
# ═══════════════════════════════════════════════════════════════════
def save_fusion_dataset(
    examples: list[FusionExample],
    out_path: str,
    stats: BuildStats,
    source: str,
    split: str,
) -> None:
    """Write examples to a compressed .npz plus a .json manifest."""
    if not examples:
        raise ValueError("No examples to save — the build produced 0 kept samples.")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    text_probs = np.stack([e.text_probs for e in examples]).astype(np.float32)
    voice_probs = np.stack([e.voice_probs for e in examples]).astype(np.float32)
    conf = np.stack([e.conf_features for e in examples]).astype(np.float32)
    mask = np.stack([e.modality_mask for e in examples]).astype(np.float32)
    targets = np.array([e.target for e in examples], dtype=np.int64)

    np.savez_compressed(
        out,
        text_probs=text_probs,
        voice_probs=voice_probs,
        conf_features=conf,
        modality_mask=mask,
        targets=targets,
    )

    manifest = {
        "project": "Aether",
        "phase": "1C",
        "component": "Fusion Dataset",
        "source": source,
        "split": split,
        "num_examples": len(examples),
        "num_emotions": NUM_EMOTIONS,
        "emotions": AETHER_EMOTIONS,
        "conf_feature_names": CONF_FEATURE_NAMES,
        "modality_mask_order": ["text_present", "voice_present"],
        "meld_supervised_emotion_ids": MELD_SUPERVISED_EMOTIONS,
        "meld_supervised_emotions": [ID_TO_EMOTION[i] for i in MELD_SUPERVISED_EMOTIONS],
        "label_mapping_convention": MELD_TO_AETHER,
        "regime_counts": _regime_counts(mask),
        "per_target_counts": {
            ID_TO_EMOTION[k]: int(v) for k, v in sorted(stats.per_target.items())
        },
        "build_stats": {
            "seen": stats.seen,
            "kept": stats.kept,
            "skipped_no_gold": stats.skipped_no_gold,
            "skipped_voice_fail": stats.skipped_voice_fail,
            "skipped_text_empty": stats.skipped_text_empty,
            "skipped_both_missing": stats.skipped_both_missing,
            "errors": stats.errors,
        },
    }
    manifest_path = out.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Saved {len(examples):,} fusion examples → {out}")
    print(f"✅ Manifest → {manifest_path}")


def _regime_counts(mask: np.ndarray) -> dict:
    """Count examples by modality regime (for ablation's per-regime metrics)."""
    text_p = mask[:, 0] > 0.5
    voice_p = mask[:, 1] > 0.5
    return {
        "text_and_voice": int(np.sum(text_p & voice_p)),
        "text_only": int(np.sum(text_p & ~voice_p)),
        "voice_only": int(np.sum(~text_p & voice_p)),
    }


def load_fusion_dataset(npz_path: str) -> dict:
    """Load a previously-built fusion dataset .npz into a dict of arrays."""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


# ═══════════════════════════════════════════════════════════════════
# SECTION 6b: Streaming builder  (crash-proof + resumable)
# ═══════════════════════════════════════════════════════════════════
# The in-memory build (build_fusion_dataset) is fine for small runs, but for
# the full ~10k-clip build it must NOT (a) leak temp WAV files, or (b) let
# memory creep over a long run. This streaming version writes each kept
# example straight to a JSONL file (one line per clip), deletes each temp WAV
# immediately after use, clears memory periodically, and records progress so a
# crash/disconnect can resume instead of restarting.

def _uid_row_index(uid: str) -> Optional[int]:
    """Extract the dataset row index from a RawSample uid like 'MELD-train-123'."""
    try:
        return int(str(uid).split("-")[-1])
    except Exception:
        return None


def _cleanup_temp_file(path: Optional[str]) -> None:
    """Delete a temp WAV we materialized (only if it lives in the temp dir)."""
    if not path:
        return
    try:
        import tempfile
        tmpdir = tempfile.gettempdir()
        if str(path).startswith(tmpdir) and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass  # best-effort; never crash the build over cleanup


def build_fusion_dataset_streaming(
    samples: Iterable[RawSample],
    models: ModelBundle,
    jsonl_path: str,
    checkpoint_path: Optional[str] = None,
    use_whisper_transcript_fallback: bool = False,
    log_every: int = 100,
    flush_every: int = 100,
    gc_every: int = 300,
) -> BuildStats:
    """Featurize samples and STREAM each kept example to a JSONL file.

    Memory stays flat (nothing accumulates), temp WAVs are deleted as we go,
    and progress is checkpointed so the run can resume.

    Args:
        samples: iterable of RawSample (from a dataset adapter).
        models: loaded ModelBundle.
        jsonl_path: file to append examples to (one JSON object per line).
        checkpoint_path: if given, the highest processed row index is written
            here periodically, so a later run can resume via skip_rows.
        use_whisper_transcript_fallback: use Whisper transcript when the
            dataset transcript is empty (needs load_whisper=True on the voice model).
        log_every / flush_every / gc_every: cadences for logging, disk flush,
            and garbage collection.

    Returns:
        BuildStats for this run (counts of kept/skipped/etc.).
    """
    import gc
    import json as _json

    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    stats = BuildStats()
    t0 = time.time()
    last_idx = -1

    # Append mode: on resume we add to what's already there.
    f = open(jsonl_path, "a")
    try:
        for sample in samples:
            stats.seen += 1
            ridx = _uid_row_index(sample.uid)
            if ridx is not None:
                last_idx = ridx

            row_obj = None  # set only if we keep this example

            if sample.gold_emotion_id is None:
                stats.skipped_no_gold += 1
            else:
                # ── Voice ──
                voice_probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
                voice_present = False
                whisper_text: Optional[str] = None
                if sample.audio_path:
                    try:
                        voice_probs, whisper_text = models.voice_vector_and_transcript(sample.audio_path)
                        voice_present = bool(voice_probs.sum() > _EPS)
                        if not voice_present:
                            stats.skipped_voice_fail += 1
                    except Exception:
                        stats.errors += 1
                        stats.skipped_voice_fail += 1
                        voice_probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
                        voice_present = False

                # ── Text ──
                transcript = sample.transcript
                if (not transcript) and use_whisper_transcript_fallback and whisper_text:
                    transcript = whisper_text
                text_probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
                text_present = False
                if transcript:
                    try:
                        text_probs = models.text_vector(transcript)
                        text_present = bool(text_probs.sum() > _EPS)
                    except Exception:
                        stats.errors += 1
                        text_probs = np.zeros(NUM_EMOTIONS, dtype=np.float32)
                        text_present = False

                if text_present or voice_present:
                    conf = compute_conf_features(text_probs, voice_probs, text_present, voice_present)
                    row_obj = {
                        "uid": sample.uid,   # for dedup at conversion time
                        "t": [round(float(x), 6) for x in text_probs],
                        "v": [round(float(x), 6) for x in voice_probs],
                        "c": [round(float(x), 6) for x in conf],
                        "m": [1.0 if text_present else 0.0, 1.0 if voice_present else 0.0],
                        "y": int(sample.gold_emotion_id),
                    }
                else:
                    stats.skipped_both_missing += 1

            # Always delete the temp WAV (the model already read it).
            _cleanup_temp_file(sample.audio_path)

            if row_obj is not None:
                f.write(_json.dumps(row_obj) + "\n")
                stats.kept += 1
                stats.per_target[sample.gold_emotion_id] = (
                    stats.per_target.get(sample.gold_emotion_id, 0) + 1
                )

            # ── Periodic housekeeping ──
            if log_every and stats.seen % log_every == 0:
                rate = stats.seen / max(time.time() - t0, _EPS)
                print(f"   … seen {stats.seen:>6} | kept {stats.kept:>6} "
                      f"| {rate:5.1f}/s", flush=True)
            if stats.seen % flush_every == 0:
                f.flush()
                if checkpoint_path:
                    with open(checkpoint_path, "w") as cp:
                        _json.dump(
                            {"last_row_idx": last_idx, "kept": stats.kept, "seen": stats.seen},
                            cp,
                        )
            if stats.seen % gc_every == 0:
                gc.collect()
    finally:
        f.flush()
        f.close()
        if checkpoint_path:
            with open(checkpoint_path, "w") as cp:
                import json as _j
                _j.dump({"last_row_idx": last_idx, "kept": stats.kept, "seen": stats.seen}, cp)

    return stats


def read_checkpoint(checkpoint_path: str) -> int:
    """Return skip_rows for a resume (last processed row + 1), or 0 if none."""
    try:
        import json as _json
        with open(checkpoint_path) as cp:
            data = _json.load(cp)
        last = int(data.get("last_row_idx", -1))
        return max(last + 1, 0)
    except Exception:
        return 0


def jsonl_to_npz(
    jsonl_path: str,
    npz_path: str,
    source: str = "MELD",
    split: str = "train",
) -> dict:
    """Convert a streamed JSONL file into the final compressed .npz + manifest.

    The JSONL is small (a few MB even for the full set), so this loads it fully
    and writes the same array layout the rest of Phase 1C expects.
    """
    import json as _json

    # Read rows, deduping by uid (keep the LAST occurrence per clip). This
    # makes the conversion robust even if a resumed/chunked build appended the
    # same clip more than once — one row per source utterance, guaranteed.
    by_uid: dict = {}
    ordered_uids: list = []
    anon_rows: list = []  # rows without a uid (older files) kept as-is
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _json.loads(line)
            uid = r.get("uid")
            if uid is None:
                anon_rows.append(r)
                continue
            if uid not in by_uid:
                ordered_uids.append(uid)
            by_uid[uid] = r

    rows = [by_uid[u] for u in ordered_uids] + anon_rows

    text_probs, voice_probs, conf, mask, targets = [], [], [], [], []
    for r in rows:
        text_probs.append(r["t"])
        voice_probs.append(r["v"])
        conf.append(r["c"])
        mask.append(r["m"])
        targets.append(r["y"])

    if not targets:
        raise ValueError(f"No examples found in {jsonl_path} — nothing to convert.")

    tp = np.array(text_probs, dtype=np.float32)
    vp = np.array(voice_probs, dtype=np.float32)
    cf = np.array(conf, dtype=np.float32)
    mk = np.array(mask, dtype=np.float32)
    ty = np.array(targets, dtype=np.int64)

    out = Path(npz_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, text_probs=tp, voice_probs=vp,
        conf_features=cf, modality_mask=mk, targets=ty,
    )

    # Manifest (mirrors save_fusion_dataset).
    per_target = {}
    for y in ty.tolist():
        per_target[y] = per_target.get(y, 0) + 1

    manifest = {
        "project": "Aether", "phase": "1C", "component": "Fusion Dataset (streamed)",
        "source": source, "split": split,
        "num_examples": int(ty.shape[0]),
        "num_emotions": NUM_EMOTIONS, "emotions": AETHER_EMOTIONS,
        "conf_feature_names": CONF_FEATURE_NAMES,
        "modality_mask_order": ["text_present", "voice_present"],
        "meld_supervised_emotion_ids": MELD_SUPERVISED_EMOTIONS,
        "meld_supervised_emotions": [ID_TO_EMOTION[i] for i in MELD_SUPERVISED_EMOTIONS],
        "regime_counts": _regime_counts(mk),
        "per_target_counts": {ID_TO_EMOTION[k]: int(v) for k, v in sorted(per_target.items())},
    }
    with open(out.with_suffix(".manifest.json"), "w") as mf:
        _json.dump(manifest, mf, indent=2)

    print(f"✅ Wrote {ty.shape[0]:,} examples → {out}")
    print(f"✅ Manifest → {out.with_suffix('.manifest.json')}")
    print(f"   Regimes: {manifest['regime_counts']}")
    print(f"   Per-target: {manifest['per_target_counts']}")
    return manifest


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: Smoke test  (no model inference — probes only the data path)
# ═══════════════════════════════════════════════════════════════════
def smoke_test(limit: int = 8) -> int:
    """Verify MELD loads and labels map, WITHOUT loading the heavy models.

    This isolates the single biggest risk (the MELD loading-script issue)
    so you find out in seconds, not after emotion2vec+ has loaded.

    Returns process exit code (0 = ok).
    """
    print("═" * 60)
    print("SMOKE TEST — MELD load + label mapping (no model inference)")
    print("═" * 60)
    try:
        n = 0
        label_hist: dict[str, int] = {}
        for sample in iter_meld_samples(split="train", limit=limit):
            n += 1
            name = ID_TO_EMOTION.get(sample.gold_emotion_id, "UNMAPPED")
            label_hist[name] = label_hist.get(name, 0) + 1
            has_audio = "yes" if sample.audio_path else "NO"
            preview = (sample.transcript[:45] + "…") if len(sample.transcript) > 45 else sample.transcript
            print(f"   [{n}] target={name:<11} audio={has_audio:<3} text=\"{preview}\"")
        print("\n   Label histogram:", label_hist)
        print(f"\n✅ MELD path OK — pulled {n} mappable samples.")
        print("   Emotions MELD can supervise:",
              [ID_TO_EMOTION[i] for i in MELD_SUPERVISED_EMOTIONS])
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Smoke test FAILED:\n{exc}\n")
        traceback.print_exc()
        print("\nThis is the MELD loader issue we anticipated. See the fallback "
              "instructions in the error above.")
        return 1


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: CLI
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Aether Phase 1C fusion dataset from MELD."
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Probe MELD loading + label mapping only (no models).")
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "dev", "test"],
                        help="MELD split to build from.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max mappable samples (0 = all). Use small for a trial run.")
    parser.add_argument("--out", default="./data/fusion/meld_train.npz",
                        help="Output .npz path.")
    parser.add_argument("--text-model-dir", default=None,
                        help="Override Phase 1A model dir.")
    parser.add_argument("--voice-model-dir", default=None,
                        help="Override Phase 1B model dir.")
    parser.add_argument("--whisper-fallback", action="store_true",
                        help="Use Whisper transcript as text when the dataset "
                             "transcript is missing (loads Whisper in Phase 1B).")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace datasets cache dir.")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if args.smoke_test:
        return smoke_test()

    print("═" * 60)
    print(f"BUILD — Aether Fusion Dataset  (MELD / {args.split})")
    print("═" * 60)

    # 1) Load models once (heavy).
    print("\n🧠 Loading Phase 1A + 1B models …")
    load_whisper = args.whisper_fallback
    try:
        models = ModelBundle(
            text_model_dir=args.text_model_dir,
            voice_model_dir=args.voice_model_dir,
            load_whisper=load_whisper,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Could not load models: {exc}")
        traceback.print_exc()
        return 1
    print("✅ Models ready.")

    # 2) Stream MELD → featurize.
    print(f"\n🔄 Building from MELD (split={args.split}, "
          f"limit={args.limit or 'ALL'}) …")
    samples = iter_meld_samples(
        split=args.split, limit=args.limit, cache_dir=args.cache_dir,
    )
    examples, stats = build_fusion_dataset(
        samples, models,
        use_whisper_transcript_fallback=args.whisper_fallback,
        log_every=args.log_every,
    )

    # 3) Report.
    print("\n" + "─" * 60)
    print("BUILD SUMMARY")
    print("─" * 60)
    print(f"   seen                : {stats.seen}")
    print(f"   kept                : {stats.kept}")
    print(f"   skipped (no gold)   : {stats.skipped_no_gold}")
    print(f"   skipped (voice fail): {stats.skipped_voice_fail}")
    print(f"   skipped (text empty): {stats.skipped_text_empty}")
    print(f"   skipped (both gone) : {stats.skipped_both_missing}")
    print(f"   errors              : {stats.errors}")
    if stats.kept:
        print("\n   Per-target kept counts:")
        for tid, c in sorted(stats.per_target.items()):
            print(f"     {ID_TO_EMOTION[tid]:>11}: {c}")

    if not examples:
        print("\n❌ 0 examples kept — nothing to save. Check the summary above.")
        return 1

    # 4) Save.
    save_fusion_dataset(examples, args.out, stats, source="MELD", split=args.split)
    print("\n🎉 Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
