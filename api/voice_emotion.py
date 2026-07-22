"""
Aether — Voice Emotion Engine (Phase-1B inference, production wrapper)
======================================================================
Wraps the trained Phase-1B voice pipeline for the live API:

    audio  ->  emotion2vec+ large (frozen)  ->  1024-dim embedding
           ->  trained MLP head             ->  15-emotion distribution
    audio  ->  Whisper (EN/HI)              ->  transcription text

Design notes:
- Lazy loading + a thread that can warm the models on page entry, so the
  first-load cost (~30s + emotion2vec download on first run) is hidden.
- Dims are read FROM the checkpoint (or inferred from weight shapes), so it
  cannot mismatch whether the head was trained at 768 or 1024 input dim.
- Returns acoustic emotion/distribution + the Whisper transcription. The
  transcription is meant to be passed to /curate|/journey|/live as `text`,
  matching the Phase-1B notebook's dual-signal design.
"""
from __future__ import annotations

import os
import threading
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Canonical 15-emotion order (index -> label) — must match training.
AETHER_EMOTIONS = [
    "happy", "sad", "angry", "calm", "anxious", "energetic", "focused",
    "nostalgic", "romantic", "melancholic", "confident", "hopeful",
    "frustrated", "lonely", "dreamy",
]


class VoiceEmotionHead(nn.Module):
    """MLP head: BN -> Linear -> ReLU -> Dropout (x2) -> Linear(15). Matches Phase-1B."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VoiceEmotionEngine:
    """Lazy-loading, thread-safe voice emotion + transcription engine."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        load_whisper: bool = True,
        whisper_size: str = "base",
    ):
        # Point AETHER_VOICE_CKPT at your saved head (best_model.pt / voice_emotion_head.pt).
        self.ckpt_path = Path(
            ckpt_path
            or os.environ.get("AETHER_VOICE_CKPT", "Aether_models/voice_emotion/best_model.pt")
        )
        self.load_whisper = load_whisper
        self.whisper_size = os.environ.get("AETHER_WHISPER_SIZE", whisper_size)

        self._e2v = None
        self._head: Optional[VoiceEmotionHead] = None
        self._whisper = None
        self._labels = AETHER_EMOTIONS

        self._status = "cold"       # cold | loading | ready | error
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        # Set when a load attempt finishes, whether it succeeded or failed.
        # A request that arrives mid-load waits on this instead of running
        # against half-built models (see _load).
        self._loaded_evt = threading.Event()
        # How long a caller will wait for an in-flight load before giving up.
        # The first load pulls emotion2vec + Whisper and can take ~30s, or
        # minutes on the very first run when the weights are downloaded.
        try:
            self.load_timeout_s = float(
                os.environ.get("AETHER_VOICE_LOAD_TIMEOUT", "180")
            )
        except ValueError:
            self.load_timeout_s = 180.0

    # ------------------------------------------------------------------ status
    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self) -> Optional[str]:
        return self._error

    def is_ready(self) -> bool:
        return self._status == "ready"

    # ----------------------------------------------------------------- loading
    def _resolve_dims(self, ckpt) -> tuple[dict, int, int, int]:
        """Return (state_dict, embedding_dim, hidden_dim, num_classes)."""
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            emb = int(ckpt.get("embedding_dim", 0)) or None
            hid = int(ckpt.get("hidden_dim", 0)) or None
            ncl = int(ckpt.get("num_classes", 0)) or None
        else:
            state = ckpt  # raw state_dict
            emb = hid = ncl = None

        # Fallback: infer from weight shapes (architecture is fixed).
        # net.1 = first Linear(emb, hid); net.8 = final Linear(hid//2, ncl)
        if emb is None or hid is None or ncl is None:
            first_lin = state.get("net.1.weight")
            last_lin = state.get("net.8.weight")
            if first_lin is not None:
                hid = hid or int(first_lin.shape[0])
                emb = emb or int(first_lin.shape[1])
            if last_lin is not None:
                ncl = ncl or int(last_lin.shape[0])
        emb = emb or 1024
        hid = hid or 256
        ncl = ncl or 15
        return state, emb, hid, ncl

    def _load(self) -> None:
        """Load the models once. A second caller waits for the first to finish.

        The previous version returned immediately when a load was already in
        flight. That looked like cooperation but it was a race: the warmup
        thread sets status to 'loading' on page entry, and a mic request
        arriving during those ~30s fell straight through this method and ran
        inference against a half-built engine. Because the load order is
        emotion2vec, then the head, then Whisper, the transcode and the
        embedding both succeeded and the failure only surfaced at
        `self._head(x)` as "'NoneType' object is not callable".

        Waiting is the correct behaviour: the caller genuinely needs the models,
        and blocking until they exist is what the request was asking for.
        """
        with self._lock:
            if self._status == "ready":
                return
            if self._status == "loading":
                wait_for_other = True
            else:
                wait_for_other = False
                self._status = "loading"
                self._error = None
                self._loaded_evt.clear()

        if wait_for_other:
            if not self._loaded_evt.wait(self.load_timeout_s):
                raise RuntimeError(
                    f"voice models are still loading after "
                    f"{self.load_timeout_s:.0f}s; try again in a moment"
                )
            if self._status != "ready":
                raise RuntimeError(f"voice model load failed: {self._error}")
            return

        try:
            # 1) emotion2vec+ large — frozen acoustic feature extractor
            from funasr import AutoModel as FunASRAutoModel
            self._e2v = FunASRAutoModel(model="iic/emotion2vec_plus_large", hub="hf")

            # 2) trained MLP head (dims read from checkpoint / inferred)
            if not self.ckpt_path.exists():
                raise FileNotFoundError(f"voice head checkpoint not found: {self.ckpt_path}")
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            state, emb, hid, ncl = self._resolve_dims(ckpt)
            head = VoiceEmotionHead(emb, hid, ncl)
            head.load_state_dict(state)
            head.eval()
            self._head = head

            # 3) Whisper (optional dual-signal transcription)
            if self.load_whisper:
                import whisper
                self._whisper = whisper.load_model(self.whisper_size)

            with self._lock:
                self._status = "ready"
            print("[voice] models ready (emotion2vec + head"
                  f"{' + whisper' if self._whisper is not None else ''})")
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._status = "error"
                self._error = f"{type(e).__name__}: {e}"
            print(f"[voice] model load FAILED: {self._error}")
            raise
        finally:
            # Release every waiter, success or failure, so a failed load can
            # never leave a request blocked until its timeout.
            self._loaded_evt.set()

    def _safe_load(self) -> None:
        try:
            self._load()
        except Exception:  # noqa: BLE001
            pass  # status/error already recorded

    def warmup_async(self) -> str:
        """Start background loading and return immediately. Called on page entry."""
        with self._lock:
            if self._status in ("ready", "loading"):
                return self._status
        threading.Thread(target=self._safe_load, daemon=True).start()
        return "loading"

    def ensure_loaded(self) -> None:
        """Block until the models are usable, or raise saying why they are not."""
        if self._status != "ready":
            self._load()
        # Belt and braces: never hand back a partially built engine. If this
        # ever trips, the message names the missing piece instead of failing
        # later with an opaque TypeError deep in predict().
        if self._status != "ready" or self._head is None or self._e2v is None:
            missing = "head" if self._head is None else "feature extractor"
            raise RuntimeError(
                f"voice engine not ready (status={self._status}, missing={missing})"
                + (f": {self._error}" if self._error else "")
            )

    # ------------------------------------------------------------------- audio
    @staticmethod
    def _to_wav16k(src_path: str) -> str:
        """Transcode browser audio (webm/opus/mp4/…) -> 16 kHz mono wav via ffmpeg."""
        dst = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", dst],
            capture_output=True,
        )
        if proc.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError(
                "audio transcode failed (is ffmpeg installed?): "
                + proc.stderr.decode(errors="ignore")[-300:]
            )
        return dst

    def _extract_embedding(self, wav_path: str) -> np.ndarray:
        res = self._e2v.generate(
            input=wav_path, granularity="utterance", extract_embedding=True
        )
        if not res or "feats" not in res[0]:
            raise RuntimeError("emotion2vec returned no features")
        feats = res[0]["feats"]
        if isinstance(feats, np.ndarray):
            arr = feats.flatten()
        elif hasattr(feats, "numpy"):
            arr = feats.numpy().flatten()
        else:
            arr = np.asarray(feats).flatten()
        return arr.astype(np.float32)

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, audio_path: str, transcribe: bool = True) -> dict:
        """audio file -> {emotion, distribution[15], text, confidence, labels}."""
        self.ensure_loaded()
        wav = self._to_wav16k(audio_path)
        try:
            emb = self._extract_embedding(wav)
            x = torch.from_numpy(emb).unsqueeze(0)  # [1, D]
            logits = self._head(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
            top = int(np.argmax(probs))

            text = ""
            if transcribe and self._whisper is not None:
                result = self._whisper.transcribe(wav)
                text = (result.get("text") or "").strip()

            return {
                "emotion": self._labels[top],
                "distribution": [round(float(p), 6) for p in probs],
                "text": text,
                "confidence": round(float(probs[top]), 4),
                "labels": self._labels,
            }
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass


# module-level singleton
_ENGINE: Optional[VoiceEmotionEngine] = None


def get_engine() -> VoiceEmotionEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VoiceEmotionEngine()
    return _ENGINE
