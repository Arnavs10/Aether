"""
═══════════════════════════════════════════════════════════════════
AETHER — API · Service Layer
═══════════════════════════════════════════════════════════════════
The single integration point where all seven phases become one usable service.
The FastAPI app (app.py) stays thin by delegating everything here:

    curate(...)        → Phase 3/4 matching + Phase 5 RAG "why"   (Main Feature)
    journey(...)       → Phase 6 LangGraph agent (autonomous arc)
    live_start/observe → Phase 7 drift + harmonic transition + crossfade (Fun)

Construction:
    AetherService.from_sample()          # in-memory catalog, zero setup (dev)
    AetherService.from_store_path(path)  # the real 1.2M store (one-line flip)

The optional LLM (Groq, free) is read from GROQ_API_KEY once; without it, RAG
falls back to grounded templates and the agent to its rule-based planner, so the
service always runs.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ── Path bootstrap: every phase (Phase 6 folder name may vary) ──
_ROOT = Path(__file__).resolve().parent.parent
_PHASE6 = "phase_6_agentic" if (_ROOT / "phase_6_agentic").exists() else "phase_6_agentic_ai"
for _name in ("", "phase_2_music_data", "phase_3_emotion_music_mapping",
              "phase_4_recommendation", "phase_5_rag", _PHASE6,
              "phase_7_drift_crossfade"):
    _p = _ROOT / _name if _name else _ROOT
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import AETHER_EMOTIONS                            # noqa: E402
from schema import Song                                       # Phase 2  # noqa: E402
from feature_store import FeatureStore                        # Phase 2  # noqa: E402
from playlist import PlaylistGenerator                        # Phase 3  # noqa: E402
from recommender import AetherRecommender                     # Phase 4  # noqa: E402
from rag import AetherExplainer                               # Phase 5  # noqa: E402
from tools import AetherTools                                 # Phase 6  # noqa: E402
from agent import AetherAgent                                 # Phase 6  # noqa: E402
from harmonic import HarmonicIndex                            # Phase 7  # noqa: E402
from drift import EmotionDriftDetector                        # Phase 7  # noqa: E402
from transition import TransitionSelector                     # Phase 7  # noqa: E402
from crossfade import CrossfadePlanner                        # Phase 7  # noqa: E402
from engine import LiveTransitionEngine                       # Phase 7  # noqa: E402


def _load_llm() -> Optional[Callable[[str], str]]:
    """Groq if a key is present, else None (service still runs on fallbacks)."""
    if not os.getenv("GROQ_API_KEY"):
        return None
    try:
        from llm_backends import make_llm
        return make_llm("groq")
    except Exception:
        return None


class AetherService:
    """Everything the API needs, wired once over a shared feature store."""

    def __init__(
        self,
        store: FeatureStore,
        harmonic_index: Optional[HarmonicIndex] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        songs: Optional[list[Song]] = None,
    ) -> None:
        self.store = store
        self.harmonic = harmonic_index
        self.llm_fn = llm_fn
        self._songs = {s.track_id: s for s in (songs or [])}

        # Phase 3/4 — matching + recommendation
        self.recommender = AetherRecommender(PlaylistGenerator(store))
        # Phase 5 — RAG explainer (TF-IDF retrieval for fast startup)
        self.explainer = AetherExplainer.default(prefer_chroma=False, llm_fn=llm_fn)
        # Phase 6 — agent over the same tools
        self.agent = AetherAgent(
            AetherTools(self.recommender, self.explainer), llm_fn=llm_fn)

        # Phase 7 — shared (stateless) transition machinery; per-session state
        # lives in each session's own engine.
        self._selector = (TransitionSelector(store, harmonic_index)
                          if harmonic_index is not None else None)
        self._planner = CrossfadePlanner()
        self._sessions: dict[str, LiveTransitionEngine] = {}

    # ── constructors ──
    @classmethod
    def from_sample(cls, llm_fn: "Optional[Callable[[str], str]] | bool" = True
                    ) -> "AetherService":
        """Build over the in-memory sample catalog (dev default)."""
        from sample_data import build_sample_songs
        songs = build_sample_songs()
        store = FeatureStore().build_from_songs(songs)
        hidx = HarmonicIndex().build_from_songs(songs)
        fn = _load_llm() if llm_fn is True else (llm_fn or None)
        return cls(store, hidx, llm_fn=fn, songs=songs)

    @classmethod
    def from_store_path(cls, path: str,
                        llm_fn: "Optional[Callable[[str], str]] | bool" = True
                        ) -> "AetherService":
        """
        Build over the real saved 1.2M store. Curate + journey work fully; the
        live player also needs a HarmonicIndex, which requires a pass over the
        source songs for key/mode (the saved store drops those) — supply it via
        `attach_harmonic_index()` when wiring the full catalog.
        """
        store = FeatureStore.load(path)
        fn = _load_llm() if llm_fn is True else (llm_fn or None)
        return cls(store, harmonic_index=None, llm_fn=fn)

    def attach_harmonic_index(self, songs: list[Song]) -> None:
        """Enable the live player on a store loaded from disk."""
        self.harmonic = HarmonicIndex().build_from_songs(songs)
        self._selector = TransitionSelector(self.store, self.harmonic)
        self._songs.update({s.track_id: s for s in songs})

    # ── distribution helpers ──
    @staticmethod
    def distribution_for_emotion(emotion: str, peak: float = 0.9) -> np.ndarray:
        d = np.zeros(len(AETHER_EMOTIONS), dtype=np.float64)
        if emotion in AETHER_EMOTIONS:
            d[AETHER_EMOTIONS.index(emotion)] = peak
        return d

    def _resolve_distribution(
        self, emotion: Optional[str], distribution: Optional[list[float]],
        text: Optional[str],
    ) -> tuple[np.ndarray, str]:
        """Turn any of {distribution, emotion, text} into a (vector, label)."""
        if distribution is not None:
            v = np.asarray(distribution, dtype=np.float64)
            if v.shape[0] != len(AETHER_EMOTIONS):
                raise ValueError(
                    f"distribution must have {len(AETHER_EMOTIONS)} values.")
            label = AETHER_EMOTIONS[int(np.argmax(v))]
            return v, label
        if emotion:
            if emotion not in AETHER_EMOTIONS:
                raise ValueError(f"unknown emotion {emotion!r}.")
            return self.distribution_for_emotion(emotion), emotion
        if text:
            from perceive import perceive           # Phase 6 rule/LLM parser
            p = perceive(text, llm_fn=self.llm_fn)
            return self.distribution_for_emotion(p.start), p.start
        # default
        return self.distribution_for_emotion("calm"), "calm"

    # ── MAIN FEATURE: curate ──
    def curate(
        self, emotion: Optional[str] = None,
        distribution: Optional[list[float]] = None,
        text: Optional[str] = None, length: int = 12, explain: bool = True,
    ):
        """Return an explained playlist for a mood (Phase 3/4 + Phase 5 RAG)."""
        dist, label = self._resolve_distribution(emotion, distribution, text)
        rec = self.recommender.recommend(dist, raw_text=text or label,
                                         length=length)
        if explain:
            self.explainer.annotate(rec)      # folds 'why' into each track
        return rec

    # ── AGENT: journey ──
    def journey(self, text: str, length: int = 12):
        """Autonomous emotional-arc playlist for a request (Phase 6)."""
        return self.agent.run(text, length=length)

    # ── FUN FEATURE: live player ──
    def live_start(self, track_id: str) -> str:
        """Open a live session on a starting track; returns a session id."""
        if self._selector is None:
            raise RuntimeError("live player needs a harmonic index "
                               "(use from_sample or attach_harmonic_index).")
        if track_id not in self._songs and self.harmonic.get(track_id) is None:
            raise KeyError(f"unknown track_id {track_id!r}.")
        sid = uuid.uuid4().hex[:12]
        engine = LiveTransitionEngine(
            self.store, self.harmonic, self._selector, self._planner,
            EmotionDriftDetector())
        engine.start(track_id)
        self._sessions[sid] = engine
        return sid

    def live_observe(
        self, session_id: str, emotion: Optional[str] = None,
        distribution: Optional[list[float]] = None,
    ):
        """Feed an emotion reading; get a hold/transition decision."""
        engine = self._sessions.get(session_id)
        if engine is None:
            raise KeyError(f"unknown session_id {session_id!r}.")
        dist, _ = self._resolve_distribution(emotion, distribution, None)
        return engine.observe(dist)

    def end_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    # ── catalog ──
    def list_tracks(self) -> list[dict]:
        out = []
        for tid, s in self._songs.items():
            prof = self.harmonic.get(tid) if self.harmonic else None
            out.append({
                "track_id": tid, "name": s.name, "artist": s.primary_artist(),
                "camelot": prof.camelot if prof else None,
                "bpm": prof.bpm if prof else s.raw_features.get("tempo"),
            })
        return out


if __name__ == "__main__":
    svc = AetherService.from_sample(llm_fn=None)   # offline (no LLM) for the smoke test
    print("service built:", len(svc.list_tracks()), "tracks, "
          f"llm={'on' if svc.llm_fn else 'off'}")
    rec = svc.curate(emotion="sad", length=4)
    print("curate('sad',4):", [(t.title, t.source_emotion) for t in rec.tracks])
    res = svc.journey("from anxious to calm", length=6)
    print("journey:", res.plan.describe(), "| tracks", res.playlist.size)
    sid = svc.live_start("sad-1")
    d = svc.live_observe(sid, emotion="energetic")
    print("live:", d.triggered, "→", d.next.track_id if d.next else None)
