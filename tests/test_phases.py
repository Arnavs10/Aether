"""
Aether — unit tests for core phase logic (Phases 5–7).

Run from the repo root:  pytest -q
These assert the load-bearing invariants each phase promises, independent of the
inline self-tests, so a regression anywhere fails CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PHASE6 = "phase_6_agentic" if (_ROOT / "phase_6_agentic").exists() else "phase_6_agentic_ai"
for _name in ("", "phase_2_music_data", "phase_3_emotion_music_mapping",
              "phase_4_recommendation", "phase_5_rag", _PHASE6,
              "phase_7_drift_crossfade"):
    _p = _ROOT / _name if _name else _ROOT
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ─────────────── Phase 7: Camelot ───────────────
from camelot import to_camelot, compatibility_score, compatible_codes


def test_camelot_conversions():
    assert to_camelot(0, 1) == "8B"     # C major
    assert to_camelot(9, 0) == "8A"     # A minor (relative)
    assert to_camelot(7, 1) == "9B"     # G major
    assert to_camelot(-1, 1) is None    # unknown key


@pytest.mark.parametrize("a,b,expected", [
    ("8B", "8B", 1.00),   # same
    ("8B", "9B", 0.85),   # adjacent, same letter
    ("8B", "8A", 0.80),   # relative major/minor
    ("8B", "2B", 0.00),   # clash
    ("1B", "12B", 0.85),  # wheel wrap
])
def test_camelot_compatibility(a, b, expected):
    assert compatibility_score(a, b) == expected


def test_camelot_compatible_codes():
    assert compatible_codes("8B") == {"8B", "9B", "7B", "8A"}


# ─────────────── Phase 7: drift ───────────────
from drift import EmotionDriftDetector, js_divergence
from config import AETHER_EMOTIONS


def _peak(emo, s=0.85):
    v = np.full(len(AETHER_EMOTIONS), (1 - s) / (len(AETHER_EMOTIONS) - 1))
    v[AETHER_EMOTIONS.index(emo)] = s
    return v


def test_js_divergence_bounds():
    assert js_divergence(_peak("calm"), _peak("calm")) == pytest.approx(0.0, abs=1e-9)
    assert js_divergence(_peak("calm"), _peak("energetic")) > 0.5


def test_drift_fires_on_switch_not_on_stability():
    d = EmotionDriftDetector(window=4, threshold=0.22)
    for _ in range(3):
        assert not d.observe(_peak("sad")).drifted
    ev = d.observe(_peak("energetic"))
    assert ev.drifted and ev.from_emotion == "sad" and ev.to_emotion == "energetic"
    # re-anchored: sustained new mood does not re-fire
    assert not d.observe(_peak("energetic")).drifted


# ─────────────── Phase 7: transition + engine ───────────────
from schema import Song
from feature_store import FeatureStore
from harmonic import HarmonicIndex
from transition import TransitionSelector
from engine import LiveTransitionEngine, _build_songs


def _song(tid, key, mode, tempo, energy=0.22, valence=0.5):
    rf = {"danceability": 0.3, "energy": energy, "key": key, "loudness": 0.4,
          "mode": mode, "speechiness": 0.1, "acousticness": 0.8,
          "instrumentalness": 0.4, "liveness": 0.1, "valence": valence,
          "tempo": tempo}
    return Song(tid, f"s-{tid}", ["A"], 2000, "2000-01-01", rf)


def test_transition_prefers_harmonically_compatible():
    songs = [_song("cur", 0, 1, 80), _song("A", 9, 0, 80),
             _song("B", 7, 1, 80), _song("C", 6, 1, 80)]
    store = FeatureStore().build_from_songs(songs)
    hidx = HarmonicIndex().build_from_songs(songs)
    sel = TransitionSelector(store, hidx)
    best = sel.select_next("cur", "calm")
    assert best.track_id == "B" and best.camelot == "9B"   # adjacent = best mix


def test_engine_live_transition_and_playlist():
    engine = LiveTransitionEngine.from_songs(_build_songs())
    engine.start("sad1")
    assert not engine.observe(_peak("sad")).triggered      # warm-up/hold
    d = engine.observe(_peak("energetic"))
    assert d.triggered and d.next.track_id in {"en1", "en2", "en3"}
    assert 3.0 <= d.crossfade.duration_s <= 5.0
    plans = engine.plan_playlist(["sad1", "sad2", "ca1"])
    assert len(plans) == 2 and all(3.0 <= p.duration_s <= 5.0 for p in plans)


# ─────────────── Phase 6: perceive ───────────────
from perceive import perceive


@pytest.mark.parametrize("text,start,target", [
    ("help me go from anxious to calm", "anxious", "calm"),
    ("lift me up out of feeling sad", "sad", "hopeful"),
    ("some happy music please", "happy", "happy"),
])
def test_perceive_journeys(text, start, target):
    p = perceive(text)
    assert (p.start, p.target) == (start, target)
