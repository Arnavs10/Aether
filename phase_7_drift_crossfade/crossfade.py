"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 7 · Crossfade Planner
═══════════════════════════════════════════════════════════════════
The *decision* half of the crossfade — not the audio itself. Given the
outgoing and incoming tracks, it outputs a CrossfadePlan (how long to fade, on
what gain curve, over how many beats) as clean data. Phase 8's Web Audio player
just executes that plan; nothing here touches sound.

Design choices (interview-defensible):
  • curve = equal-power (constant-power) by default. A linear crossfade dips in
    perceived loudness at the midpoint because two uncorrelated signals sum in
    power, not amplitude; the equal-power (√) curve keeps loudness steady — this
    is what Apple Music / DJ software use.
  • duration scales 3–5s with how *different* the two tracks are (energy + tempo
    gap): near-identical tracks get a short 3s blend, a bigger jump gets the
    full 5s so the change is masked smoothly.
  • beats = duration × BPM / 60, a hint so the renderer can beat-align the fade.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


@dataclass
class CrossfadePlan:
    """A renderable crossfade specification (data only — no audio)."""
    out_track_id: str
    in_track_id: str
    duration_s: float
    curve: str                       # "equal_power" | "linear"
    out_bpm: float
    in_bpm: float
    beats: float                     # fade length in beats of the incoming track
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "out_track_id": self.out_track_id,
            "in_track_id": self.in_track_id,
            "duration_s": round(self.duration_s, 2),
            "curve": self.curve,
            "out_bpm": round(self.out_bpm, 1),
            "in_bpm": round(self.in_bpm, 1),
            "beats": round(self.beats, 1),
            "notes": self.notes,
            **({"extra": self.extra} if self.extra else {}),
        }


class CrossfadePlanner:
    """Turns a pair of tracks into a 3–5s equal-power crossfade plan."""

    def __init__(self, min_s: float = 3.0, max_s: float = 5.0,
                 curve: str = "equal_power", tempo_span: float = 40.0) -> None:
        """
        Args:
            min_s / max_s: crossfade duration bounds (seconds).
            curve: default gain curve ("equal_power" recommended).
            tempo_span: BPM gap treated as "maximally different" for scaling.
        """
        self.min_s = min_s
        self.max_s = max_s
        self.curve = curve
        self.tempo_span = max(1.0, tempo_span)

    def plan(
        self,
        out_track_id: str,
        in_track_id: str,
        out_bpm: float,
        in_bpm: float,
        out_energy: float,
        in_energy: float,
        curve: Optional[str] = None,
    ) -> CrossfadePlan:
        """Build a crossfade plan from the two tracks' tempo + energy."""
        energy_gap = min(1.0, abs(float(out_energy) - float(in_energy)))
        tempo_gap = min(1.0, abs(float(out_bpm) - float(in_bpm)) / self.tempo_span)
        difference = 0.5 * energy_gap + 0.5 * tempo_gap        # 0 (same) … 1 (far)

        duration = self.min_s + difference * (self.max_s - self.min_s)
        duration = max(self.min_s, min(self.max_s, duration))

        # Beat-span uses the incoming track's tempo (what the listener lands on).
        bpm_for_beats = in_bpm if in_bpm > 0 else out_bpm
        beats = duration * bpm_for_beats / 60.0 if bpm_for_beats > 0 else 0.0

        notes = (
            f"{'gentle' if difference < 0.34 else 'longer' if difference > 0.66 else 'moderate'} "
            f"{self.curve} crossfade (Δenergy {energy_gap:.2f}, "
            f"Δtempo {abs(out_bpm - in_bpm):.0f} BPM)"
        )
        return CrossfadePlan(
            out_track_id=out_track_id, in_track_id=in_track_id,
            duration_s=duration, curve=curve or self.curve,
            out_bpm=out_bpm, in_bpm=in_bpm, beats=beats, notes=notes,
        )

    def plan_profiles(self, out_profile: Any, in_profile: Any,
                      curve: Optional[str] = None) -> CrossfadePlan:
        """Convenience for HarmonicProfile-like objects (bpm + energy attrs)."""
        return self.plan(
            out_profile.track_id, in_profile.track_id,
            out_profile.bpm, in_profile.bpm,
            out_profile.energy, in_profile.energy, curve=curve,
        )


# ─────────────────────────────────────────────────────────────
# Self-test — pure, offline
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Crossfade planner self-test")
    print("-" * 55)

    p = CrossfadePlanner(min_s=3.0, max_s=5.0)

    # 1. Near-identical tracks → short (~min) fade.
    same = p.plan("a", "b", out_bpm=120, in_bpm=120, out_energy=0.5, in_energy=0.5)
    assert abs(same.duration_s - 3.0) < 1e-6, same.duration_s
    assert same.curve == "equal_power"
    print(f"  identical tracks → {same.duration_s:.1f}s ({same.notes})")

    # 2. Very different tracks → long fade (near the top of the range).
    far = p.plan("a", "c", out_bpm=90, in_bpm=140, out_energy=0.2, in_energy=0.9)
    assert far.duration_s > 4.5, far.duration_s
    print(f"  far tracks (Δtempo 50, Δenergy .7) → {far.duration_s:.1f}s")

    # 2b. Maximal difference → clamped to max_s.
    mx = p.plan("a", "e", out_bpm=80, in_bpm=160, out_energy=0.0, in_energy=1.0)
    assert abs(mx.duration_s - 5.0) < 1e-6, mx.duration_s
    print(f"  maximal difference → {mx.duration_s:.1f}s (clamped to max) ✓")

    # 3. Moderate difference → between bounds; duration clamped in range.
    mid = p.plan("a", "d", out_bpm=120, in_bpm=132, out_energy=0.5, in_energy=0.6)
    assert 3.0 <= mid.duration_s <= 5.0
    print(f"  moderate → {mid.duration_s:.2f}s, beats {mid.beats:.1f} @ {mid.in_bpm:.0f}bpm")

    # 4. beats = duration × bpm / 60.
    assert abs(mid.beats - (mid.duration_s * 132 / 60.0)) < 1e-6
    print("  beat-span math ✓")

    # 5. as_dict is renderer-ready.
    d = far.as_dict()
    assert {"out_track_id", "in_track_id", "duration_s", "curve", "beats"} <= set(d)
    print(f"  as_dict → {d}")

    print("-" * 55)
    print("✅ All crossfade-planner self-tests passed.")


if __name__ == "__main__":
    _selftest()
