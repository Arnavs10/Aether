"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 6 · Agentic Layer (LangGraph)
═══════════════════════════════════════════════════════════════════
An autonomous agent that turns a free-text request into an explained playlist
that TRAVELS an emotional arc. It runs the classic agent loop as a LangGraph
state machine:

    perceive → plan → act → reflect ─┐
                 ▲                    │ (arc too abrupt & budget left)
                 └────────────────────┘
                          │ (good enough)
                          ▼
                       explain → END

  • perceive — read the request → start & target emotion (Phase 6 perceiver).
  • plan     — lay out the waypoint arc in valence–arousal space (Phase 6 arc).
  • act      — recommend a segment per waypoint and stitch them (Phase 4 tools).
  • reflect  — check the assembled arc actually moves the right way; if not and
               there's iteration budget, REPLAN with a smoother (longer) arc.
  • explain  — ground a "why this journey" in retrieval (Phase 5 RAG).

Runs on a real LangGraph `StateGraph` when the library is installed; otherwise a
built-in runner executes the identical nodes/edges/loop, so the agent (and its
self-test) works with zero extra deps.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

# ── Path bootstrap: root + local ──
_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from perceive import perceive, Perceived                     # noqa: E402
from arc import plan_arc, ArcPlan, energy_of                 # noqa: E402
from tools import AetherTools                                 # noqa: E402


# ──────────────────────────────────────────────
# Graph state + result
# ──────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    request: str
    length: int
    max_waypoints: int
    iterations: int
    max_iterations: int
    perceived: Perceived
    plan: ArcPlan
    playlist: Any            # Recommendation
    explanation: Any         # Explanation
    reflection: dict
    trace: list[str]


@dataclass
class AgentResult:
    """The agent's finished output."""
    request: str
    perceived: Perceived
    plan: ArcPlan
    playlist: Any            # annotated Recommendation
    explanation: Any        # Explanation
    trace: list[str]

    def summary(self) -> str:
        return (f"'{self.request}' → {self.plan.describe()} "
                f"({self.playlist.size} tracks). {self.explanation.summary}")


def _allocate_lengths(k: int, total: int) -> list[int]:
    """Split `total` tracks across `k` waypoints as evenly as possible (≥1 each)."""
    if k <= 0:
        return []
    base, rem = divmod(total, k)
    lengths = [base + (1 if i < rem else 0) for i in range(k)]
    return [max(1, x) for x in lengths]


# ──────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────
class AetherAgent:
    """Autonomous playlist-arc planner over Phases 4–5, wired as a LangGraph."""

    def __init__(
        self,
        tools: AetherTools,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_iterations: int = 2,
        use_langgraph: bool = True,
    ):
        self.tools = tools
        self.llm_fn = llm_fn
        self.max_iterations = max_iterations
        self.use_langgraph = use_langgraph and _langgraph_available()
        self._app = self._build_graph() if self.use_langgraph else None

    # ── convenience constructor (real store + RAG + optional LLM) ──
    @classmethod
    def default(
        cls,
        store_path: Optional[str] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        prefer_chroma: bool = True,
        max_iterations: int = 2,
    ) -> "AetherAgent":
        tools = AetherTools.default(store_path=store_path, llm_fn=llm_fn,
                                    prefer_chroma=prefer_chroma)
        return cls(tools, llm_fn=llm_fn, max_iterations=max_iterations)

    # ────────────── nodes ──────────────
    def _perceive(self, state: AgentState) -> dict:
        p = perceive(state["request"], llm_fn=self.llm_fn,
                     default_length=state.get("length", 12))
        return {"perceived": p, "length": p.length,
                "trace": state.get("trace", []) + ["perceive"]}

    def _plan(self, state: AgentState) -> dict:
        p: Perceived = state["perceived"]
        mw = state.get("max_waypoints", 3)
        plan = plan_arc(p.start, p.target, max_waypoints=mw)
        return {"plan": plan,
                "trace": state.get("trace", []) + [f"plan(mw={mw})"]}

    def _act(self, state: AgentState) -> dict:
        plan: ArcPlan = state["plan"]
        total = state["length"]
        lengths = _allocate_lengths(len(plan.waypoints), total)
        segments = [
            self.tools.recommend_segment(w, length=n, arc=plan.direction)
            for w, n in zip(plan.waypoints, lengths)
        ]
        playlist = AetherTools.assemble_journey(
            segments, plan, state["request"], total)
        return {"playlist": playlist,
                "trace": state.get("trace", []) + ["act"]}

    def _reflect(self, state: AgentState) -> dict:
        plan: ArcPlan = state["plan"]
        pl = state["playlist"]
        iterations = state.get("iterations", 0) + 1
        notes: list[str] = []
        ok = True

        # Length: allow small shortfall if the store simply lacks tracks.
        if pl.size < max(1, int(0.6 * state["length"])):
            ok = False
            notes.append(f"only {pl.size}/{state['length']} tracks")

        # Arc integrity: for a journey, energy must actually move the right way.
        if plan.is_journey and pl.size >= 2:
            e_first = pl.tracks[0].energy
            e_last = pl.tracks[-1].energy
            if e_first is not None and e_last is not None:
                delta = e_last - e_first
                want = plan.direction
                if want == "ascending" and delta <= 0.05:
                    ok = False; notes.append("energy did not rise enough")
                elif want == "descending" and delta >= -0.05:
                    ok = False; notes.append("energy did not fall enough")

        if ok:
            notes.append("arc integrity ok")
        return {"reflection": {"ok": ok, "notes": notes},
                "iterations": iterations,
                # Replanning smooths the arc by allowing one more waypoint.
                "max_waypoints": state.get("max_waypoints", 3) + 1,
                "trace": state.get("trace", []) + [f"reflect({'ok' if ok else 'revise'})"]}

    def _explain(self, state: AgentState) -> dict:
        pl = state["playlist"]
        expl = self.tools.explain(pl)
        self.tools.annotate(pl)      # fold 'why' into the Recommendation in place
        return {"explanation": expl,
                "trace": state.get("trace", []) + ["explain"]}

    # ────────────── routing ──────────────
    def _route(self, state: AgentState) -> str:
        """After reflect: replan if fixable and budget remains, else explain."""
        refl = state.get("reflection", {"ok": True})
        if refl.get("ok"):
            return "explain"
        if state.get("iterations", 0) >= state.get("max_iterations", self.max_iterations):
            return "explain"      # out of budget — ship the best we have
        return "plan"

    # ────────────── graph wiring (real LangGraph) ──────────────
    def _build_graph(self):
        from langgraph.graph import StateGraph, END
        g = StateGraph(AgentState)
        g.add_node("perceive", self._perceive)
        g.add_node("plan", self._plan)
        g.add_node("act", self._act)
        g.add_node("reflect", self._reflect)
        g.add_node("explain", self._explain)
        g.set_entry_point("perceive")
        g.add_edge("perceive", "plan")
        g.add_edge("plan", "act")
        g.add_edge("act", "reflect")
        g.add_conditional_edges("reflect", self._route,
                                {"plan": "plan", "explain": "explain"})
        g.add_edge("explain", END)
        return g.compile()

    # ────────────── run ──────────────
    def run(self, request: str, length: Optional[int] = None) -> AgentResult:
        """Perceive → plan → act → reflect(⟲) → explain, returning the result."""
        init: AgentState = {
            "request": request,
            "length": length or 12,
            "max_waypoints": 3,
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "trace": [],
        }
        final = self._app.invoke(init) if self._app is not None \
            else self._run_builtin(init)
        return AgentResult(
            request=request, perceived=final["perceived"], plan=final["plan"],
            playlist=final["playlist"], explanation=final["explanation"],
            trace=final["trace"],
        )

    def _run_builtin(self, state: AgentState) -> AgentState:
        """Dependency-free equivalent of the LangGraph loop (same nodes/edges)."""
        state = {**state, **self._perceive(state)}
        while True:
            state = {**state, **self._plan(state)}
            state = {**state, **self._act(state)}
            state = {**state, **self._reflect(state)}
            if self._route(state) == "explain":
                break
        state = {**state, **self._explain(state)}
        return state


def _langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
        from langgraph.graph import StateGraph  # noqa: F401
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Self-test — full agent run on a tiny in-memory store (no CSV/network)
# ─────────────────────────────────────────────────────────────
def _build_tiny_tools() -> AetherTools:
    from schema import Song
    from feature_store import FeatureStore
    from playlist import PlaylistGenerator
    from recommender import AetherRecommender
    from rag import AetherExplainer

    def mk(tid, name, artist, feats, year=2000):
        base = {f: 0.5 for f in
                ["danceability", "energy", "key", "loudness", "mode",
                 "speechiness", "acousticness", "instrumentalness",
                 "liveness", "valence", "tempo"]}
        base.update(feats)
        return Song(tid, name, [artist], year, f"{year}-01-01", base)

    songs = [
        mk("a1", "Edge", "Ana", {"tempo": 105, "energy": 0.62, "valence": 0.22, "danceability": 0.38, "acousticness": 0.30, "instrumentalness": 0.25}),
        mk("a2", "Nerve", "Bo", {"tempo": 108, "energy": 0.60, "valence": 0.24, "danceability": 0.40, "acousticness": 0.28, "instrumentalness": 0.22}),
        mk("f1", "Flow", "Cy", {"tempo": 92, "energy": 0.35, "valence": 0.42, "danceability": 0.18, "acousticness": 0.45, "instrumentalness": 0.70}),
        mk("f2", "Deep Work", "Di", {"tempo": 90, "energy": 0.34, "valence": 0.40, "danceability": 0.20, "acousticness": 0.48, "instrumentalness": 0.72}),
        mk("n1", "Old Film", "Ed", {"tempo": 88, "energy": 0.38, "valence": 0.48, "danceability": 0.32, "acousticness": 0.72, "instrumentalness": 0.15}),
        mk("c1", "Still", "Fe", {"tempo": 80, "energy": 0.22, "valence": 0.50, "danceability": 0.28, "acousticness": 0.80, "instrumentalness": 0.40}),
        mk("c2", "Breath", "Gi", {"tempo": 78, "energy": 0.20, "valence": 0.52, "danceability": 0.26, "acousticness": 0.82, "instrumentalness": 0.42}),
        mk("h1", "Sun Up", "Ha", {"tempo": 112, "energy": 0.65, "valence": 0.82, "danceability": 0.55, "acousticness": 0.35, "instrumentalness": 0.12}),
    ]
    store = FeatureStore().build_from_songs(songs)
    return AetherTools(
        AetherRecommender(PlaylistGenerator(store, max_per_artist=2)),
        AetherExplainer.default(prefer_chroma=False),
    )


def _selftest() -> None:
    print("Agent self-test")
    print("-" * 55)
    print(f"  langgraph available: {_langgraph_available()}")

    tools = _build_tiny_tools()

    # Force the built-in runner first so this passes even without langgraph.
    agent = AetherAgent(tools, max_iterations=2, use_langgraph=False)

    # 1. Journey request → planned arc, assembled + explained.
    res = agent.run("help me go from anxious to calm", length=6)
    assert res.perceived.start == "anxious" and res.perceived.target == "calm"
    assert res.plan.is_journey and res.plan.waypoints[0] == "anxious"
    assert res.playlist.size >= 4, res.playlist.size
    assert res.explanation.summary
    # every track carries a grounded 'why' after explain/annotate
    assert all(t.extra.get("why") for t in res.playlist.tracks)
    print(f"  run(anxious→calm) → {res.plan.describe()} | trace={res.trace}")
    print(f"    energies: {[round(t.energy,2) for t in res.playlist.tracks]}")

    # 2. Energy trends downward across the whole journey (arc integrity).
    assert res.playlist.tracks[-1].energy < res.playlist.tracks[0].energy

    # 3. Single-mood request → no journey, still explained.
    res2 = agent.run("some focused study music", length=3)
    assert not res2.plan.is_journey and res2.perceived.start == "focused"
    assert res2.playlist.size >= 1 and res2.explanation.summary
    print(f"  run(focused) → single mood, {res2.playlist.size} tracks, "
          f"trace={res2.trace}")

    # 4. Reflect triggers a replan when the arc is too flat (unit-level).
    flat_state: AgentState = {
        "request": "x", "length": 6, "max_waypoints": 3,
        "iterations": 0, "max_iterations": 2, "trace": [],
        "plan": plan_arc("sad", "energetic"),
    }
    # fake a playlist whose energy does NOT rise (violates ascending intent)
    seg = tools.recommend_segment("sad", length=4)
    flat_state["playlist"] = seg
    seg.tracks[0].energy, seg.tracks[-1].energy = 0.5, 0.5
    upd = agent._reflect(flat_state)
    assert upd["reflection"]["ok"] is False, upd["reflection"]
    flat_state.update(upd)
    assert agent._route(flat_state) == "plan"      # → replan
    print(f"  reflect(flat ascending) → revise & replan ✓ "
          f"({upd['reflection']['notes']})")

    # 5. Same run via the REAL LangGraph, if installed — identical contract.
    if _langgraph_available():
        lg_agent = AetherAgent(tools, max_iterations=2, use_langgraph=True)
        assert lg_agent._app is not None
        rlg = lg_agent.run("take me from lonely to hopeful", length=6)
        assert rlg.plan.waypoints[0] == "lonely" and rlg.plan.waypoints[-1] == "hopeful"
        assert rlg.playlist.size >= 3 and rlg.explanation.summary
        assert "explain" in rlg.trace and "perceive" in rlg.trace
        print(f"  LangGraph run(lonely→hopeful) → {rlg.plan.describe()} "
              f"| {rlg.playlist.size} tracks ✓")
    else:
        print("  LangGraph not installed → built-in runner path validated only")

    print("-" * 55)
    print("✅ All agent self-tests passed.")


if __name__ == "__main__":
    _selftest()
