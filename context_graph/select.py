"""Choose what to carry over: relevance x decay, pins first, then the budget solver."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import budget as bud
from .atoms import Atom
from .decay import decay
from .graph import Graph
from .relevance import heat_kernel, shortest_path_relevance

MIN_VALUE = 1e-6


@dataclass
class Selection:
    selector: str
    relevance: str
    solver: str
    budget: int
    now_turn: int
    focus: dict[str, float]
    kept: list[str]
    dropped: list[str]
    pinned: list[str]
    scores: dict[str, dict[str, Any]]
    pinned_tokens: int = 0
    chosen_tokens: int = 0
    overflow: bool = False
    n_atoms: int = 0
    n_edges: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def focus_sources(atoms: list[Atom], n_turns: int) -> dict[str, float]:
    """Heat / distance starts on the latest user turns, halving per turn back."""
    turns = sorted({a.turn for a in atoms if a.kind == "turn"}, reverse=True)[: max(1, n_turns)]
    return {f"u{t}": 0.5**i for i, t in enumerate(turns)}


def relevance_scores(g: Graph, sources: dict[str, float], cfg: dict[str, Any]) -> dict[str, float]:
    if cfg.get("relevance") == "diffusion":
        return heat_kernel(g, sources, float(cfg.get("diffusion_time", 2.0)))
    return shortest_path_relevance(g, sources)


def select(atoms: list[Atom], cfg: dict[str, Any], budget: int, reserved_tokens: int = 0,
           user_share: float = 0.0) -> Selection:
    """Pins first, then the user's own prompts (newest first, up to user_share of
    the budget), then the solver fills what is left by relevance x decay."""
    if cfg.get("selector") == "recency":
        return select_recency(atoms, cfg, budget, reserved_tokens)
    t0 = time.perf_counter()
    g = Graph(atoms)
    t1 = time.perf_counter()
    now = g.latest_turn()
    sources = focus_sources(atoms, int(cfg.get("focus_turns", 2)))
    rel = relevance_scores(g, sources, cfg)
    t2 = time.perf_counter()
    half_life = float(cfg.get("half_life_turns", 8.0))

    latest = f"u{now}" if f"u{now}" in g.atoms else None
    must = [a for a in atoms if a.pinned or a.id == latest]
    must_ids = {a.id for a in must}
    pinned_tokens = sum(a.tokens for a in must)
    capacity = max(0, budget - reserved_tokens - pinned_tokens)

    # The user's own words are the densest record of intent, and the native
    # compaction prompt also asks for every user message. Keep them verbatim,
    # newest first, within their share of the budget.
    user_kept: list[Atom] = []
    user_cap = min(capacity, int(budget * max(0.0, user_share)))
    used = 0
    for a in sorted((a for a in atoms if a.kind == "turn" and a.id not in must_ids), key=lambda a: -a.turn):
        if used + a.tokens > user_cap:
            break
        user_kept.append(a)
        used += a.tokens
    user_ids = {a.id for a in user_kept}
    capacity -= used

    scores: dict[str, dict[str, Any]] = {}
    items: list[bud.Item] = []
    for a in atoms:
        d = decay(now, a.last_used, half_life)
        v = rel.get(a.id, 0.0) * d
        scores[a.id] = {
            "kind": a.kind,
            "turn": a.turn,
            "tokens": a.tokens,
            "relevance": round(rel.get(a.id, 0.0), 6),
            "decay": round(d, 6),
            "value": round(v, 6),
            "pinned": a.id in must_ids,
            "reason": "pinned" if a.id in must_ids else "user prompt" if a.id in user_ids else "",
        }
        if a.id not in must_ids and a.id not in user_ids and v > MIN_VALUE:
            items.append(bud.Item(a.id, v, a.tokens))

    bucket = int(cfg.get("bucket_tokens", 10))
    solver = cfg.get("solver", "knapsack")
    chosen = (bud.greedy if solver == "greedy" else bud.knapsack)(items, capacity, bucket)
    t3 = time.perf_counter()

    for aid in chosen:
        scores[aid]["reason"] = solver
    chosen_set = set(chosen) | user_ids
    kept = [a.id for a in atoms if a.id in must_ids or a.id in chosen_set]
    dropped = [a.id for a in atoms if a.id not in must_ids and a.id not in chosen_set]
    return Selection(
        selector="graph",
        relevance=cfg.get("relevance", "dijkstra"),
        solver=solver,
        budget=budget,
        now_turn=now,
        focus=sources,
        kept=kept,
        dropped=dropped,
        pinned=[a.id for a in must],
        scores=scores,
        pinned_tokens=pinned_tokens,
        chosen_tokens=sum(a.tokens for a in atoms if a.id in chosen_set),
        overflow=pinned_tokens + reserved_tokens > budget,
        n_atoms=len(atoms),
        n_edges=g.n_edges,
        timings_ms={
            "graph": round((t1 - t0) * 1000, 2),
            "relevance": round((t2 - t1) * 1000, 2),
            "solver": round((t3 - t2) * 1000, 2),
        },
    )


RECENCY_KINDS = ("turn", "reply", "command", "edit", "lookup", "tool", "error")


def select_recency(atoms: list[Atom], cfg: dict[str, Any], budget: int, reserved_tokens: int = 0) -> Selection:
    """Baseline: keep the most recent stretch of the session that fits, nothing else.

    No pins and no graph. This is plain truncation of the conversation to the
    same budget the graph method gets.
    """
    t0 = time.perf_counter()
    now = max((a.turn for a in atoms), default=0)
    capacity = max(0, budget - reserved_tokens)
    stream = sorted((a for a in atoms if a.kind in RECENCY_KINDS), key=lambda a: (a.seq, a.id))
    kept_set: set[str] = set()
    used = 0
    for a in reversed(stream):
        if used + a.tokens > capacity:
            break
        kept_set.add(a.id)
        used += a.tokens
    scores = {a.id: {"kind": a.kind, "turn": a.turn, "tokens": a.tokens, "pinned": False} for a in atoms}
    return Selection(
        selector="recency",
        relevance="none",
        solver="truncate",
        budget=budget,
        now_turn=now,
        focus={},
        kept=[a.id for a in atoms if a.id in kept_set],
        dropped=[a.id for a in atoms if a.id not in kept_set],
        pinned=[],
        scores=scores,
        chosen_tokens=used,
        n_atoms=len(atoms),
        timings_ms={"solver": round((time.perf_counter() - t0) * 1000, 2)},
    )
