"""Relevance over the session graph.

Shortest path (Dijkstra 1959). Edge weight w = -log p, so the shortest path
from the focus to an atom is its most probable chain of relevance, and
relevance = exp(-distance). One strong chain is enough; many weak chains add
nothing.
"""

from __future__ import annotations

import heapq
import math

from .graph import Graph


def dijkstra(g: Graph, sources: dict[str, float]) -> tuple[dict[str, float], dict[str, str | None]]:
    """Multi-source Dijkstra. A source with weight s starts at distance -log s."""
    dist: dict[str, float] = {}
    prev: dict[str, str | None] = {}
    heap: list[tuple[float, int, str, str | None]] = []
    tie = 0  # keeps heap entries comparable without comparing node ids to None
    for sid, s in sources.items():
        if sid in g.atoms and s > 0:
            heapq.heappush(heap, (-math.log(s), tie, sid, None))
            tie += 1
    while heap:
        d, _, node, parent = heapq.heappop(heap)
        if node in dist:
            continue
        dist[node] = d
        prev[node] = parent
        for nbr, e in g.adj[node].items():
            if nbr not in dist:
                heapq.heappush(heap, (d + e.weight, tie, nbr, node))
                tie += 1
    return dist, prev


def shortest_path_relevance(g: Graph, sources: dict[str, float]) -> dict[str, float]:
    dist, _ = dijkstra(g, sources)
    return {aid: math.exp(-dist[aid]) if aid in dist else 0.0 for aid in g.atoms}


def path_to(prev: dict[str, str | None], target: str) -> list[str]:
    if target not in prev:
        return []
    out = [target]
    while prev[out[-1]] is not None:
        out.append(prev[out[-1]])  # type: ignore[arg-type]
    return list(reversed(out))
