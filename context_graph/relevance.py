"""Two relevance models over the session graph.

Shortest path (Dijkstra 1959). Edge weight w = -log p, so the shortest path
from the focus to an atom is its most probable chain of relevance, and
relevance = exp(-distance). One strong chain is enough; many weak chains add
nothing.

Heat diffusion (Chung 2007, "The heat kernel as the pagerank of a graph").
Heat starts on the focus atoms and flows along edges for time t:

    u(t) = u0 exp(-t L),   L = I - P,   P = D^-1 W

with W the edge affinities (p) and D their row sums. This is Chung's heat
kernel pagerank, computed as the truncated series

    u(t) = e^-t * sum_k (t^k / k!) u0 P^k

Unlike the shortest path, it adds up every path, so an atom reached by many
weak links can outrank one reached by a single strong link.
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


def heat_kernel(g: Graph, sources: dict[str, float], t: float, tol: float = 1e-9,
                normalise: bool = True) -> dict[str, float]:
    """Heat kernel pagerank. The raw vector sums to 1 (heat is conserved); by
    default it is returned max-normalised to [0, 1] so it is on the same scale
    as shortest-path relevance."""
    ids = list(g.atoms)
    total = sum(s for sid, s in sources.items() if sid in g.atoms)
    if total <= 0:
        return {aid: 0.0 for aid in ids}
    x = {sid: s / total for sid, s in sources.items() if sid in g.atoms and s > 0}
    deg = {aid: g.degree(aid) for aid in ids}
    coef = math.exp(-t)
    rho = {aid: 0.0 for aid in ids}
    k = 0
    while True:
        for aid, v in x.items():
            rho[aid] += coef * v
        k += 1
        coef *= t / k
        if coef < tol and k > t:
            break
        nxt: dict[str, float] = {}
        for aid, v in x.items():
            if v == 0.0:
                continue
            d = deg[aid]
            if d == 0.0:
                nxt[aid] = nxt.get(aid, 0.0) + v  # isolated node keeps its heat
                continue
            for nbr, e in g.adj[aid].items():
                nxt[nbr] = nxt.get(nbr, 0.0) + v * e.p / d
        x = nxt
    if not normalise:
        return rho
    peak = max(rho.values()) or 1.0
    return {aid: v / peak for aid, v in rho.items()}
