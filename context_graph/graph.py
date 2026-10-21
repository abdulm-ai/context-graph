"""The session graph: atoms as nodes, edges weighted by conditional relevance.

Each edge carries p = P(j is relevant | i is relevant), set by why the two atoms
are linked. When several reasons link the same pair, they are combined as
independent evidence with a noisy-OR: p = 1 - prod(1 - p_k).

Shortest-path search uses w = -log p, so a path's length is its surprisal and
exp(-length) is the probability of the whole chain of relevance.
Diffusion uses p directly as the edge affinity.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .atoms import Atom

# Link strengths. These are priors, not fitted values; the benchmark measures
# the end result, and `ctx explain` prints the reasons behind every path.
P_FILE = {"edit": 0.8, "error": 0.8, "command": 0.6, "lookup": 0.5, "turn": 0.6,
          "constraint": 0.7, "decision": 0.7, "reply": 0.5, "tool": 0.4}
P_SYMBOL_MAX = 0.6
SYMBOL_MAX_DF = 12  # a symbol used by more atoms than this says little
P_FIX = 0.9  # error -> the next edit of the same file (within FIX_WINDOW turns)
FIX_WINDOW = 3
P_ERROR_OUTPUT = 0.95  # error -> the command that produced it
P_STATED_IN = 0.9  # constraint/decision -> the turn or reply it was stated in
P_IN_TURN = {"reply": 0.8, "command": 0.6, "edit": 0.6, "lookup": 0.5, "tool": 0.5, "error": 0.6}
P_NEXT_TURN = 0.5
P_NEXT_TOOL = 0.4

TEMPORAL_REASONS = ("in turn", "next turn", "next step")


@dataclass
class Edge:
    p: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def weight(self) -> float:
        return -math.log(max(self.p, 1e-12))


class Graph:
    def __init__(self, atoms: list[Atom]):
        self.atoms: dict[str, Atom] = {a.id: a for a in atoms}
        self.adj: dict[str, dict[str, Edge]] = defaultdict(dict)
        for a in atoms:
            self.adj[a.id]  # every atom is a node, even if isolated
        self._build(atoms)
        self._rehearse()

    # ------------------------------------------------------------ edges
    def link(self, a: str, b: str, p: float, reason: str) -> None:
        if a == b or a not in self.atoms or b not in self.atoms:
            return
        e = self.adj[a].get(b)
        if e is None:
            e = Edge()
            self.adj[a][b] = e
            self.adj[b][a] = e
        e.p = 1.0 - (1.0 - e.p) * (1.0 - p)
        if reason not in e.reasons:
            e.reasons.append(reason)

    def _build(self, atoms: list[Atom]) -> None:
        order = sorted(atoms, key=lambda a: (a.seq, a.id))

        # shared file path
        for a in order:
            if a.kind == "file":
                continue
            for path in a.files:
                self.link(a.id, f"f:{path}", P_FILE.get(a.kind, 0.4), f"file {path}")

        # shared symbol, weaker the more atoms share it
        by_sym: dict[str, list[str]] = defaultdict(list)
        for a in order:
            if a.kind == "file":
                continue
            for s in set(a.symbols):
                by_sym[s].append(a.id)
        for sym, ids in by_sym.items():
            df = len(ids)
            if df < 2 or df > SYMBOL_MAX_DF:
                continue
            p = P_SYMBOL_MAX / (1.0 + math.log(df))
            for i in range(df):
                for j in range(i + 1, df):
                    self.link(ids[i], ids[j], p, f"symbol {sym}")

        # error -> fix on the same file, error -> its command
        edits = [a for a in order if a.kind == "edit"]
        for a in order:
            if a.kind != "error":
                continue
            if a.parent:
                self.link(a.id, a.parent, P_ERROR_OUTPUT, "output of")
            files = set(a.files)
            if not files and a.parent in self.atoms:
                files = set(self.atoms[a.parent].files)
            for e in edits:
                if e.seq > a.seq and e.turn <= a.turn + FIX_WINDOW and files & set(e.files):
                    self.link(a.id, e.id, P_FIX, "error then fix")
                    break

        # stated in
        for a in order:
            if a.kind in ("constraint", "decision") and a.parent:
                self.link(a.id, a.parent, P_STATED_IN, "stated in")

        # temporal adjacency
        turns = {a.turn: a.id for a in order if a.kind == "turn"}
        for a in order:
            if a.kind in P_IN_TURN and a.turn in turns:
                self.link(a.id, turns[a.turn], P_IN_TURN[a.kind], "in turn")
        tnums = sorted(turns)
        for x, y in zip(tnums, tnums[1:]):
            self.link(turns[x], turns[y], P_NEXT_TURN, "next turn")
        prev = None
        for a in order:
            if a.kind not in ("command", "edit", "lookup", "tool"):
                continue
            if prev is not None and prev.turn == a.turn:
                self.link(prev.id, a.id, P_NEXT_TOOL, "next step")
            prev = a

    def _rehearse(self) -> None:
        """An atom counts as used again when a later atom links to it by content.

        Temporal links do not count: being next to something is not a reuse.
        """
        for aid, nbrs in self.adj.items():
            a = self.atoms[aid]
            last = a.meta.get("last_touch", a.turn)
            for bid, e in nbrs.items():
                if any(not r.startswith(TEMPORAL_REASONS) for r in e.reasons):
                    last = max(last, self.atoms[bid].turn)
            a.last_used = last

    # ------------------------------------------------------------ helpers
    @property
    def n_edges(self) -> int:
        return sum(len(n) for n in self.adj.values()) // 2

    def degree(self, aid: str) -> float:
        return sum(e.p for e in self.adj[aid].values())

    def latest_turn(self) -> int:
        return max((a.turn for a in self.atoms.values()), default=0)
