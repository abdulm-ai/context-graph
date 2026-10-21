import math
import os
import unittest

from context_graph import config
from context_graph.atoms import Atom
from context_graph.engine import build_pack
from context_graph.atoms import extract_atoms
from context_graph.graph import Graph
from context_graph.relevance import heat_kernel, shortest_path_relevance
from context_graph.transcript import parse_transcript

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "session_basic.jsonl")


def graph_with(edges, nodes):
    g = Graph([Atom(n, "reply", 1, i, n, n) for i, n in enumerate(nodes)])
    for a, b, p in edges:
        g.link(a, b, p, "test")
    return g


class DiffusionTest(unittest.TestCase):
    def setUp(self):
        # X: one strong chain s-a-X (0.9 * 0.9). Y: six weak chains s-b_i-Y (0.3 * 0.3 each).
        bs = [f"b{i}" for i in range(6)]
        edges = [("s", "a", 0.9), ("a", "X", 0.9)]
        edges += [("s", b, 0.3) for b in bs] + [(b, "Y", 0.3) for b in bs]
        self.g = graph_with(edges, ["s", "a", "X", "Y"] + bs)

    def test_shortest_path_prefers_the_strong_chain(self):
        rel = shortest_path_relevance(self.g, {"s": 1.0})
        self.assertAlmostEqual(rel["X"], 0.81)
        self.assertAlmostEqual(rel["Y"], 0.09)

    def test_diffusion_adds_up_weak_paths(self):
        rel = heat_kernel(self.g, {"s": 1.0}, t=2.0)
        self.assertGreater(rel["Y"], rel["X"])

    def test_heat_is_conserved(self):
        raw = heat_kernel(self.g, {"s": 1.0}, t=3.0, normalise=False)
        self.assertAlmostEqual(sum(raw.values()), 1.0, places=6)

    def test_small_t_keeps_heat_on_the_source(self):
        raw = heat_kernel(self.g, {"s": 1.0}, t=1e-4, normalise=False)
        self.assertGreater(raw["s"], 0.999)

    def test_matches_matrix_exponential_on_a_path(self):
        # two nodes, one edge: P = [[0,1],[1,0]], u(t) = e^-t [cosh t, sinh t]
        g = graph_with([("p", "q", 0.5)], ["p", "q"])
        raw = heat_kernel(g, {"p": 1.0}, t=1.3, normalise=False)
        self.assertAlmostEqual(raw["p"], math.exp(-1.3) * math.cosh(1.3), places=8)
        self.assertAlmostEqual(raw["q"], math.exp(-1.3) * math.sinh(1.3), places=8)

    def test_config_switch(self):
        cfg = config.load("/nonexistent", env={"CONTEXT_GRAPH_RELEVANCE": "diffusion"})
        self.assertEqual(cfg["relevance"], "diffusion")
        atoms = extract_atoms(parse_transcript(FIX).events, cwd="/work/ledger")
        _, sel = build_pack(atoms, cfg, "replace", budget=600)
        self.assertEqual(sel.relevance, "diffusion")
        self.assertTrue(all(0.0 <= s["relevance"] <= 1.0 for s in sel.scores.values()))
        self.assertEqual(max(s["relevance"] for s in sel.scores.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
