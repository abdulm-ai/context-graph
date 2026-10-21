import contextlib
import io
import itertools
import math
import os
import random
import tempfile
import unittest

from context_graph import cli, config
from context_graph.atoms import Atom, extract_atoms
from context_graph.budget import Item, greedy, knapsack, total_value
from context_graph.decay import decay
from context_graph.engine import build_pack, select_and_log
from context_graph.events import PROMPT
from context_graph.graph import Graph
from context_graph.relevance import dijkstra, path_to, shortest_path_relevance
from context_graph.store import Store
from context_graph.tokens import estimate
from context_graph.transcript import parse_transcript

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "session_basic.jsonl")


def atom(aid, kind="reply", turn=1, seq=1, files=(), symbols=(), pinned=False, text="x"):
    a = Atom(aid, kind, turn, seq, text, text, files=list(files), symbols=list(symbols), pinned=pinned)
    a.tokens = estimate(text) + 4
    a.last_used = turn
    return a


class KnapsackTest(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(7)
        for _ in range(60):
            items = [Item(f"i{k}", rng.random(), rng.randint(5, 120)) for k in range(rng.randint(1, 9))]
            cap = rng.randint(20, 400)
            best = 0.0
            for r in range(len(items) + 1):
                for combo in itertools.combinations(items, r):
                    if sum(math.ceil(it.tokens / 10) for it in combo) <= cap // 10:
                        best = max(best, sum(it.value for it in combo))
            got = total_value(items, knapsack(items, cap))
            self.assertAlmostEqual(got, best, places=9)

    def test_greedy_can_lose(self):
        items = [Item("a", 10, 60), Item("b", 6, 50), Item("c", 6, 50)]
        self.assertEqual(sorted(knapsack(items, 100)), ["b", "c"])
        self.assertEqual(greedy(items, 100), ["a"])

    def test_capacity_respected(self):
        items = [Item(str(i), 1.0, 33) for i in range(50)]
        chosen = knapsack(items, 1000)
        self.assertLessEqual(sum(math.ceil(33 / 10) for _ in chosen), 100)


class RelevanceTest(unittest.TestCase):
    def setUp(self):
        # u2 -> a -> c is two strong hops; u2 -> c directly is one weak hop
        self.atoms = [
            atom("u2", "turn", 2, 1),
            atom("a", "edit", 2, 2, files=["x.py"]),
            atom("f:x.py", "file", 2, 2, files=["x.py"]),
            atom("c", "error", 1, 0, files=["x.py"]),
            atom("lonely", "reply", 1, 0),
        ]
        self.g = Graph(self.atoms)

    def test_edge_probability_and_weight(self):
        e = self.g.adj["a"]["f:x.py"]
        self.assertAlmostEqual(e.p, 0.8)
        self.assertAlmostEqual(e.weight, -math.log(0.8))

    def test_most_probable_chain(self):
        rel = shortest_path_relevance(self.g, {"u2": 1.0})
        # via the file: .6 * .8 * .8 = .384; via error-then-fix: .6 * .9 = .54
        self.assertAlmostEqual(rel["c"], 0.6 * 0.9, places=6)
        self.assertEqual(rel["lonely"], 0.0)
        _, prev = dijkstra(self.g, {"u2": 1.0})
        self.assertEqual(path_to(prev, "c"), ["u2", "a", "c"])

    def test_noisy_or_combines_reasons(self):
        g = Graph([atom("p", "error", 1, 1, files=["x.py"], symbols=["get_balance"]),
                   atom("q", "edit", 1, 2, files=["x.py"], symbols=["get_balance"])])
        p_sym = 0.6 / (1 + math.log(2))
        e = g.adj["p"]["q"]
        self.assertEqual(e.reasons, ["symbol get_balance", "error then fix"])
        self.assertAlmostEqual(e.p, 1 - (1 - p_sym) * (1 - 0.9))

    def test_rehearsal_sets_last_used(self):
        atoms = [atom("old", "edit", 1, 1, files=["y.py"]), atom("f:y.py", "file", 1, 1, files=["y.py"]),
                 atom("new", "edit", 9, 9, files=["y.py"])]
        Graph(atoms)
        self.assertEqual(atoms[0].last_used, 1)  # an edit is not reused by a later edit of the file...
        self.assertEqual(atoms[1].last_used, 9)  # ...but the file itself is


class DecayTest(unittest.TestCase):
    def test_half_life(self):
        self.assertAlmostEqual(decay(10, 2, 8), 0.5)
        self.assertAlmostEqual(decay(10, 10, 8), 1.0)
        self.assertAlmostEqual(decay(18, 2, 8), 0.25)
        self.assertEqual(decay(5, 1, 0), 1.0)


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = config.load("/nonexistent", env={})
        self.atoms = extract_atoms(parse_transcript(FIX).events, cwd="/work/ledger")

    def test_pins_survive_any_budget(self):
        text, sel = build_pack(self.atoms, self.cfg, "replace", budget=50)
        for a in self.atoms:
            if a.pinned:
                self.assertIn(a.id, sel.kept)
                self.assertIn(a.text, text)  # verbatim
        self.assertTrue(sel.overflow)

    def test_pack_fits_budget(self):
        for b in (400, 800, 6000):
            text, sel = build_pack(self.atoms, self.cfg, "replace", budget=b)
            self.assertLessEqual(estimate(text), b)
            self.assertFalse(sel.overflow)

    def test_latest_request_is_last(self):
        text, _ = build_pack(self.atoms, self.cfg, "replace", budget=6000)
        self.assertTrue(text.rstrip().endswith('user: "Now add a refund() function."'))

    def test_recency_baseline_keeps_a_contiguous_tail(self):
        cfg = dict(self.cfg, selector="recency")
        _, sel = build_pack(self.atoms, cfg, "replace", budget=300)
        self.assertEqual(sel.pinned, [])
        self.assertIn("u3", sel.kept)
        self.assertNotIn("u1", sel.kept)

    def test_greedy_switch(self):
        cfg = dict(self.cfg, solver="greedy")
        _, sel = build_pack(self.atoms, cfg, "replace", budget=400)
        self.assertEqual(sel.solver, "greedy")


class StoreAndCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self.cwd = "/work/ledger"

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_numbers_turns_and_seq(self):
        st = Store(self.home)
        t = st.new_thread(self.cwd, "s1")
        e1 = t.append(PROMPT, {"text": "hi"}, "s1", new_turn=True)
        e2 = t.append("tool", {"name": "Bash"}, "s1")
        e3 = t.append(PROMPT, {"text": "again"}, "s1", new_turn=True)
        self.assertEqual([(e.seq, e.turn) for e in (e1, e2, e3)], [(1, 1), (2, 1), (3, 2)])
        self.assertEqual(len(t.events()), 3)

    def test_clear_continues_the_thread(self):
        st = Store(self.home)
        t = st.new_thread(self.cwd, "s1")
        t.append(PROMPT, {"text": "hi"}, "s1", new_turn=True)
        st.record_end(self.cwd, "s1", "clear")
        nxt = st.thread_after_clear(self.cwd)
        self.assertEqual(nxt.id, t.id)
        st.attach(self.cwd, "s2", nxt)
        self.assertEqual(st.lookup(self.cwd, "s2").id, t.id)
        self.assertEqual(t.meta()["sessions"], ["s1", "s2"])

    def test_selection_log_and_cli(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["--home", self.home, "--cwd", self.cwd, "import", FIX])
        st = Store(self.home)
        t = st.find(self.cwd)
        cfg = config.load(self.home, env={})
        text, rec = select_and_log(t, cfg, "replace", "clear")
        self.assertTrue(os.path.exists(rec["log_path"]))
        self.assertEqual(len(rec["kept"]) + len(rec["dropped"]), rec["n_atoms"])
        self.assertTrue(all("value" in r for r in rec["kept"]))
        for args in (["show"], ["explain", "e3"], ["get", "t3"], ["log"], ["stats"], ["pack", "--budget", "300"]):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["--home", self.home, "--cwd", self.cwd] + args), 0)
            self.assertTrue(out.getvalue().strip(), args)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["--home", self.home, "--cwd", self.cwd, "get", "t3"])
        self.assertIn("AssertionError: expected 1050 got 1050.0", out.getvalue())


if __name__ == "__main__":
    unittest.main()
