import os
import unittest

from context_graph.atoms import extract_atoms, find_paths, find_symbols, is_constraint, split_sentences
from context_graph.transcript import parse_transcript

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "session_basic.jsonl")


class RulesTest(unittest.TestCase):
    def test_constraints(self):
        yes = [
            "Never use floats for money; store every amount as integer cents.",
            "Don't add third-party dependencies.",
            "Please always run the linter before committing.",
            "The CLI output must stay CSV.",
            "From now on, keep functions under 40 lines.",
        ]
        no = [
            "Can you look at ledger/core.py and run the tests?",
            "I don't know why this fails.",
            "Must we really do this?",
            "Sounds good.",
        ]
        for s in yes:
            self.assertTrue(is_constraint(s), s)
        for s in no:
            self.assertFalse(is_constraint(s), s)

    def test_sentences_skip_code_fences(self):
        text = "First line. Second one!\n```\nnever = 1\n```\n- Bullet item"
        self.assertEqual(split_sentences(text), ["First line.", "Second one!", "Bullet item"])

    def test_paths_and_symbols(self):
        self.assertEqual(find_paths('see ledger/core.py and File "/a/b.py", line 3'), ["/a/b.py", "ledger/core.py"])
        self.assertNotIn("example.com", " ".join(find_paths("visit https://example.com/x.html")))
        syms = find_symbols("call get_balance(entries) on LedgerStore; `parse_amount` too, not this")
        self.assertEqual(syms, ["get_balance", "LedgerStore", "parse_amount"])


class ExtractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atoms = extract_atoms(parse_transcript(FIX).events, cwd="/work/ledger")
        cls.by_id = {a.id: a for a in cls.atoms}

    def test_turn_atoms(self):
        self.assertIn("u1", self.by_id)
        self.assertEqual(self.by_id["u3"].text, "Now add a refund() function.")

    def test_constraints_are_pinned_verbatim(self):
        k = self.by_id["k1.1"]
        self.assertTrue(k.pinned)
        self.assertEqual(k.text, "Never use floats for money; store every amount as integer cents.")
        k2 = [a for a in self.atoms if a.kind == "constraint" and a.turn == 2]
        self.assertEqual([a.text for a in k2], ["Don't add third-party dependencies."])

    def test_decisions_from_user_and_assistant(self):
        dec = {a.id: a.text for a in self.atoms if a.kind == "decision"}
        self.assertIn("Let's use SQLite for storage, not a JSON file.", dec.values())
        self.assertTrue(any(t.startswith("The root cause is") for t in dec.values()))
        self.assertTrue(all(self.by_id[i].pinned for i in dec))

    def test_failed_command_makes_error_atom(self):
        cmd = [a for a in self.atoms if a.kind == "command"][0]
        self.assertEqual(cmd.meta["exit"], 1)
        self.assertIn("exit 1", cmd.render)
        err = self.by_id["e" + cmd.id[1:]]
        self.assertEqual(err.parent, cmd.id)
        self.assertIn("ledger/core.py", err.files)
        self.assertIn("AssertionError: expected 1050 got 1050.0", err.text)

    def test_file_atoms_and_relative_paths(self):
        f = self.by_id["f:ledger/core.py"]
        self.assertIn("edited (turns 2)", f.render)
        self.assertIn("read (turns 1)", f.render)
        self.assertIn("f:ledger/refund.py", self.by_id)
        # the bare mention in turn 1 resolves to the same file
        self.assertIn("ledger/core.py", self.by_id["u1"].files)

    def test_tokens_positive(self):
        self.assertTrue(all(a.tokens > 0 for a in self.atoms))


if __name__ == "__main__":
    unittest.main()
