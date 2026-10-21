import os
import unittest

from context_graph.events import PROMPT, REPLY, TOOL
from context_graph.transcript import parse_transcript

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "session_basic.jsonl")


class TranscriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr = parse_transcript(FIX)

    def test_prompts_are_verbatim_and_numbered(self):
        prompts = [e for e in self.tr.events if e.kind == PROMPT]
        self.assertEqual([e.turn for e in prompts], [1, 2, 3])
        self.assertEqual(
            prompts[0].data["text"],
            "Never use floats for money; store every amount as integer cents. "
            "Can you look at ledger/core.py and run the tests?",
        )
        self.assertEqual(prompts[2].data["text"], "Now add a refund() function.")

    def test_local_commands_meta_and_sidechain_are_skipped(self):
        texts = [e.data.get("text", "") for e in self.tr.events if e.kind == PROMPT]
        self.assertFalse(any("command-name" in t or "Caveat" in t for t in texts))
        self.assertFalse(any("subagent" in t for t in texts))
        self.assertFalse(any("continued from a previous conversation" in t for t in texts))

    def test_tool_calls_pair_with_results(self):
        tools = [e for e in self.tr.events if e.kind == TOOL]
        self.assertEqual([t.data["name"] for t in tools], ["Read", "Bash", "Edit", "Bash", "Write"])
        failed = tools[1]
        self.assertIn("Exit code 1", failed.data["error"])
        self.assertIn("AssertionError", failed.data["error"])
        self.assertIsNone(failed.data["response"])
        ok = tools[3]
        self.assertEqual(ok.data["response"]["stdout"], "..\n2 passed in 0.02s")
        self.assertNotIn("error", ok.data)

    def test_reply_is_last_assistant_message_of_turn(self):
        replies = {e.turn: e.data["text"] for e in self.tr.events if e.kind == REPLY}
        self.assertTrue(replies[1].startswith("The root cause is"))
        # turn 2 had two text messages; only the final one is the reply (as in the Stop hook)
        self.assertTrue(replies[2].startswith("I decided to keep amounts"))
        self.assertIn("refund", replies[3])

    def test_sequence_is_monotonic(self):
        seqs = [e.seq for e in self.tr.events]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))

    def test_compaction_metadata_and_summary(self):
        self.assertEqual(len(self.tr.compactions), 1)
        c = self.tr.compactions[0]
        self.assertEqual((c.trigger, c.pre_tokens, c.post_tokens, c.duration_ms), ("manual", 437210, 21450, 134000))
        self.assertTrue(c.summary.startswith("This session is being continued from a previous conversation"))

    def test_truncated_last_line_is_tolerated(self):
        self.assertEqual(self.tr.session_ids, ["sess-0001"])


if __name__ == "__main__":
    unittest.main()
