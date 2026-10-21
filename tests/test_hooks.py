import contextlib
import io
import json
import os
import tempfile
import unittest

from context_graph import cli, hooks
from context_graph.store import Store
from context_graph.trim import trim_response, trim_text

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "session_basic.jsonl")
TRIM = {"enabled": True, "tools": ["Bash"], "max_lines": 80, "max_chars": 8000,
        "head_lines": 25, "tail_lines": 25, "max_error_lines": 20}


class TrimTest(unittest.TestCase):
    def test_short_output_untouched(self):
        self.assertIsNone(trim_text("a\nb\nc", TRIM, "t1"))

    def test_long_output_keeps_head_tail_errors(self):
        lines = [f"line {i}" for i in range(1, 301)]
        lines[150] = "ERROR: disk quota exceeded"
        out = trim_text("\n".join(lines), TRIM, "t7")
        kept = out.split("\n")
        self.assertEqual(kept[0], "line 1")
        self.assertEqual(kept[-1], "line 300")
        self.assertIn("ERROR: disk quota exceeded", kept)
        self.assertNotIn("line 100", kept)
        self.assertIn("ctx get t7", out)
        self.assertLess(len(kept), 60)

    def test_shape_is_preserved(self):
        resp = {"stdout": "\n".join(map(str, range(500))), "stderr": "", "interrupted": False, "isImage": False}
        new = trim_response("Bash", resp, TRIM, "t3")
        self.assertEqual(set(new), set(resp))
        self.assertEqual(resp["stdout"].count("\n"), 499)  # original not mutated
        self.assertIsNone(trim_response("Grep", resp, TRIM, "t3"))
        self.assertIsNone(trim_response("Bash", resp, dict(TRIM, enabled=False), "t3"))

    def test_long_single_line(self):
        out = trim_text("x" * 20000, TRIM, "t9")
        self.assertLess(len(out), 8000)
        self.assertIn("characters hidden", out)


class HookFlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"CONTEXT_GRAPH_HOME": self.tmp.name, "CLAUDE_PLUGIN_ROOT": "/plugin"}
        self.cwd = "/work/demo"

    def tearDown(self):
        self.tmp.cleanup()

    def fire(self, payload):
        payload.setdefault("cwd", self.cwd)
        payload.setdefault("session_id", "s1")
        payload.setdefault("transcript_path", "/nonexistent.jsonl")
        out = io.StringIO()
        self.assertEqual(hooks.main(io.StringIO(json.dumps(payload)), out, self.env), 0)
        return json.loads(out.getvalue()) if out.getvalue() else None

    def ctx(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["--home", self.tmp.name, "--cwd", self.cwd] + list(args))
        return out.getvalue()

    def session(self):
        self.assertIsNone(self.fire({"hook_event_name": "SessionStart", "source": "startup"}))
        self.fire({"hook_event_name": "UserPromptSubmit",
                   "prompt": "Never commit generated files. Run the build and show me the log."})
        big = "\n".join(f"build step {i}" for i in range(1, 401))
        out = self.fire({"hook_event_name": "PostToolUse", "tool_name": "Bash",
                         "tool_input": {"command": "make build"},
                         "tool_response": {"stdout": big, "stderr": "", "interrupted": False, "isImage": False}})
        self.fire({"hook_event_name": "PostToolUseFailure", "tool_name": "Bash",
                   "tool_input": {"command": "python -m pytest src/app.py"},
                   "error": "Exit code 1\nTraceback (most recent call last):\n  File \"src/app.py\", line 3\nNameError: name 'cfg' is not defined"})
        self.fire({"hook_event_name": "Stop",
                   "last_assistant_message": "The root cause is a missing import of cfg in src/app.py."})
        return out

    def test_capture_trim_and_page_back(self):
        out = self.session()
        upd = out["hookSpecificOutput"]["updatedToolOutput"]
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"], "PostToolUse")
        self.assertIn("ctx get t2", upd["stdout"])
        self.assertLess(upd["stdout"].count("\n"), 60)
        full = self.ctx("get", "t2")
        self.assertIn("build step 200", full)
        self.assertEqual(self.ctx("get", "t2", "--lines", "200:201").split(), ["build", "step", "200", "build", "step", "201"])
        self.assertEqual(full.split("\n")[0], "build step 1")
        # paging back with ctx get through Bash is never trimmed again
        out = self.fire({"hook_event_name": "PostToolUse", "tool_name": "Bash",
                         "tool_input": {"command": "ctx get t2"},
                         "tool_response": {"stdout": full, "stderr": "", "interrupted": False, "isImage": False}})
        self.assertIsNone(out)
        err = self.ctx("explain", "e3")
        self.assertIn("NameError", err)

    def test_compact_injects_assist_pack(self):
        self.session()
        self.fire({"hook_event_name": "PreCompact", "trigger": "manual", "custom_instructions": ""})
        out = self.fire({"hook_event_name": "SessionStart", "source": "compact"})
        ctx = out["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"], "SessionStart")
        self.assertIn("Never commit generated files.", ctx)
        self.assertIn("The root cause is a missing import of cfg in src/app.py.", ctx)
        self.fire({"hook_event_name": "PostCompact", "trigger": "manual", "compact_summary": "Summary: built."})
        t = Store(self.tmp.name).find(self.cwd)
        self.assertTrue(os.path.exists(os.path.join(t.dir, "native_summaries.jsonl")))
        self.assertTrue(os.listdir(os.path.join(t.dir, "snapshots")))
        self.assertTrue(t.selections()[-1].endswith("compact.json"))

    def test_clear_carries_the_thread_to_a_new_session(self):
        self.session()
        self.fire({"hook_event_name": "SessionEnd", "reason": "clear"})
        out = self.fire({"hook_event_name": "SessionStart", "source": "clear", "session_id": "s2"})
        ctx = out["hookSpecificOutput"]["additionalContext"]
        self.assertIn("conversation was cleared", ctx)
        self.assertIn("Never commit generated files.", ctx)
        self.fire({"hook_event_name": "UserPromptSubmit", "prompt": "continue", "session_id": "s2"})
        st = Store(self.tmp.name)
        self.assertEqual(st.lookup(self.cwd, "s2").id, st.lookup(self.cwd, "s1").id)
        self.assertEqual(st.lookup(self.cwd, "s2").meta()["turn"], 2)

    def test_unknown_session_backfills_from_transcript(self):
        self.fire({"hook_event_name": "UserPromptSubmit", "prompt": "Now add a refund() function.",
                   "session_id": "fork-1", "transcript_path": FIX, "cwd": "/work/ledger"})
        t = Store(self.tmp.name).lookup("/work/ledger", "fork-1")
        prompts = [e.data["text"] for e in t.events() if e.kind == "prompt"]
        self.assertEqual(len(prompts), 3)  # 3 from the transcript, the current one not doubled
        self.assertEqual(prompts[-1], "Now add a refund() function.")

    def test_disabled_and_garbage_input(self):
        out = io.StringIO()
        self.assertEqual(hooks.main(io.StringIO("not json"), out, self.env), 0)
        env = dict(self.env, CONTEXT_GRAPH_DISABLE="1")
        self.assertEqual(hooks.main(io.StringIO("{}"), out, env), 0)
        self.assertEqual(out.getvalue(), "")

    def test_errors_are_logged_not_raised(self):
        out = io.StringIO()
        bad = {"hook_event_name": "UserPromptSubmit", "cwd": self.cwd, "session_id": "s1", "prompt": "x"}
        os.makedirs(self.tmp.name, exist_ok=True)
        # make the projects path a file so the store cannot be created
        with open(os.path.join(self.tmp.name, "projects"), "w") as fh:
            fh.write("x")
        self.assertEqual(hooks.main(io.StringIO(json.dumps(bad)), out, self.env), 0)
        self.assertEqual(out.getvalue(), "")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "errors.log")))


if __name__ == "__main__":
    unittest.main()
