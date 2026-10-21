"""Claude Code hook handler. One entry point for every event.

  SessionStart        map the session to a thread; after /compact (source
                      "compact") or /clear (source "clear") return the pack
                      as additionalContext
  UserPromptSubmit    store the prompt word for word; it becomes the focus
  PostToolUse         store the tool call with its full output, then return a
                      trimmed copy as updatedToolOutput when it is long
  PostToolUseFailure  store the failed call and its error text
  Stop                store last_assistant_message
  PreCompact          snapshot the event log
  PostCompact         store the native compact_summary (the baseline)
  SessionEnd          remember which session ended and why, so the session
                      that /clear starts can find its thread

A hook must never break the session, so every failure is caught, written to
<home>/errors.log, and the hook exits 0 with no output.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from typing import Any, TextIO

from . import config as cfgmod
from .engine import build_pack, select_and_log, thread_atoms
from .events import PROMPT, REPLY, TOOL
from .store import Store, Thread
from .transcript import parse_transcript
from .pack import split_parts
from .trim import trim_response

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ctx_command(env: dict[str, str]) -> str:
    # Inside a plugin, Claude Code puts the plugin's bin/ on the Bash tool's PATH.
    if env.get("CLAUDE_PLUGIN_ROOT"):
        return "ctx"
    return os.path.join(PLUGIN_ROOT, "bin", "ctx")


_CTX_GET = re.compile(r"(?:^|[\s;&|(/])ctx\s+get\b")


def _pages_back(tool: str, tool_input: Any) -> bool:
    """Output that `ctx get` pages back must reach the model whole, not be trimmed again."""
    return tool == "Bash" and isinstance(tool_input, dict) and bool(_CTX_GET.search(str(tool_input.get("command", ""))))


def _backfill(thread: Thread, transcript_path: str, drop_last_prompt: str | None = None) -> int:
    """Import history the hooks did not see (plugin enabled mid-session, or a forked session)."""
    if not transcript_path or not os.path.isfile(transcript_path) or thread.events():
        return 0
    events = parse_transcript(transcript_path).events
    if drop_last_prompt is not None:
        for i in range(len(events) - 1, -1, -1):
            if events[i].kind == PROMPT:
                if events[i].data.get("text") == drop_last_prompt:
                    events = events[:i] + events[i + 1:]
                break
    for ev in events:
        thread.append(ev.kind, ev.data, ev.session_id, new_turn=ev.kind == PROMPT)
    return len(events)


def resolve_thread(store: Store, p: dict[str, Any]) -> Thread:
    cwd = p.get("cwd") or os.getcwd()
    sid = p.get("session_id", "")
    t = store.lookup(cwd, sid)
    if t is not None:
        return t
    if p.get("hook_event_name") == "SessionStart" and p.get("source") == "clear":
        prev = store.thread_after_clear(cwd)
        if prev is not None:
            return store.attach(cwd, sid, prev)
    t = store.new_thread(cwd, sid)
    if not (p.get("hook_event_name") == "SessionStart" and p.get("source") == "startup"):
        current = p.get("prompt") if p.get("hook_event_name") == "UserPromptSubmit" else None
        _backfill(t, p.get("transcript_path", ""), drop_last_prompt=current)
    return t


def _pack_mode(p: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    source = p.get("source")
    if source == "clear" and cfg["inject_on_clear"]:
        return "replace"
    if source == "compact" and cfg["inject_on_compact"]:
        return "assist"
    return None


def _part_output(text: str, part: int, cfg: dict[str, Any]) -> dict[str, Any] | None:
    parts = split_parts(text, int(cfg["pack_part_chars"]))
    if part > len(parts):
        return None
    return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": parts[part - 1]}}


def handle_extra_part(p: dict[str, Any], env: dict[str, str], part: int) -> dict[str, Any] | None:
    """SessionStart parts 2..n: rebuild the same pack read-only and return one part.

    Hook commands for one event run in parallel, so these must not write: part 1
    maps the session, writes the selection log and returns the first part. The
    selection is deterministic, so every part sees the same pack.
    """
    if p.get("hook_event_name") != "SessionStart":
        return None
    cwd = p.get("cwd") or os.getcwd()
    store = Store(env.get("CONTEXT_GRAPH_HOME") or None)
    cfg = cfgmod.load(store.home, cwd, env)
    mode = _pack_mode(p, cfg)
    if mode is None:
        return None
    thread = store.lookup(cwd, p.get("session_id", ""))
    if thread is None and p.get("source") == "clear":
        thread = store.thread_after_clear(cwd)
    if thread is None or not thread.events():
        return None
    text, _ = build_pack(thread_atoms(thread, cfg), cfg, mode, ctx=ctx_command(env))
    return _part_output(text, part, cfg)


def handle(p: dict[str, Any], env: dict[str, str]) -> dict[str, Any] | None:
    event = p.get("hook_event_name", "")
    cwd = p.get("cwd") or os.getcwd()
    sid = p.get("session_id", "")
    store = Store(env.get("CONTEXT_GRAPH_HOME") or None)
    cfg = cfgmod.load(store.home, cwd, env)

    if event == "SessionEnd":
        store.record_end(cwd, sid, str(p.get("reason", "")))
        return None

    thread = resolve_thread(store, p)

    if event == "SessionStart":
        mode = _pack_mode(p, cfg)
        if mode is None or not thread.events():
            return None
        text, _ = select_and_log(thread, cfg, mode, trigger=str(p.get("source")), ctx=ctx_command(env))
        return _part_output(text, 1, cfg)

    if event == "UserPromptSubmit":
        thread.append(PROMPT, {"text": str(p.get("prompt", ""))}, sid, new_turn=True)
        return None

    if event in ("PostToolUse", "PostToolUseFailure"):
        name = str(p.get("tool_name", ""))
        data: dict[str, Any] = {"name": name, "input": p.get("tool_input"), "response": p.get("tool_response"),
                                "cwd": cwd}
        if event == "PostToolUseFailure":
            data["error"] = str(p.get("error", ""))
            if p.get("is_interrupt"):
                data["interrupted"] = True
        ev = thread.append(TOOL, data, sid)
        if event == "PostToolUse" and not _pages_back(name, p.get("tool_input")):
            trimmed = trim_response(name, p.get("tool_response"), cfg["trim"], f"t{ev.seq}", ctx_command(env))
            if trimmed is not None:
                return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "updatedToolOutput": trimmed}}
        return None

    if event == "Stop":
        text = str(p.get("last_assistant_message") or "").strip()
        if text:
            thread.append(REPLY, {"text": text}, sid)
        return None

    if event == "PreCompact":
        thread.snapshot(f"precompact-{p.get('trigger', '')}")
        return None

    if event == "PostCompact":
        summary = str(p.get("compact_summary") or "")
        thread.add_native_summary({"session_id": sid, "trigger": p.get("trigger"), "time": time.time(),
                                   "chars": len(summary), "compact_summary": summary})
        return None

    return None


def main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, env: dict[str, str] | None = None,
         argv: list[str] | None = None) -> int:
    env = dict(os.environ) if env is None else env
    argv = sys.argv[1:] if argv is None else argv
    part = int(argv[argv.index("--part") + 1]) if "--part" in argv else 1
    if env.get("CONTEXT_GRAPH_DISABLE", "").lower() in ("1", "true", "yes"):
        return 0
    try:
        payload = json.load(stdin)
    except ValueError:
        return 0
    try:
        out = handle(payload, env) if part == 1 else handle_extra_part(payload, env, part)
    except Exception:  # never break the user's session
        try:
            home = Store(env.get("CONTEXT_GRAPH_HOME") or None).home
            os.makedirs(home, exist_ok=True)
            with open(os.path.join(home, "errors.log"), "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {payload.get('hook_event_name')}\n")
                fh.write(traceback.format_exc() + "\n")
        except OSError:
            pass
        return 0
    if out is not None:
        stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
