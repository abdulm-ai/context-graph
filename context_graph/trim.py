"""Trim long tool output at the source.

The PostToolUse hook stores the full output in the graph first, then hands the
model a shorter copy through ``updatedToolOutput``: the first and last lines,
every error line in between, and the atom id to page the rest back with
``ctx get <id>``. This is MemGPT's idea (Packer et al. 2023) of keeping the
full record outside the context window and paging it in on demand.

The replacement has to keep the tool's output shape, so only the text fields
change: ``stdout`` / ``stderr`` for Bash, ``file.content`` for Read.
"""

from __future__ import annotations

import copy
import re
from typing import Any

_ERROR_LINE = re.compile(
    r"(Traceback|Error\b|Exception\b|FAILED|FAIL:|\bfailed\b|error:|ERROR|assert|"
    r"No such file|not found|Permission denied|panic:|fatal:|SyntaxError|Errno|warning:)"
)
MAX_LINE_CHARS = 400


def trim_text(text: str, cfg: dict[str, Any], atom_id: str, ctx: str = "ctx") -> str | None:
    """Return a trimmed copy of text, or None if it is short enough already."""
    if not text:
        return None
    lines = text.split("\n")
    max_lines = int(cfg.get("max_lines", 80))
    max_chars = int(cfg.get("max_chars", 8000))
    if len(lines) <= max_lines and len(text) <= max_chars:
        return None
    head_n = int(cfg.get("head_lines", 25))
    tail_n = int(cfg.get("tail_lines", 25))
    if len(lines) <= head_n + tail_n:
        # few but very long lines (minified JSON, one-line logs): cut by characters
        keep = max(200, max_chars // 2)
        hidden = len(text) - keep - keep // 2
        return (text[:keep] + f"\n[context-graph: {hidden} characters hidden. Full output: `{ctx} get {atom_id}`]\n"
                + text[-(keep // 2):])
    head, middle, tail = lines[:head_n], lines[head_n:-tail_n], lines[-tail_n:]
    errs = [ln for ln in middle if _ERROR_LINE.search(ln)]
    max_err = int(cfg.get("max_error_lines", 20))
    errs = errs[-max_err:]

    def clip(ln: str) -> str:
        return ln if len(ln) <= MAX_LINE_CHARS else ln[:MAX_LINE_CHARS] + " [...]"

    note = (f"[context-graph: lines {head_n + 1}-{len(lines) - tail_n} of {len(lines)} hidden"
            + (f"; the {len(errs)} error lines among them are shown next" if errs else "")
            + f". Full output: `{ctx} get {atom_id}`, or a range: `{ctx} get {atom_id} --lines A:B`]")
    out = [clip(x) for x in head] + [note] + [clip(x) for x in errs]
    if errs:
        out.append("[context-graph: end of kept error lines]")
    out += [clip(x) for x in tail]
    return "\n".join(out)


def trim_response(tool: str, response: Any, cfg: dict[str, Any], atom_id: str, ctx: str = "ctx") -> Any | None:
    """Trimmed copy of a tool_response with the same shape, or None if unchanged."""
    if not cfg.get("enabled", True) or tool not in cfg.get("tools", ["Bash"]):
        return None
    if not isinstance(response, dict):
        return None
    new = copy.deepcopy(response)
    changed = False
    # The full output is paged back with `ctx get`, which prints stdout then stderr
    # as one text, so line numbers in the note refer to that combined text.
    if tool == "Bash":
        for key in ("stdout", "stderr"):
            val = new.get(key)
            if isinstance(val, str):
                t = trim_text(val, cfg, atom_id, ctx)
                if t is not None:
                    new[key] = t
                    changed = True
    elif tool == "Read" and isinstance(new.get("file"), dict):
        val = new["file"].get("content")
        if isinstance(val, str):
            t = trim_text(val, cfg, atom_id, ctx)
            if t is not None:
                new["file"]["content"] = t
                changed = True
    return new if changed else None
