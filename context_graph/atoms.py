"""Turn events into atoms, the nodes of the session graph.

Atom kinds:
  turn        a user prompt, word for word                      id u<turn>
  constraint  a rule the user stated (pinned, verbatim)         id k<turn>.<n>
  decision    a decision, rejected approach or root cause,
              stated by the user or the assistant (pinned)      id d<turn>.<n>
  reply       the assistant's final message of a turn           id r<turn>
  command     a Bash command with its exit status               id t<seq>
  edit        an Edit / Write / MultiEdit                        id t<seq>
  lookup      a Read / Grep / Glob                               id t<seq>
  tool        any other tool call                                id t<seq>
  error       the error lines of a failed or erroring tool call  id e<seq>
  file        a file path touched during the session             id f:<path>

Ids are derived from the event sequence, so they are stable: the hooks can
hand an id to the model (``ctx get t12``) before the graph is ever built.

Detection of constraints and decisions is rule based (regular expressions over
sentences). There is no model call anywhere in extraction or selection.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .events import PROMPT, REPLY, TOOL, Event, tool_output_text
from .tokens import estimate

PINNED_KINDS = ("constraint", "decision")


@dataclass
class Atom:
    id: str
    kind: str
    turn: int
    seq: int
    text: str  # full verbatim content, returned by `ctx get`
    render: str  # the line that goes into a pack
    source: str = ""  # user | assistant | tool
    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    pinned: bool = False
    parent: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    last_used: int = 0  # turn this atom was last created or referenced again

    def to_json(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("text")
        return d


# ---------------------------------------------------------------- text rules

_IMPERATIVE_START = re.compile(
    r"^(?:please\s+|and\s+|but\s+|also\s+|so\s+|oh,?\s+and\s+)?"
    r"(?:don'?t|do not|never|avoid|stop|always|keep|make sure|ensure)\b",
    re.I,
)
_CONSTRAINT = re.compile(
    r"\b(?:must|mustn'?t|shall|should never|should always|should not|shouldn'?t|"
    r"not allowed|is required|are required|required to|from now on|going forward|"
    r"make sure|ensure that|never|hard rule|rule:|constraint:|requirement:|"
    r"remember:|important:)\b",
    re.I,
)
_DECISION_USER = re.compile(
    r"\b(?:let'?s (?:use|go with|keep|switch|drop|stick with|not)|we'?ll (?:use|go with|keep)|"
    r"we will (?:use|go with|keep)|i(?:'ve| have)? decided|decided|decision:|go with|stick with|"
    r"we tried|tried that|didn'?t work|did not work|rejected|ruled out|"
    r"root cause|the cause (?:is|was)|caused by)\b",
    re.I,
)
_DECISION_ASSISTANT = re.compile(
    r"\b(?:root cause|the cause (?:is|was)|caused by|the bug (?:is|was)|"
    r"decided|decision|i'?ll (?:use|go with|keep)|we'?ll (?:use|go with|keep)|"
    r"going with|settled on|opted|chose|switched to|rejected|ruled out|"
    r"(?:doesn'?t|does not|didn'?t|did not|won'?t) work|reverted|abandoned)\b",
    re.I,
)
_STRONG_DECISION = re.compile(r"\b(?:root cause|decided|decision|rejected|ruled out)\b", re.I)
_MAX_ASSISTANT_DECISIONS = 3
_MAX_TURN_CHARS = 1500  # longer prompts (pasted logs) are clipped in the pack, kept whole in the graph

_PATH = re.compile(
    r"(?<![\w/.-])((?:\.{0,2}/)?(?:[\w.-]+/)*[\w-][\w.-]*\.(?:py|pyi|js|mjs|cjs|ts|tsx|jsx|json|"
    r"jsonl|md|rst|txt|toml|ya?ml|cfg|ini|sh|bash|go|rs|java|kt|c|h|cc|cpp|hpp|rb|php|css|"
    r"scss|html|sql|csv|lock|env|xml|proto|tf))(?![\w/])"
)
_TRACEBACK_FILE = re.compile(r'File "([^"]+)", line (\d+)')
_IDENT = re.compile(r"`([^`\n]{2,80})`|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(\()?")
_ERROR_LINE = re.compile(
    r"(Traceback|Error\b|Exception\b|FAILED|FAIL:|\bfailed\b|error:|ERROR|assert|"
    r"Exit code \d+|No such file|not found|Permission denied|panic:|fatal:|"
    r"SyntaxError|Errno)"
)
_STOP_SYMBOLS = frozenset(
    """self cls None True False print return import from with else elif while
    for def class pass break continue lambda yield async await assert raise
    try except finally global nonlocal this that there these those which
    when where what should would could None null true false main test tests
    file files path line lines todo note json http https""".split()
)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, skipping fenced code blocks."""
    out: list[str] = []
    in_fence = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+|#+\s+|>\s*)", "", line).strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z`\"'(\[])", line):
            part = part.strip()
            if part:
                out.append(part)
    return out


def is_constraint(sentence: str) -> bool:
    if sentence.endswith("?"):
        return False
    return bool(_IMPERATIVE_START.search(sentence) or _CONSTRAINT.search(sentence))


def find_paths(text: str) -> list[str]:
    seen: list[str] = []
    for m in _TRACEBACK_FILE.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    for m in _PATH.finditer(text):
        p = m.group(1)
        if "://" in p or p.startswith("www."):
            continue
        if p not in seen:
            seen.append(p)
    return seen


def find_symbols(text: str) -> list[str]:
    found: list[str] = []
    for m in _IDENT.finditer(text):
        if m.group(1):
            cand = m.group(1).strip()
            # a backticked span: keep it whole if it is identifier-like
            if re.fullmatch(r"[\w.]+(\(\))?", cand):
                cand = cand.rstrip("()")
            else:
                continue
            called = True
        else:
            cand = m.group(2)
            called = bool(m.group(3))
        base = cand.split(".")[-1]
        if len(base) < 4 or base.lower() in _STOP_SYMBOLS or base.isdigit():
            continue
        snake = "_" in base.strip("_")
        camel = bool(re.search(r"[a-z][A-Z]", base)) or bool(re.match(r"[A-Z][a-z]+[A-Z]", base))
        if snake or camel or called:
            if cand not in found:
                found.append(cand)
    return found


def error_lines(text: str, limit: int = 12) -> list[str]:
    lines = text.splitlines()
    hits = [i for i, ln in enumerate(lines) if _ERROR_LINE.search(ln)]
    if not hits:
        return []
    keep: list[int] = []
    for i in hits:
        if i not in keep:
            keep.append(i)
    # the end of a traceback (the exception itself) matters most: keep the last lines
    keep = keep[-limit:]
    return [lines[i].rstrip()[:300] for i in keep]


def _norm_path(p: str, cwd: str) -> str:
    if cwd and os.path.isabs(p):
        try:
            rel = os.path.relpath(p, cwd)
        except ValueError:
            return p
        if not rel.startswith(".."):
            return rel
    if p.startswith("./"):
        return p[2:]
    return p


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


# ---------------------------------------------------------------- tools


def _exit_status(data: dict[str, Any]) -> tuple[int | str, str]:
    """Return (exit status, error text) for a tool event."""
    err = data.get("error")
    resp = data.get("response")
    if isinstance(resp, dict):
        for key in ("returnCode", "exit_code", "exitCode", "code"):
            if isinstance(resp.get(key), int):
                return resp[key], str(err or "")
        if resp.get("interrupted"):
            return "interrupted", str(err or "")
    if err:
        m = re.search(r"Exit code (\d+)", str(err))
        return (int(m.group(1)) if m else 1), str(err)
    return 0, ""


def _edit_text(name: str, inp: dict[str, Any]) -> tuple[str, int, int]:
    if name == "Write":
        content = str(inp.get("content", ""))
        return content, 0, content.count("\n") + 1
    edits = inp.get("edits") if name == "MultiEdit" else [inp]
    chunks, removed, added = [], 0, 0
    for e in edits or []:
        old, new = str(e.get("old_string", "")), str(e.get("new_string", ""))
        removed += old.count("\n") + 1 if old else 0
        added += new.count("\n") + 1 if new else 0
        chunks.append("\n".join(["- " + ln for ln in old.splitlines()] + ["+ " + ln for ln in new.splitlines()]))
    return "\n".join(chunks), removed, added


def _tool_atoms(ev: Event, cwd: str) -> list[Atom]:
    d = ev.data
    name = str(d.get("name", ""))
    inp = d.get("input") or {}
    if not isinstance(inp, dict):
        inp = {"value": inp}
    out_text = tool_output_text(d.get("response"))
    status, err_text = _exit_status(d)
    aid = f"t{ev.seq}"
    files: list[str] = []
    for key in ("file_path", "path", "notebook_path"):
        if isinstance(inp.get(key), str) and inp[key]:
            files.append(_norm_path(inp[key], cwd))
    atoms: list[Atom] = []

    if name == "Bash":
        cmd = str(inp.get("command", ""))
        files += [_norm_path(p, cwd) for p in find_paths(cmd)]
        status_txt = f"exit {status}" if status != "interrupted" else "interrupted"
        full = f"$ {cmd}\n[{status_txt}]\n{out_text}" + (f"\n{err_text}" if err_text else "")
        tail = [ln for ln in out_text.strip().splitlines() if ln.strip()][-1:] if status == 0 else []
        render = f"$ {_clip(cmd, 160)} -> {status_txt}" + (f" | {_clip(tail[0], 100)}" if tail else "")
        atoms.append(
            Atom(aid, "command", ev.turn, ev.seq, full, render, "tool", meta={"exit": status, "tool": name},
                 symbols=find_symbols(cmd))
        )
    elif name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        body, removed, added = _edit_text(name, inp)
        path = files[0] if files else "?"
        first_new = next((ln[2:].strip() for ln in body.splitlines() if ln.startswith("+ ") and ln[2:].strip()), "")
        verb = "wrote" if name == "Write" else "edited"
        render = f"{verb} {path} (-{removed} +{added} lines)" + (f": {_clip(first_new, 90)}" if first_new else "")
        full = f"{name} {path}\n{body}" + (f"\n[error] {err_text}" if err_text else "")
        atoms.append(
            Atom(aid, "edit", ev.turn, ev.seq, full, render, "tool", meta={"tool": name, "exit": status},
                 symbols=find_symbols(body)[:40])
        )
    elif name in ("Read", "Grep", "Glob", "LS"):
        detail = inp.get("pattern") or (files[0] if files else "")
        n_lines = out_text.count("\n") + 1 if out_text else 0
        render = f"{name} {_clip(str(detail), 120)} ({n_lines} lines)"
        if name in ("Grep", "Glob"):
            for p in find_paths(out_text)[:20]:
                files.append(_norm_path(p, cwd))
        sym_src = str(inp.get("pattern", ""))
        atoms.append(
            Atom(aid, "lookup", ev.turn, ev.seq, f"{name} {json.dumps(inp)}\n{out_text}", render, "tool",
                 meta={"tool": name, "exit": status}, symbols=find_symbols(sym_src))
        )
    else:
        args = _clip(json.dumps(inp, ensure_ascii=False), 160)
        render = f"{name} {args}"
        atoms.append(
            Atom(aid, "tool", ev.turn, ev.seq, f"{name} {json.dumps(inp)}\n{out_text}", render, "tool",
                 meta={"tool": name, "exit": status})
        )

    looks_failed = status not in (0,) or bool(err_text)
    errs = error_lines(err_text or out_text) if (looks_failed or name == "Bash") else []
    if looks_failed and not errs:
        errs = [_clip(err_text or out_text or "failed", 200)]
    if errs and (looks_failed or any("Traceback" in e or "FAILED" in e or "Error" in e for e in errs)):
        src = err_text or out_text
        err_files = [_norm_path(m.group(1), cwd) for m in _TRACEBACK_FILE.finditer(src)]
        err_files += [_norm_path(p, cwd) for p in find_paths("\n".join(errs))]
        err_files = [p for p in dict.fromkeys(err_files) if "site-packages" not in p and not p.startswith("<")]
        head = atoms[0].render.split(" -> ")[0] if name == "Bash" else f"{name} {files[0] if files else ''}"
        text = "\n".join(errs)
        atoms.append(
            Atom(f"e{ev.seq}", "error", ev.turn, ev.seq, text, f"error in {_clip(head, 100)}: "
                 + " / ".join(_clip(e, 160) for e in errs[-4:]), "tool", files=err_files,
                 symbols=find_symbols(text), parent=aid, meta={"exit": status})
        )
    for a in atoms:
        if a.kind != "error":
            a.files = list(dict.fromkeys(files))
    return atoms


# ---------------------------------------------------------------- main entry


def extract_atoms(events: Iterable[Event], cwd: str = "", chars_per_token: float = 3.5) -> list[Atom]:
    atoms: list[Atom] = []
    for ev in events:
        if ev.kind == PROMPT:
            text = str(ev.data.get("text", ""))
            shown = text.strip()
            if len(shown) > _MAX_TURN_CHARS:
                shown = shown[:_MAX_TURN_CHARS] + f" ... [truncated; full text: ctx get u{ev.turn}]"
            u = Atom(f"u{ev.turn}", "turn", ev.turn, ev.seq, text, f'user: "{shown}"', "user",
                     files=[_norm_path(p, cwd) for p in find_paths(text)], symbols=find_symbols(text))
            atoms.append(u)
            n_k = n_d = 0
            for s in split_sentences(text):
                if is_constraint(s):
                    n_k += 1
                    atoms.append(Atom(f"k{ev.turn}.{n_k}", "constraint", ev.turn, ev.seq, s, s, "user",
                                      files=[_norm_path(p, cwd) for p in find_paths(s)],
                                      symbols=find_symbols(s), pinned=True, parent=u.id))
                elif _DECISION_USER.search(s) and not s.endswith("?"):
                    n_d += 1
                    atoms.append(Atom(f"d{ev.turn}.{n_d}", "decision", ev.turn, ev.seq, s, s, "user",
                                      files=[_norm_path(p, cwd) for p in find_paths(s)],
                                      symbols=find_symbols(s), pinned=True, parent=u.id))
        elif ev.kind == REPLY:
            text = str(ev.data.get("text", ""))
            r = Atom(f"r{ev.turn}", "reply", ev.turn, ev.seq, text, "assistant: " + _clip(text, 600), "assistant",
                     files=[_norm_path(p, cwd) for p in find_paths(text)], symbols=find_symbols(text)[:40])
            atoms.append(r)
            cands = [s for s in split_sentences(text) if _DECISION_ASSISTANT.search(s) and not s.endswith("?")]
            cands.sort(key=lambda s: 0 if _STRONG_DECISION.search(s) else 1)
            for n, s in enumerate(cands[:_MAX_ASSISTANT_DECISIONS], start=1):
                atoms.append(Atom(f"d{ev.turn}.a{n}", "decision", ev.turn, ev.seq, s, s, "assistant",
                                  files=[_norm_path(p, cwd) for p in find_paths(s)],
                                  symbols=find_symbols(s), pinned=True, parent=r.id))
        elif ev.kind == TOOL:
            atoms.extend(_tool_atoms(ev, cwd))

    _resolve_paths(atoms)
    atoms.extend(_file_atoms(atoms))
    _dedupe_pins(atoms)
    for a in atoms:
        a.last_used = a.turn
        a.tokens = estimate(a.render, chars_per_token) + 4  # +4 for the id and bullet
    return atoms


def _resolve_paths(atoms: list[Atom]) -> None:
    """Map bare mentions such as ``ledger.py`` onto a full path touched by a tool."""
    known = {p for a in atoms if a.source == "tool" for p in a.files}
    by_base: dict[str, set[str]] = {}
    for p in known:
        by_base.setdefault(os.path.basename(p), set()).add(p)
    for a in atoms:
        out = []
        for p in a.files:
            if p not in known:
                hits = [k for k in known if k.endswith("/" + p.lstrip("./"))] or list(by_base.get(os.path.basename(p), ()))
                if len(hits) == 1:
                    p = hits[0]
            if p not in out:
                out.append(p)
        a.files = out


def _file_atoms(atoms: list[Atom]) -> list[Atom]:
    uses: dict[str, list[tuple[int, str, int]]] = {}
    for a in atoms:
        if a.kind == "file":
            continue
        for p in a.files:
            uses.setdefault(p, []).append((a.turn, a.kind, a.seq))
    out = []
    for path, us in uses.items():
        touched = [u for u in us if u[1] in ("edit", "command", "lookup", "error")]
        if not touched:
            continue  # only mentioned in prose, never touched: not a file atom
        parts = []
        for kind, label in (("edit", "edited"), ("lookup", "read"), ("command", "used in commands"), ("error", "in errors")):
            turns = sorted({u[0] for u in us if u[1] == kind})
            if turns:
                parts.append(f"{label} (turns {', '.join(map(str, turns[-6:]))})")
        first = min(u[0] for u in us)
        seq = min(u[2] for u in us)
        render = f"file {path}: " + "; ".join(parts)
        out.append(Atom(f"f:{path}", "file", first, seq, render, render, "tool", files=[path],
                        meta={"last_touch": max(u[0] for u in touched)}))
    return out


def _dedupe_pins(atoms: list[Atom]) -> None:
    """When the same rule is stated twice, keep only the latest statement pinned."""
    latest: dict[str, Atom] = {}
    for a in atoms:
        if a.pinned:
            key = re.sub(r"\W+", " ", a.text.lower()).strip()
            if key in latest:
                latest[key].pinned = False
                latest[key].meta["duplicate_of"] = a.id
            latest[key] = a
