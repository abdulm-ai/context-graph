"""Parse a Claude Code transcript (JSONL) into events and compaction records.

The transcript is the fallback source: hooks capture the same data live, but a
transcript lets us import a session that ran without the plugin, and it is the
only place the native compaction metadata is written.

Schema notes (checked against Claude Code 2.1.x transcripts):
  * user records carry ``message.content`` as a string (a typed prompt) or a
    list of blocks (``text``, ``tool_result``). Tool results also carry the
    structured ``toolUseResult``, the same object hooks see as tool_response.
  * assistant records hold one content block each (``text``, ``tool_use``,
    ``thinking``); blocks of one API message share ``message.id``.
  * a ``type: system`` record with ``subtype: compact_boundary`` has
    ``compactMetadata`` (trigger, preTokens, postTokens, durationMs); the next
    record with ``isCompactSummary: true`` holds the summary text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .events import PROMPT, REPLY, TOOL, Event, tool_output_text

_COMMAND_PREFIXES = (
    "<command-name>",
    "<command-message>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<local-command-caveat>",
)


@dataclass
class Compaction:
    trigger: str
    pre_tokens: int | None
    post_tokens: int | None
    duration_ms: int | None
    summary: str = ""
    timestamp: str = ""


@dataclass
class Transcript:
    events: list[Event] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)


def _is_prompt_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(_COMMAND_PREFIXES):
        return False
    if stripped.startswith("<system-reminder>") and stripped.endswith("</system-reminder>"):
        return False
    return True


def _text_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    out = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(str(block.get("text", "")))
    return out


def read_records(lines: Iterable[str]) -> list[dict[str, Any]]:
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a partially written last line is normal while a session runs
        if isinstance(rec, dict):
            records.append(rec)
    return records


def parse_records(records: list[dict[str, Any]]) -> Transcript:
    tr = Transcript()
    seq = 0
    turn = 0
    pending_tools: dict[str, dict[str, Any]] = {}
    reply_text = ""
    reply_msg_id = None
    reply_session = ""
    reply_ts = ""
    waiting_summary: Compaction | None = None

    def emit(kind: str, data: dict[str, Any], rec: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        tr.events.append(
            Event(
                seq=seq,
                turn=turn,
                kind=kind,
                data=data,
                session_id=rec.get("sessionId", ""),
                ts=rec.get("timestamp", ""),
            )
        )

    def flush_reply() -> None:
        nonlocal reply_text, reply_msg_id
        if reply_text.strip():
            emit(REPLY, {"text": reply_text.strip()}, {"sessionId": reply_session, "timestamp": reply_ts})
        reply_text = ""
        reply_msg_id = None

    for rec in records:
        sid = rec.get("sessionId")
        if sid and sid not in tr.session_ids:
            tr.session_ids.append(sid)
        if rec.get("isSidechain"):
            continue
        rtype = rec.get("type")

        if rtype == "system" and rec.get("subtype") == "compact_boundary":
            meta = rec.get("compactMetadata") or {}
            waiting_summary = Compaction(
                trigger=str(meta.get("trigger", "")),
                pre_tokens=meta.get("preTokens"),
                post_tokens=meta.get("postTokens"),
                duration_ms=meta.get("durationMs"),
                timestamp=rec.get("timestamp", ""),
            )
            tr.compactions.append(waiting_summary)
            continue

        if rtype == "user":
            msg = rec.get("message") or {}
            content = msg.get("content")
            if rec.get("isCompactSummary"):
                if waiting_summary is not None:
                    waiting_summary.summary = "\n".join(_text_blocks(content))
                    waiting_summary = None
                continue
            if rec.get("isMeta"):
                continue
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    use = pending_tools.pop(block.get("tool_use_id", ""), None)
                    if use is None:
                        continue
                    is_error = bool(block.get("is_error"))
                    response = rec.get("toolUseResult")
                    if is_error:
                        # As in the PostToolUseFailure hook: no response, an error string.
                        data = {"name": use["name"], "input": use["input"],
                                "response": response if isinstance(response, dict) else None,
                                "error": tool_output_text(block.get("content"))}
                    else:
                        if response is None:
                            response = block.get("content")
                        data = {"name": use["name"], "input": use["input"], "response": response}
                    emit(TOOL, data, rec)
            texts = [t for t in _text_blocks(content) if _is_prompt_text(t)]
            if texts:
                flush_reply()
                turn += 1
                emit(PROMPT, {"text": "\n".join(texts)}, rec)
            continue

        if rtype == "assistant":
            msg = rec.get("message") or {}
            content = msg.get("content")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    pending_tools[block.get("id", "")] = {
                        "name": block.get("name", ""),
                        "input": block.get("input") or {},
                    }
                elif block.get("type") == "text" and str(block.get("text", "")).strip():
                    # The Stop hook reports only the final assistant message, so
                    # keep the text of the latest message id, as it does.
                    if msg.get("id") is not None and msg.get("id") == reply_msg_id:
                        reply_text += "\n" + block["text"]
                    else:
                        reply_text = block["text"]
                        reply_msg_id = msg.get("id")
                    reply_session = rec.get("sessionId", "")
                    reply_ts = rec.get("timestamp", "")
            continue

    flush_reply()
    return tr


def parse_transcript(path: str) -> Transcript:
    with open(path, encoding="utf-8") as fh:
        return parse_records(read_records(fh))
