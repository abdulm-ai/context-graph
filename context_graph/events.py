"""The event record shared by the hooks and the transcript parser.

Every source of session data (live hooks, or a transcript JSONL read after the
fact) is reduced to the same three kinds of event, so atom extraction only has
one input format to handle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PROMPT = "prompt"  # a user turn, word for word
TOOL = "tool"  # one tool call with its input and full output
REPLY = "reply"  # the assistant's final message for a turn


@dataclass
class Event:
    seq: int
    turn: int
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    ts: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Event":
        return cls(
            seq=int(d["seq"]),
            turn=int(d["turn"]),
            kind=d["kind"],
            data=d.get("data") or {},
            session_id=d.get("session_id", ""),
            ts=d.get("ts", ""),
        )


def tool_output_text(response: Any) -> str:
    """Flatten a tool_response (any tool's shape) into plain text."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        parts = []
        for item in response:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(tool_output_text(item))
        return "\n".join(p for p in parts if p)
    if isinstance(response, dict):
        if "stdout" in response or "stderr" in response:
            out = str(response.get("stdout") or "")
            err = str(response.get("stderr") or "")
            return out + ("\n" + err if err and out else err)
        if isinstance(response.get("file"), dict):
            return str(response["file"].get("content", ""))
        for key in ("content", "output", "result", "text"):
            if isinstance(response.get(key), (str, list)):
                return tool_output_text(response[key])
        if "filenames" in response and isinstance(response["filenames"], list):
            return "\n".join(map(str, response["filenames"]))
    import json

    return json.dumps(response, ensure_ascii=False)
