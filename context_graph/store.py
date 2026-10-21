"""On-disk store: an append-only event log per thread.

A thread is one line of work in one project directory. It starts with a fresh
session and continues across /compact (same session id) and /clear (new session
id, same thread), so the graph survives both.

    <home>/projects/<project key>/
        sessions.json          session id -> thread id
        last_ended.json        the session that ended most recently, and why
        threads/<thread id>/
            meta.json          cwd, sessions, current turn, next sequence number
            events.jsonl       one Event per line (prompts, tool calls, replies)
            selections/        one JSON log per selection: kept, dropped, scores
            snapshots/         copies of the event log taken at PreCompact
            native_summaries.jsonl   compact_summary from PostCompact

<home> is $CONTEXT_GRAPH_HOME, else ~/.context-graph.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import time
from typing import Any, Iterator

from .events import Event

try:
    import fcntl
except ImportError:  # Windows: no advisory locks; hooks for one session run one at a time anyway
    fcntl = None  # type: ignore[assignment]

CLEAR_LINK_SECONDS = 300


def default_home() -> str:
    return os.environ.get("CONTEXT_GRAPH_HOME") or os.path.join(os.path.expanduser("~"), ".context-graph")


def project_key(cwd: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", os.path.abspath(cwd or ".")).strip("-") or "root"


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write_json(path: str, data: Any) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


@contextlib.contextmanager
def _locked(path: str) -> Iterator[None]:
    with open(path, "a+") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)


class Thread:
    def __init__(self, path: str):
        self.dir = path
        self.id = os.path.basename(path)
        os.makedirs(os.path.join(path, "selections"), exist_ok=True)

    @property
    def meta_path(self) -> str:
        return os.path.join(self.dir, "meta.json")

    @property
    def events_path(self) -> str:
        return os.path.join(self.dir, "events.jsonl")

    def meta(self) -> dict[str, Any]:
        return _read_json(self.meta_path, {})

    def append(self, kind: str, data: dict[str, Any], session_id: str = "", new_turn: bool = False) -> Event:
        with _locked(os.path.join(self.dir, ".lock")):
            meta = self.meta()
            turn = int(meta.get("turn", 0)) + (1 if new_turn else 0)
            seq = int(meta.get("next_seq", 1))
            ev = Event(seq=seq, turn=turn, kind=kind, data=data, session_id=session_id,
                       ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with open(self.events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev.to_json(), ensure_ascii=False) + "\n")
            meta["turn"] = turn
            meta["next_seq"] = seq + 1
            meta["updated"] = time.time()
            _write_json(self.meta_path, meta)
        return ev

    def peek_seq(self) -> int:
        return int(self.meta().get("next_seq", 1))

    def events(self) -> list[Event]:
        out = []
        try:
            with open(self.events_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            out.append(Event.from_json(json.loads(line)))
                        except (ValueError, KeyError):
                            continue
        except OSError:
            pass
        return out

    def write_selection(self, record: dict[str, Any]) -> str:
        sel_dir = os.path.join(self.dir, "selections")
        n = len([f for f in os.listdir(sel_dir) if f.endswith(".json")]) + 1
        path = os.path.join(sel_dir, f"{n:04d}-{record.get('trigger', 'manual')}.json")
        _write_json(path, record)
        return path

    def selections(self) -> list[str]:
        sel_dir = os.path.join(self.dir, "selections")
        return sorted(os.path.join(sel_dir, f) for f in os.listdir(sel_dir) if f.endswith(".json"))

    def snapshot(self, label: str) -> str:
        snap = os.path.join(self.dir, "snapshots", f"{time.strftime('%Y%m%d-%H%M%S')}-{label}")
        os.makedirs(snap, exist_ok=True)
        for name in ("events.jsonl", "meta.json"):
            src = os.path.join(self.dir, name)
            if os.path.exists(src):
                shutil.copy2(src, snap)
        return snap

    def add_native_summary(self, record: dict[str, Any]) -> None:
        with open(os.path.join(self.dir, "native_summaries.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class Store:
    def __init__(self, home: str | None = None):
        self.home = home or default_home()

    def project_dir(self, cwd: str) -> str:
        path = os.path.join(self.home, "projects", project_key(cwd))
        os.makedirs(os.path.join(path, "threads"), exist_ok=True)
        return path

    def _sessions(self, pdir: str) -> dict[str, str]:
        return _read_json(os.path.join(pdir, "sessions.json"), {})

    def _map(self, pdir: str, session_id: str, thread_id: str) -> None:
        with _locked(os.path.join(pdir, ".lock")):
            m = self._sessions(pdir)
            m[session_id] = thread_id
            _write_json(os.path.join(pdir, "sessions.json"), m)

    def threads(self, cwd: str) -> list[Thread]:
        tdir = os.path.join(self.project_dir(cwd), "threads")
        ts = [Thread(os.path.join(tdir, d)) for d in os.listdir(tdir)]
        ts.sort(key=lambda t: t.meta().get("updated", 0), reverse=True)
        return ts

    def lookup(self, cwd: str, session_id: str) -> Thread | None:
        pdir = self.project_dir(cwd)
        tid = self._sessions(pdir).get(session_id)
        if tid and os.path.isdir(os.path.join(pdir, "threads", tid)):
            return Thread(os.path.join(pdir, "threads", tid))
        return None

    def new_thread(self, cwd: str, session_id: str) -> Thread:
        pdir = self.project_dir(cwd)
        t = Thread(os.path.join(pdir, "threads", session_id))
        if not os.path.exists(t.meta_path):
            _write_json(t.meta_path, {"cwd": os.path.abspath(cwd), "sessions": [session_id], "turn": 0,
                                      "next_seq": 1, "created": time.time(), "updated": time.time()})
        self._map(pdir, session_id, t.id)
        return t

    def attach(self, cwd: str, session_id: str, thread: Thread) -> Thread:
        self._map(self.project_dir(cwd), session_id, thread.id)
        with _locked(os.path.join(thread.dir, ".lock")):
            meta = thread.meta()
            if session_id not in meta.setdefault("sessions", []):
                meta["sessions"].append(session_id)
            _write_json(thread.meta_path, meta)
        return thread

    def record_end(self, cwd: str, session_id: str, reason: str) -> None:
        pdir = self.project_dir(cwd)
        tid = self._sessions(pdir).get(session_id)
        _write_json(os.path.join(pdir, "last_ended.json"),
                    {"session_id": session_id, "thread_id": tid, "reason": reason, "ts": time.time()})

    def thread_after_clear(self, cwd: str) -> Thread | None:
        """The thread a /clear continues: the session that just ended with reason 'clear'."""
        pdir = self.project_dir(cwd)
        last = _read_json(os.path.join(pdir, "last_ended.json"), {})
        tid = last.get("thread_id")
        if (last.get("reason") == "clear" and tid and time.time() - float(last.get("ts", 0)) < CLEAR_LINK_SECONDS
                and os.path.isdir(os.path.join(pdir, "threads", tid))):
            return Thread(os.path.join(pdir, "threads", tid))
        ts = self.threads(cwd)
        return ts[0] if ts else None

    def find(self, cwd: str, session_id: str | None = None) -> Thread | None:
        """For the CLI: the given session's thread, else the most recently updated one."""
        if session_id:
            t = self.lookup(cwd, session_id)
            if t:
                return t
            pdir = self.project_dir(cwd)
            if os.path.isdir(os.path.join(pdir, "threads", session_id)):
                return Thread(os.path.join(pdir, "threads", session_id))
            return None
        ts = self.threads(cwd)
        return ts[0] if ts else None
