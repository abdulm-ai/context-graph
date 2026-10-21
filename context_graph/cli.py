"""ctx: inspect the session graph.

  ctx show              what would be kept right now, with scores
  ctx explain <id>      why an atom scores what it does: its chain from the focus
  ctx get <id>          the full verbatim text of an atom (trimmed outputs included)
  ctx pack              the pack that /clear would inject
  ctx log               the last selection log written by a hook
  ctx stats             counts of events, atoms and edges
  ctx import <jsonl>    build a thread from an existing transcript
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

from . import config as cfgmod
from .atoms import Atom
from .engine import build_pack, thread_atoms
from .events import PROMPT
from .graph import Graph
from .relevance import dijkstra, path_to
from .select import focus_sources
from .store import Store, Thread
from .transcript import parse_transcript


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _thread(args: argparse.Namespace) -> Thread:
    t = Store(args.home).find(args.cwd, args.session)
    if t is None:
        sys.exit(f"ctx: no context-graph data for {args.cwd}" + (f" session {args.session}" if args.session else ""))
    return t


def _cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = cfgmod.load(Store(args.home).home, args.cwd)
    for key in ("relevance", "solver", "selector"):
        if getattr(args, key, None):
            cfg[key] = getattr(args, key)
    return cfg


def cmd_show(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    atoms = thread_atoms(_thread(args), cfg)
    if not atoms:
        print("no atoms yet")
        return 0
    text, sel = build_pack(atoms, cfg, args.mode, budget=args.budget)
    by_id = {a.id: a for a in atoms}
    print(f"{sel.n_atoms} atoms, {sel.n_edges} edges, turn {sel.now_turn}. "
          f"selector={sel.selector} relevance={sel.relevance} solver={sel.solver} mode={args.mode}")
    print(f"budget {sel.budget} tokens: pinned {sel.pinned_tokens}, chosen {sel.chosen_tokens}, "
          f"pack ~{len(text) / float(cfg['chars_per_token']):.0f}. focus: "
          + ", ".join(f"{k} ({v:g})" for k, v in sel.focus.items()))
    print()
    hdr = f"{'':4}{'id':<22}{'kind':<11}{'turn':>4}{'tok':>6}{'rel':>8}{'decay':>7}{'value':>8}  text"
    print(hdr)

    def line(aid: str, mark: str) -> str:
        s, a = sel.scores[aid], by_id[aid]
        rel = s.get("relevance")
        num = (f"{rel:>8.3f}{s['decay']:>7.2f}{s['value']:>8.4f}" if rel is not None else f"{'':>8}{'':>7}{'':>8}")
        return f"{mark:<4}{_clip(aid, 21):<22}{a.kind:<11}{a.turn:>4}{a.tokens:>6}{num}  {_clip(a.render, 70)}"

    kept = sorted(sel.kept, key=lambda i: (not sel.scores[i].get("pinned"), -float(sel.scores[i].get("value", 0) or 0)))
    for aid in kept:
        print(line(aid, "PIN" if sel.scores[aid].get("pinned") else "+"))
    dropped = sorted(sel.dropped, key=lambda i: -float(sel.scores[i].get("value", 0) or 0))
    shown = dropped if args.all else dropped[:10]
    if shown:
        print(f"\ndropped: {len(dropped)}" + ("" if args.all else " (top 10 by value; --all for every one)"))
        for aid in shown:
            print(line(aid, "-"))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    atoms = thread_atoms(_thread(args), cfg)
    by_id = {a.id: a for a in atoms}
    a = by_id.get(args.id)
    if a is None:
        sys.exit(f"ctx: no atom {args.id}")
    text, sel = build_pack(atoms, cfg, args.mode)
    g = Graph(atoms)
    s = sel.scores[a.id]
    print(f"{a.id}  {a.kind}  turn {a.turn}  {a.tokens} tokens  source={a.source}"
          + ("  PINNED" if a.pinned else ""))
    print(f"render: {a.render[:400]}")
    if a.files:
        print("files: " + ", ".join(a.files))
    if a.symbols:
        print("symbols: " + ", ".join(a.symbols[:12]))
    print()
    if "relevance" in s:
        print(f"relevance {s['relevance']:.4f} ({sel.relevance})  x  decay {s['decay']:.3f} "
              f"(last used turn {a.last_used}, now {sel.now_turn}, half-life {cfg['half_life_turns']:g} turns)"
              f"  =  value {s['value']:.5f}")
    sources = focus_sources(atoms, int(cfg["focus_turns"]))
    dist, prev = dijkstra(g, sources)
    path = path_to(prev, a.id)
    if path:
        print(f"\nmost probable chain from the focus (Dijkstra, w = -log p):")
        start = path[0]
        print(f"  {start:<14} focus weight {sources[start]:g}  -> p = {sources[start]:.3f}")
        cum = sources[start]
        for x, y in zip(path, path[1:]):
            e = g.adj[x][y]
            cum *= e.p
            print(f"  {y:<14} p(edge) = {e.p:.3f} [{'; '.join(e.reasons)}]  -> p = {cum:.4f}"
                  f"  surprisal {-math.log(cum):.2f}")
    else:
        print("\nno path from the focus: relevance comes only from pinning, if any")
    nbrs = sorted(g.adj[a.id].items(), key=lambda kv: -kv[1].p)
    if nbrs:
        print(f"\nedges ({len(nbrs)}):")
        for nid, e in nbrs[:12]:
            print(f"  {nid:<14} p = {e.p:.3f}  {'; '.join(e.reasons)}")
    print()
    if s.get("pinned"):
        print("decision: KEPT (pinned: always carried over, verbatim)")
    elif a.id in sel.kept:
        print(f"decision: KEPT by the {sel.solver} solver")
    else:
        kept_dens = [sel.scores[i]["value"] / max(1, by_id[i].tokens) for i in sel.kept
                     if not sel.scores[i].get("pinned") and sel.scores[i].get("value")]
        dens = s.get("value", 0) / max(1, a.tokens)
        floor = min(kept_dens) if kept_dens else 0
        print(f"decision: DROPPED by the {sel.solver} solver. value per token {dens:.2e}; "
              f"lowest kept {floor:.2e}; budget {sel.budget}, pinned {sel.pinned_tokens}, chosen {sel.chosen_tokens}")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    t = _thread(args)
    atoms = thread_atoms(t, cfg)
    by_id = {a.id: a for a in atoms}
    a = by_id.get(args.id)
    if a is None:
        sys.exit(f"ctx: no atom {args.id}")
    lines = a.text.splitlines()
    if args.lines:
        lo, _, hi = args.lines.partition(":")
        lines = lines[int(lo or 1) - 1: int(hi) if hi else None]
    print("\n".join(lines))
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    atoms = thread_atoms(_thread(args), cfg)
    text, _ = build_pack(atoms, cfg, args.mode, budget=args.budget)
    print(text)
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    sels = _thread(args).selections()
    if not sels:
        print("no selections logged yet")
        return 0
    with open(sels[-1], encoding="utf-8") as fh:
        rec = json.load(fh)
    if args.json:
        print(json.dumps(rec, indent=1))
        return 0
    print(f"{sels[-1]}\n{rec['time']} trigger={rec['trigger']} mode={rec['mode']} {rec['config']}")
    print(f"budget {rec['budget']}, pack ~{rec['pack_tokens_est']} tokens ({rec['pack_chars']} chars), "
          f"{len(rec['kept'])} kept / {len(rec['dropped'])} dropped, {rec['elapsed_ms']} ms")
    for r in rec["kept"]:
        print(f"  + {r['id']:<14}{r.get('value', '')!s:>10}  {_clip(r['render'], 80)}")
    for r in rec["dropped"][:10]:
        print(f"  - {r['id']:<14}{r.get('value', '')!s:>10}  {_clip(r['render'], 80)}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    t = _thread(args)
    evs = t.events()
    atoms = thread_atoms(t, cfg)
    g = Graph(atoms)
    kinds: dict[str, int] = {}
    for a in atoms:
        kinds[a.kind] = kinds.get(a.kind, 0) + 1
    meta = t.meta()
    print(f"thread {t.id}  sessions {len(meta.get('sessions', []))}  turns {meta.get('turn', 0)}  events {len(evs)}")
    print(f"atoms {len(atoms)}: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print(f"edges {g.n_edges}; full text held {sum(len(a.text) for a in atoms)} chars")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    tr = parse_transcript(args.transcript)
    sid = args.session or (tr.session_ids[0] if tr.session_ids else os.path.basename(args.transcript).split(".")[0])
    store = Store(args.home)
    t = store.new_thread(args.cwd, sid)
    if t.events():
        sys.exit(f"ctx: thread {t.id} already has events")
    turn = 0
    for ev in tr.events:
        t.append(ev.kind, ev.data, ev.session_id, new_turn=ev.kind == PROMPT)
        turn = max(turn, ev.turn)
    print(f"imported {len(tr.events)} events, {turn} turns, {len(tr.compactions)} compactions into thread {t.id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ctx", description="Inspect the context-graph of a Claude Code session.")
    p.add_argument("--cwd", default=os.getcwd(), help="project directory (default: current)")
    p.add_argument("--session", help="session or thread id (default: most recent in this project)")
    p.add_argument("--home", help="store directory (default: $CONTEXT_GRAPH_HOME or ~/.context-graph)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def sel_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--mode", choices=("replace", "assist"), default="replace")
        sp.add_argument("--relevance", choices=("dijkstra", "diffusion"))
        sp.add_argument("--solver", choices=("knapsack", "greedy"))
        sp.add_argument("--selector", choices=("graph", "recency"))
        sp.add_argument("--budget", type=int)

    sp = sub.add_parser("show", help="current selection with scores")
    sel_opts(sp)
    sp.add_argument("--all", action="store_true", help="list every dropped atom")
    sp.set_defaults(fn=cmd_show)
    sp = sub.add_parser("explain", help="why an atom is kept or dropped")
    sp.add_argument("id")
    sel_opts(sp)
    sp.set_defaults(fn=cmd_explain)
    sp = sub.add_parser("get", help="full text of an atom")
    sp.add_argument("id")
    sp.add_argument("--lines", help="line range, e.g. 100:200")
    sp.set_defaults(fn=cmd_get)
    sp = sub.add_parser("pack", help="print the pack /clear would inject")
    sel_opts(sp)
    sp.set_defaults(fn=cmd_pack)
    sp = sub.add_parser("log", help="last selection log")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_log)
    sp = sub.add_parser("stats", help="counts")
    sp.set_defaults(fn=cmd_stats)
    sp = sub.add_parser("import", help="import a transcript JSONL")
    sp.add_argument("transcript")
    sp.set_defaults(fn=cmd_import)
    args = p.parse_args(argv)
    args.cwd = os.path.abspath(args.cwd)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
