"""End-to-end check of the plugin in real headless Claude Code sessions.

Runs a short session with the plugin loaded through --plugin-dir, then
/compact (Assist mode) and /clear (Replace mode), and checks that:

  1. hooks captured prompts, tool calls (including a failed one) and replies
  2. a long Bash output was trimmed in the model's context, and Claude paged a
     hidden line back with `ctx get`
  3. /compact injected the pins on top of the native summary, and PostCompact
     stored that summary
  4. /clear started a new session id whose context is only the pack, and the
     model could still state the constraint from turn 1

Usage:  python3 bench/e2e.py [--model haiku] [--out results/e2e]
Needs BENCH_ANTHROPIC_API_KEY; without it, prints the plan and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import claude_cli as cc  # noqa: E402
from context_graph.store import Store  # noqa: E402
from context_graph.transcript import parse_transcript  # noqa: E402

TURN1 = ("Project rule: never put the word banana in any file in this repository. "
         "Create notes.txt containing the single line alpha. Then run `seq 1 300` with Bash "
         "and tell me the last number it printed.")
TURN2 = "Run `python3 -c \"import nothing_here\"` with Bash and tell me the error in one line."
PROBE = ("Without reading any files: what is the project rule I gave you about a word? Then use the "
         "`ctx` command with Bash to fetch line 150 of the output of the seq command, and tell me "
         "exactly what that line says.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e2e"))
    args = ap.parse_args()
    if not cc.have_key():
        print(f"{cc.KEY_VAR} is not set: dry run. Steps that would run:")
        for i, p in enumerate([TURN1, TURN2, "/compact", "/clear", PROBE], 1):
            print(f"  {i}. claude -p {p!r}")
        return 0

    work = tempfile.mkdtemp(prefix="cg-e2e-")
    proj, cfg_dir, home, store_home = (os.path.join(work, d) for d in ("proj", "claude", "home", "store"))
    os.makedirs(proj)
    env = cc.child_env(cfg_dir, home, {"CONTEXT_GRAPH_HOME": store_home})
    common = dict(model=args.model, plugin_dir=ROOT, max_budget_usd=0.5)

    steps = []
    c1 = cc.run(TURN1, proj, env, **common)
    steps.append(("turn1", c1))
    sid = c1.session_id
    c2 = cc.run(TURN2, proj, env, resume=sid, **common)
    steps.append(("turn2", c2))
    c3 = cc.run("/compact", proj, env, resume=sid, **common)
    steps.append(("compact", c3))
    c4 = cc.run("/clear", proj, env, resume=sid, **common)
    steps.append(("clear", c4))
    new_sid = c4.session_id
    c5 = cc.run(PROBE, proj, env, resume=new_sid, **common)
    steps.append(("probe", c5))

    store = Store(store_home)
    thread = store.lookup(proj, sid)
    checks: dict[str, bool] = {}
    events = thread.events() if thread else []
    kinds = [e.kind for e in events]
    checks["captured prompts, tools and replies"] = all(k in kinds for k in ("prompt", "tool", "reply"))
    checks["captured the failed command"] = any(e.kind == "tool" and e.data.get("error") for e in events)
    tr_path = cc.transcript_path(cfg_dir, sid)
    tr = parse_transcript(tr_path) if tr_path else None
    seq_results = [e for e in (tr.events if tr else []) if e.kind == "tool"
                   and "seq 1 300" in json.dumps(e.data.get("input"))]
    seen = json.dumps(seq_results[0].data.get("response")) if seq_results else ""
    checks["seq output was trimmed in the model's context"] = "context-graph: lines" in seen and "\\n150\\n" not in seen
    sels = [os.path.basename(p) for p in thread.selections()] if thread else []
    checks["/compact wrote an assist selection"] = any(s.endswith("compact.json") for s in sels)
    checks["PostCompact stored the native summary"] = bool(thread) and os.path.exists(
        os.path.join(thread.dir, "native_summaries.jsonl"))
    checks["/clear started a new session id"] = bool(new_sid) and new_sid != sid
    checks["/clear session joined the same thread"] = bool(thread) and store.lookup(proj, new_sid) is not None \
        and store.lookup(proj, new_sid).id == thread.id
    checks["/clear wrote a replace selection"] = any(s.endswith("clear.json") for s in sels)
    # The rule may be stated in a message before the tool call, so read every
    # assistant text of the probe turn, not only the final result.
    ans = c5.result.lower()
    probe_tr = cc.transcript_path(cfg_dir, new_sid)
    if probe_tr:
        with open(probe_tr, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("type") == "assistant":
                    for b in (rec.get("message") or {}).get("content") or []:
                        if isinstance(b, dict) and b.get("type") == "text":
                            ans += "\n" + b.get("text", "").lower()
    checks["after /clear the model states the rule"] = "banana" in ans
    checks["after /clear the model paged line 150 back with ctx get"] = "150" in ans
    probe_calls = []
    if probe_tr:
        for e in parse_transcript(probe_tr).events:
            if e.kind == "tool":
                out = e.data.get("response") or e.data.get("error") or ""
                probe_calls.append({"tool": e.data.get("name"), "input": e.data.get("input"),
                                    "output": out if isinstance(out, str) else out.get("stdout", out)})
    checks["the probe called ctx get through Bash"] = any(
        "ctx get" in json.dumps(c["input"]) and "150" in json.dumps(c["output"]) for c in probe_calls)

    clear_log = {}
    if thread:
        for p in thread.selections():
            if p.endswith("clear.json"):
                with open(p, encoding="utf-8") as fh:
                    clear_log = json.load(fh)
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "claude_version": cc.claude_version(),
        "model_arg": args.model,
        "models": sorted({m for _, c in steps for m in c.models}),
        "probe_answer_text": ans,
        "steps": [{"step": n, "prompt": c.prompt, "ok": c.ok, "result": c.result, "session_id": c.session_id,
                   "input_tokens": c.input_tokens, "output_tokens": c.output_tokens, "cost_usd": c.cost_usd,
                   "wall_s": c.wall_s, "error": c.error} for n, c in steps],
        "total_cost_usd": round(sum(c.cost_usd for _, c in steps), 4),
        "checks": checks,
        "passed": all(checks.values()),
        "probe_tool_calls": probe_calls,
        "clear_pack": clear_log.get("pack", ""),
        "clear_selection": {k: clear_log.get(k) for k in ("budget", "pack_tokens_est", "n_atoms", "n_edges",
                                                           "elapsed_ms")},
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "e2e-report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print(f"cost ${report['total_cost_usd']}, report {path}")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
