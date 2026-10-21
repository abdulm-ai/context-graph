"""Run the context-graph benchmark.

One run = one scripted base session on a fresh copy of bench/demo_repo, then
every compaction method applied to a copy of that same session:

  full          no compaction (the ceiling)
  native        /compact
  native_instr  /compact with instructions to keep constraints and facts
  assist        /compact plus the context-graph pins (Assist mode)
  recency       /clear plus the last stretch of the session, same budget
  cg_dijkstra   /clear plus the context-graph pack: shortest path, knapsack
  cg_diffusion  /clear plus the pack: heat diffusion, knapsack
  cg_greedy     /clear plus the pack: shortest path, greedy
  cg_2k         cg_dijkstra with a 2,000 token budget instead of 6,000

After each method: one probe call (tools off) with the twelve planted-fact
questions, and one follow-up coding task on a reset repository that a
forgotten constraint would break. Probe and follow-up each fork the
post-compaction session, so neither sees the other.

Variant A runs the base session with tool-output trimming on, variant B with
it off (the trimming ablation); B runs a subset of methods.

Usage:
  python3 bench/run_bench.py --runs A1 A2 A3 B1 B2 B3 --parallel 3
  python3 bench/run_bench.py --runs P1 --turns 6 --methods native cg_dijkstra   # quick pilot
  python3 bench/run_bench.py --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import checks  # noqa: E402
import claude_cli as cc  # noqa: E402
import scenario as sc  # noqa: E402
from context_graph.transcript import parse_transcript  # noqa: E402

COMPACT_INSTRUCTIONS = ("Preserve every constraint, rule, decision, rejected approach, root cause and stated "
                        "fact from the user word for word, including names, hosts, ports, ticket numbers and dates.")
BUDGET = 6000

METHODS: dict[str, dict] = {
    "full": {"kind": "none"},
    "native": {"kind": "compact", "env": {"CONTEXT_GRAPH_INJECT_ON_COMPACT": "0"}},
    "native_instr": {"kind": "compact", "instructions": COMPACT_INSTRUCTIONS,
                     "env": {"CONTEXT_GRAPH_INJECT_ON_COMPACT": "0"}},
    "assist": {"kind": "compact", "env": {"CONTEXT_GRAPH_INJECT_ON_COMPACT": "1"}},
    "recency": {"kind": "clear", "env": {"CONTEXT_GRAPH_SELECTOR": "recency"}},
    "cg_dijkstra": {"kind": "clear", "env": {"CONTEXT_GRAPH_RELEVANCE": "dijkstra", "CONTEXT_GRAPH_SOLVER": "knapsack"}},
    "cg_diffusion": {"kind": "clear", "env": {"CONTEXT_GRAPH_RELEVANCE": "diffusion", "CONTEXT_GRAPH_SOLVER": "knapsack"}},
    "cg_greedy": {"kind": "clear", "env": {"CONTEXT_GRAPH_RELEVANCE": "dijkstra", "CONTEXT_GRAPH_SOLVER": "greedy"}},
    "cg_2k": {"kind": "clear", "budget": 2000,
              "env": {"CONTEXT_GRAPH_RELEVANCE": "dijkstra", "CONTEXT_GRAPH_SOLVER": "knapsack"}},
}
METHODS_A = list(METHODS)
METHODS_B = ["full", "native", "cg_dijkstra"]

_print_lock = threading.Lock()
_spent_lock = threading.Lock()
_spent = {"usd": 0.0}


def log(run_id: str, msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {run_id}: {msg}", flush=True)


class BudgetExceeded(RuntimeError):
    pass


def account(call: cc.Call, cap: float) -> cc.Call:
    with _spent_lock:
        _spent["usd"] += call.cost_usd
        if _spent["usd"] > cap:
            raise BudgetExceeded(f"total spend ${_spent['usd']:.2f} passed the cap ${cap:.2f}")
    return call


def git(repo: str, *args: str) -> str:
    out = subprocess.run(["git", "-c", "user.name=bench", "-c", "user.email=bench@localhost", *args], cwd=repo,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def reset_repo(repo: str, rev: str) -> None:
    git(repo, "reset", "-q", "--hard", rev)
    git(repo, "clean", "-qfdx")


def compaction_record(cfg_dir: str, session_id: str) -> dict:
    path = cc.transcript_path(cfg_dir, session_id)
    if not path:
        return {}
    tr = parse_transcript(path)
    if not tr.compactions:
        return {}
    c = tr.compactions[-1]
    return {"trigger": c.trigger, "pre_tokens": c.pre_tokens, "post_tokens": c.post_tokens,
            "duration_ms": c.duration_ms, "summary_chars": len(c.summary), "summary": c.summary}


def latest_selection(store: str) -> dict:
    newest, rec = 0.0, {}
    for dirpath, _, files in os.walk(store):
        if os.path.basename(dirpath) != "selections":
            continue
        for f in files:
            p = os.path.join(dirpath, f)
            if os.path.getmtime(p) >= newest:
                newest = os.path.getmtime(p)
                with open(p, encoding="utf-8") as fh:
                    rec = json.load(fh)
    return rec


class Run:
    def __init__(self, run_id: str, args: argparse.Namespace):
        self.id = run_id
        self.args = args
        self.variant = run_id[0].upper()
        self.trim = self.variant != "B"
        self.work = os.path.join(args.work, run_id)
        self.repo = os.path.join(self.work, "repo")
        self.out = os.path.join(args.out, run_id)
        self.n_turns = args.turns
        self.methods = args.methods or (METHODS_B if self.variant == "B" else METHODS_A)

    # -------------------------------------------------------------- helpers
    def env(self, cfg_dir: str, home: str, store: str, extra: dict | None = None) -> dict:
        e = {"CONTEXT_GRAPH_HOME": store, "CONTEXT_GRAPH_TRIM": "1" if self.trim else "0",
             "CONTEXT_GRAPH_BUDGET_REPLACE": str(BUDGET)}
        e.update(extra or {})
        return cc.child_env(cfg_dir, home, e)

    def call(self, prompt: str, env: dict, **kw) -> cc.Call:
        kw.setdefault("model", self.args.model)
        kw.setdefault("plugin_dir", ROOT)
        kw.setdefault("max_budget_usd", self.args.call_budget)
        c = cc.run(prompt, self.repo, env, **kw)
        if not c.ok and c.cost_usd == 0:
            time.sleep(10)  # transient API or startup error: one retry
            c = cc.run(prompt, self.repo, env, **kw)
        return account(c, self.args.total_budget)

    def save(self, name: str, data: dict) -> None:
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, ensure_ascii=False)

    # -------------------------------------------------------------- base session
    def base_session(self) -> dict:
        shutil.rmtree(self.work, ignore_errors=True)
        shutil.copytree(os.path.join(HERE, "demo_repo"), self.repo)
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "start")
        base = os.path.join(self.work, "base")
        env = self.env(os.path.join(base, "claude"), os.path.join(base, "home"), os.path.join(base, "store"))
        calls, sid = [], None
        for turn in range(1, self.n_turns + 1):
            c = self.call(sc.turn_prompt(turn), env, resume=sid)
            sid = sid or c.session_id
            calls.append({"turn": turn, **c.to_json()})
            log(self.id, f"turn {turn}/{self.n_turns} ok={c.ok} in={c.input_tokens} ${c.cost_usd:.4f} {c.wall_s}s")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "--allow-empty", "-m", "base session end")
        rev = git(self.repo, "rev-parse", "HEAD")
        data = {"run": self.id, "variant": self.variant, "trim": self.trim, "session_id": sid, "base_rev": rev,
                "n_turns": self.n_turns, "calls": calls,
                "input_tokens": sum(c["input_tokens"] for c in calls),
                "output_tokens": sum(c["output_tokens"] for c in calls),
                "cost_usd": round(sum(c["cost_usd"] for c in calls), 4),
                "wall_s": round(sum(c["wall_s"] for c in calls), 1)}
        self.save("base", data)
        return data

    def baseline_probe(self, base: dict) -> dict:
        """The probe in a fresh, empty session: its input tokens are the fixed overhead."""
        d = os.path.join(self.work, "baseline")
        env = self.env(os.path.join(d, "claude"), os.path.join(d, "home"), os.path.join(d, "store"),
                       {"CONTEXT_GRAPH_DISABLE": "1"})
        c = self.call(sc.probe_prompt(), env, tools="")
        data = {"input_tokens": c.input_tokens, "call": c.to_json()}
        self.save("baseline_probe", data)
        return data

    # -------------------------------------------------------------- one method
    def method(self, name: str, base: dict, baseline: dict) -> dict:
        spec = METHODS[name]
        mdir = os.path.join(self.work, name)
        shutil.rmtree(mdir, ignore_errors=True)
        src = os.path.join(self.work, "base")
        shutil.copytree(os.path.join(src, "claude"), os.path.join(mdir, "claude"))
        shutil.copytree(os.path.join(src, "store"), os.path.join(mdir, "store"))
        cfg_dir, home, store = (os.path.join(mdir, x) for x in ("claude", "home", "store"))
        extra = dict(spec.get("env", {}))
        if spec.get("budget"):
            extra["CONTEXT_GRAPH_BUDGET_REPLACE"] = str(spec["budget"])
        env = self.env(cfg_dir, home, store, extra)
        reset_repo(self.repo, base["base_rev"])
        sid = base["session_id"]
        rec: dict = {"run": self.id, "method": name, "variant": self.variant, "spec": spec}

        if spec["kind"] == "compact":
            prompt = "/compact" + (f" {spec['instructions']}" if spec.get("instructions") else "")
            c = self.call(prompt, env, resume=sid)
            rec["compaction_call"] = c.to_json()
            rec["compaction"] = compaction_record(cfg_dir, sid)
            post = sid
        elif spec["kind"] == "clear":
            c = self.call("/clear", env, resume=sid)
            rec["compaction_call"] = c.to_json()
            post = c.session_id
        else:
            post = sid
        rec["post_session"] = post
        sel = latest_selection(store)
        if sel and spec["kind"] != "none":
            rec["selection"] = {k: sel.get(k) for k in ("trigger", "mode", "config", "budget", "pack_chars",
                                                         "pack_tokens_est", "n_atoms", "n_edges", "pinned_tokens",
                                                         "overflow", "elapsed_ms", "timings_ms")}
            rec["selection"]["kept"] = [r["id"] for r in sel.get("kept", [])]
            rec["pack"] = sel.get("pack", "")

        quiet = dict(env, CONTEXT_GRAPH_DISABLE="1")
        probe = self.call(sc.probe_prompt(), quiet, resume=post, tools="", extra_args=["--fork-session"])
        scores = sc.score_probe(probe.result)
        rec["probe"] = {"call": probe.to_json(), "answer": probe.result, "scores": scores,
                        "accuracy": sum(scores.values()) / len(scores),
                        "carried_tokens": probe.input_tokens - baseline["input_tokens"]}
        log(self.id, f"{name}: probe {sum(scores.values())}/{len(scores)} carried={rec['probe']['carried_tokens']}")

        if not self.args.no_followup:
            reset_repo(self.repo, base["base_rev"])
            fu = self.call(sc.FOLLOWUP, quiet, resume=post, extra_args=["--fork-session"],
                           max_budget_usd=self.args.call_budget * 2)
            rec["followup"] = {"call": fu.to_json(), "checks": checks.run_checks(self.repo, base["base_rev"])}
            log(self.id, f"{name}: follow-up violations={rec['followup']['checks']['n_violations']} "
                         f"tests_pass={rec['followup']['checks']['tests_pass']}")
            reset_repo(self.repo, base["base_rev"])
        self.save(name, rec)
        return rec

    def go(self) -> None:
        try:
            base_path = os.path.join(self.out, "base.json")
            if self.args.reuse_base and os.path.exists(base_path) and os.path.isdir(self.repo):
                with open(base_path, encoding="utf-8") as fh:
                    base = json.load(fh)
                log(self.id, "reusing base session")
            else:
                base = self.base_session()
            baseline = self.baseline_probe(base)
            for name in self.methods:
                self.method(name, base, baseline)
            log(self.id, "done")
        except BudgetExceeded as exc:
            log(self.id, f"stopped: {exc}")
        except Exception:
            log(self.id, "failed:\n" + traceback.format_exc())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["A1"])
    ap.add_argument("--methods", nargs="+", choices=list(METHODS))
    ap.add_argument("--turns", type=int, default=sc.N_TURNS)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--call-budget", type=float, default=1.0, help="--max-budget-usd for each claude call")
    ap.add_argument("--total-budget", type=float, default=40.0, help="stop when total spend passes this")
    ap.add_argument("--work", default=os.environ.get("BENCH_WORK", "/tmp/cg-bench"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "raw"))
    ap.add_argument("--no-followup", action="store_true")
    ap.add_argument("--reuse-base", action="store_true", help="reuse a finished base session of the same run id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run or not cc.have_key():
        if not args.dry_run:
            print(f"{cc.KEY_VAR} is not set: dry run only.")
        print(f"runs {args.runs}, {args.turns} turns each, model {args.model}")
        for t in range(1, args.turns + 1):
            print(f"  turn {t:>2}: {sc.turn_prompt(t)}")
        print("methods A:", ", ".join(METHODS_A))
        print("methods B:", ", ".join(METHODS_B))
        print("probe:\n" + sc.probe_prompt())
        print("follow-up:\n" + sc.FOLLOWUP)
        return 0

    meta = {"claude_version": cc.claude_version(), "model_arg": args.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "runs": args.runs, "turns": args.turns, "budget_replace": BUDGET, "methods": {k: METHODS[k] for k in METHODS}}
    os.makedirs(args.out, exist_ok=True)
    runs = [Run(r, args) for r in args.runs]
    with cf.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        list(pool.map(lambda r: r.go(), runs))
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["spent_usd_this_invocation"] = round(_spent["usd"], 4)
    with open(os.path.join(args.out, f"meta-{time.strftime('%Y%m%d-%H%M%S')}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"spent ${_spent['usd']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
