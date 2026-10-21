"""Turn results/raw into results/summary.json, results/tables.md and the plots.

Every number in the README comes from this script's output. Usage:

  python3 bench/analyze.py                      # from results/raw
  python3 bench/analyze.py --work /tmp/cg-bench # also run the offline budget sweep
                                                # on the saved base-session event logs

matplotlib is a dev-only dependency, used here and nowhere in the plugin.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# CG_CODE_ROOT points the offline sweep at another checkout of the plugin
# (used to compare extractor versions on the same recorded sessions).
sys.path.insert(0, os.environ.get("CG_CODE_ROOT") or ROOT)
sys.path.insert(0, HERE)

import scenario as sc  # noqa: E402

LABELS = {
    "full": "no compaction",
    "native": "/compact",
    "native_instr": "/compact + instructions",
    "assist": "/compact + pins (Assist)",
    "recency": "/clear + recency truncation",
    "cg_dijkstra": "/clear + graph, shortest path, knapsack",
    "cg_diffusion": "/clear + graph, diffusion, knapsack",
    "cg_greedy": "/clear + graph, shortest path, greedy",
    "cg_2k": "/clear + graph, 2k budget",
    "cg_no_user": "/clear + graph, no user-prompt share",
}
ORDER = list(LABELS)
PINNED_KINDS = {"constraint", "decision", "rejected", "root_cause"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    return st.stdev(xs) if len(xs) > 1 else 0.0


def load(raw: str):
    runs = {}
    for d in sorted(glob.glob(os.path.join(raw, "*"))):
        if not os.path.isdir(d) or not glob.glob(os.path.join(d, "*.json")):
            continue
        run = os.path.basename(d)
        recs = {}
        for f in glob.glob(os.path.join(d, "*.json")):
            name = os.path.basename(f)[:-5]
            with open(f, encoding="utf-8") as fh:
                recs[name] = json.load(fh)
        runs[run] = recs
    return runs


def method_row(recs: list[dict]) -> dict:
    acc = [r["probe"]["accuracy"] for r in recs]
    facts = {f.id: mean([1.0 if r["probe"]["scores"][f.id] else 0.0 for r in recs]) for f in sc.FACTS}
    pinned = [f.id for f in sc.FACTS if f.kind in PINNED_KINDS]
    plain = [f.id for f in sc.FACTS if f.kind not in PINNED_KINDS]
    comp_in, comp_cost, comp_wall, comp_ms, summary_chars = [], [], [], [], []
    for r in recs:
        c = r.get("compaction_call")
        if r["spec"]["kind"] == "compact":
            comp_in.append((r.get("compaction") or {}).get("pre_tokens"))
            summary_chars.append((r.get("compaction") or {}).get("summary_chars"))
            comp_ms.append((r.get("compaction") or {}).get("duration_ms"))
        elif r["spec"]["kind"] == "clear":
            comp_in.append(0)
            comp_ms.append((r.get("selection") or {}).get("elapsed_ms"))
        if c:
            comp_cost.append(c["cost_usd"])
            comp_wall.append(c["wall_s"])
    fu = [r["followup"] for r in recs if r.get("followup")]
    viol = {k: sum(1 for f in fu if f["checks"]["violations"][k]) for k in
            (fu[0]["checks"]["violations"] if fu else {})}
    return {
        "n": len(recs),
        "recall": mean(acc), "recall_sd": sd(acc),
        "recall_pinned_kind": mean([facts[i] for i in pinned]),
        "recall_plain_facts": mean([facts[i] for i in plain]),
        "recall_by_fact": facts,
        "carried_tokens": mean([r["probe"]["carried_tokens"] for r in recs]),
        "carried_tokens_sd": sd([r["probe"]["carried_tokens"] for r in recs]),
        "pack_tokens_est": mean([(r.get("selection") or {}).get("pack_tokens_est") for r in recs
                                 if r["spec"]["kind"] == "clear"]),
        "summary_chars": mean(summary_chars),
        "compaction_input_tokens": mean(comp_in),
        "compaction_cost_usd": mean(comp_cost),
        "compaction_wall_s": mean(comp_wall),
        "compaction_internal_ms": mean(comp_ms),
        "followup_input_tokens": mean([f["call"]["input_tokens"] for f in fu]),
        "followup_input_tokens_sd": sd([f["call"]["input_tokens"] for f in fu]),
        "followup_cost_usd": mean([f["call"]["cost_usd"] for f in fu]),
        "followup_violations": mean([f["checks"]["n_violations"] for f in fu]),
        "followup_violations_total": sum(f["checks"]["n_violations"] for f in fu),
        "n_followup": len(fu),
        "followup_violations_by_rule": viol,
        "followup_tests_pass": mean([1.0 if f["checks"]["tests_pass"] else 0.0 for f in fu]),
        "followup_refund_ran": mean([1.0 if f["checks"]["refund_ran"] else 0.0 for f in fu]),
    }


def summarize_variant(runs: dict, variant: str) -> dict | None:
    sel = {k: r for k, r in runs.items() if k.startswith(variant) and "base" in r}
    if not sel:
        return None
    bases = [r["base"] for r in sel.values()]
    models = sorted({m for b in bases for c in b["calls"] for m in c.get("models", [])})
    out = {
        "runs": sorted(sel),
        "base": {
            "n": len(bases), "trim": bases[0]["trim"], "turns": bases[0]["n_turns"], "models": models,
            "input_tokens": mean([b["input_tokens"] for b in bases]),
            "input_tokens_sd": sd([b["input_tokens"] for b in bases]),
            "output_tokens": mean([b["output_tokens"] for b in bases]),
            "cost_usd": mean([b["cost_usd"] for b in bases]),
            "wall_s": mean([b["wall_s"] for b in bases]),
            "context_at_end": mean([r["full"]["probe"]["carried_tokens"] for r in sel.values() if "full" in r]),
            "failed_calls": sum(1 for b in bases for c in b["calls"] if not c["ok"]),
        },
        "methods": {},
    }
    for m in ORDER:
        recs = [r[m] for r in sel.values() if m in r]
        if recs:
            out["methods"][m] = method_row(recs)
    return out


def run_cost(r: dict) -> float:
    total = 0.0
    for k, v in r.items():
        if k == "base":
            total += v["cost_usd"]
            continue
        for part in ("compaction_call", "call"):
            total += (v.get(part) or {}).get("cost_usd", 0) or 0
        total += ((v.get("probe") or {}).get("call") or {}).get("cost_usd", 0) or 0
        total += ((v.get("followup") or {}).get("call") or {}).get("cost_usd", 0) or 0
    return total


def summarize(rounds: dict[str, dict]) -> dict:
    """rounds: {"v1": runs, "v2": runs}. Methods that do not depend on the plugin's
    selection code (no compaction, /compact, recency) are measured once, in v1,
    on the same base sessions, and reused for v2."""
    out: dict = {"rounds": {}}
    shared = ("full", "native", "native_instr", "recency")
    for rnd, runs in rounds.items():
        out["rounds"][rnd] = {}
        for variant in ("A", "B", "H"):
            sv = summarize_variant(runs, variant)
            if sv:
                out["rounds"][rnd][variant] = sv
    v1, v2 = out["rounds"].get("v1", {}), out["rounds"].get("v2", {})
    for variant in ("A", "B"):
        if variant in v1 and variant in v2:
            for m in shared:
                if m in v1[variant]["methods"] and m not in v2[variant]["methods"]:
                    v2[variant]["methods"][m] = dict(v1[variant]["methods"][m], measured_in="v1")
            v2[variant]["base"] = v1[variant]["base"]
    for variant in ("A", "B", "H"):
        if variant in v2:
            v2[variant]["methods"] = {m: v2[variant]["methods"][m] for m in ORDER if m in v2[variant]["methods"]}
    out["total_cost_usd"] = round(sum(run_cost(r) for runs in rounds.values() for r in runs.values()), 2)
    return out


# ------------------------------------------------------------------ offline sweep


def save_events(work: str, raw: str) -> None:
    """Copy each base session's event log into results/raw so the sweep is reproducible."""
    for run_dir in sorted(glob.glob(os.path.join(work, "*"))):
        run = os.path.basename(run_dir)
        evs = glob.glob(os.path.join(run_dir, "base", "store", "projects", "*", "threads", "*", "events.jsonl"))
        if not evs or not os.path.isdir(os.path.join(raw, run)):
            continue
        dest = os.path.join(raw, run, "base_events.jsonl.gz")
        with open(evs[0], "rb") as src, gzip.open(dest, "wb") as dst:
            dst.write(src.read())
        meta = os.path.join(os.path.dirname(evs[0]), "meta.json")
        with open(meta, encoding="utf-8") as fh:
            cwd = json.load(fh).get("cwd", "")
        with open(os.path.join(raw, run, "base_cwd.txt"), "w") as fh:
            fh.write(cwd)


def planted_in(statement: str, pack: str) -> bool:
    """True if every sentence of the planted statement is in the pack, word for word.
    Checked per sentence because one statement can become two pinned atoms."""
    import re

    sents = [x for x in re.split(r"(?<=[.!?])\s+", statement.strip()) if x]
    return all(x[:50] in pack for x in sents)


def offline_sweep(raw: str, variant: str = "A") -> dict:
    """Which planted facts are in the pack, by budget and selector. No model calls."""
    from context_graph import config
    from context_graph.atoms import extract_atoms
    from context_graph.budget import Item, total_value
    from context_graph.engine import build_pack
    from context_graph.events import Event

    sc.use("holdout" if variant == "H" else "main")
    budgets = [1000, 2000, 3000, 4000, 6000, 8000, 12000]
    variants = {
        "graph, shortest path": {"relevance": "dijkstra", "solver": "knapsack"},
        "graph, diffusion": {"relevance": "diffusion", "solver": "knapsack"},
        "graph, greedy": {"relevance": "dijkstra", "solver": "greedy"},
        "graph, no user-prompt share": {"relevance": "dijkstra", "solver": "knapsack", "user_share_replace": 0.0},
        "recency truncation": {"selector": "recency"},
    }
    out: dict = {"budgets": budgets, "coverage": {}, "solver_value": [], "timing_ms": []}
    for path in sorted(glob.glob(os.path.join(raw, f"{variant}*", "base_events.jsonl.gz"))):
        run = os.path.basename(os.path.dirname(path))
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            events = [Event.from_json(json.loads(line)) for line in fh if line.strip()]
        with open(os.path.join(os.path.dirname(path), "base_cwd.txt")) as fh:
            cwd = fh.read().strip()
        base_cfg = config.load("/nonexistent", env={})
        atoms = extract_atoms(events, cwd=cwd, chars_per_token=base_cfg["chars_per_token"])
        for name, over in variants.items():
            cfg = dict(base_cfg, **over)
            for b in budgets:
                text, sel = build_pack(atoms, cfg, "replace", budget=b)
                hits = {f.id: planted_in(sc.PLANTS[f.turn], text) for f in sc.FACTS}
                out["coverage"].setdefault(name, {}).setdefault(str(b), []).append(hits)
                if over.get("solver") in ("knapsack", "greedy") and "user_share_replace" not in over \
                        and over.get("relevance") == "dijkstra":
                    items = [Item(i, sel.scores[i]["value"], sel.scores[i]["tokens"]) for i in sel.scores
                             if sel.scores[i]["reason"] in ("knapsack", "greedy")]
                    out["solver_value"].append({"run": run, "solver": cfg["solver"], "budget": b,
                                                "value": total_value(items, [i.id for i in items])})
                if name == "graph, shortest path" and b == 6000:
                    out["timing_ms"].append(dict(sel.timings_ms, n_atoms=sel.n_atoms, n_edges=sel.n_edges,
                                                 pinned_tokens=sel.pinned_tokens))
    out["coverage_mean"] = {name: {b: mean([sum(h.values()) / len(h) for h in hs]) for b, hs in by_b.items()}
                            for name, by_b in out["coverage"].items()}
    sc.use("main")
    return out


# ------------------------------------------------------------------ outputs


def fmt(x, nd=0, pct=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{100 * x:.0f}%"
    return f"{x:,.{nd}f}"


def method_table(L: list, ms: dict, note_shared: bool = True) -> None:
    L.append("| Method | Runs | Probe recall | Rule-like facts | Plain facts | Carried-over tokens | "
             "Follow-up input tokens | Constraint violations (total) | Follow-up tests pass |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m, r in ms.items():
        mark = " *" if note_shared and r.get("measured_in") == "v1" else ""
        n_fu = r["n_followup"]
        L.append(f"| {LABELS[m]}{mark} | {r['n']} ({n_fu} follow-up) | "
                 f"{fmt(r['recall'], pct=True)} (sd {fmt(r['recall_sd'], pct=True)}) | "
                 f"{fmt(r['recall_pinned_kind'], pct=True)} | {fmt(r['recall_plain_facts'], pct=True)} | "
                 f"{fmt(r['carried_tokens'])} | {fmt(r['followup_input_tokens'])} | "
                 f"{r['followup_violations_total']} | {fmt(r['followup_tests_pass'], pct=True)} |")


def by_turn_table(L: list, ms: dict) -> None:
    L.append("| Method | " + " | ".join(f"{f.id} t{f.turn}" for f in sc.FACTS) + " |")
    L.append("|---|" + "---|" * len(sc.FACTS))
    for m, r in ms.items():
        L.append(f"| {LABELS[m]} | " + " | ".join(fmt(r["recall_by_fact"][f.id], pct=True) for f in sc.FACTS) + " |")


def tables(s: dict, sweeps: dict) -> str:
    L: list[str] = []
    v1, v2 = s["rounds"].get("v1", {}), s["rounds"].get("v2", {})
    A = v2.get("A") or v1.get("A")
    rnd = "v2" if v2.get("A") else "v1"
    L.append(f"Model(s): {', '.join(A['base']['models'])}. Base sessions per variant: {A['base']['n']}. "
             f"Total API cost of every recorded call: ${s['total_cost_usd']}.\n")
    if v2.get("A"):
        L.append("## Main scenario (variant A), extractor v2\n")
        method_table(L, A["methods"])
        L.append("\n\\* measured once in round v1 on the same base sessions; these methods do not use the "
                 "plugin's selection code.\n")
    if v1.get("A"):
        L.append("## Main scenario (variant A), extractor v1 (before pinning user-stated facts)\n")
        method_table(L, {m: r for m, r in v1["A"]["methods"].items()}, note_shared=False)
    if v2.get("H"):
        L.append("\n## Held-out facts (variant H), extractor v2\n")
        method_table(L, v2["H"]["methods"], note_shared=False)
        if v1.get("H"):
            L.append("\nSame held-out sessions with the v1 extractor:\n")
            method_table(L, v1["H"]["methods"], note_shared=False)
    L.append(f"\n## Cost of the compaction step itself (variant A, round {rnd})\n")
    L.append("| Method | Model input tokens | Cost (USD) | Wall time (s) | Time inside (ms) | Summary or pack size |")
    L.append("|---|---|---|---|---|---|")
    for m, r in A["methods"].items():
        if m == "full":
            continue
        size = (f"{fmt(r['summary_chars'])} chars" if r["summary_chars"] else
                f"{fmt(r['pack_tokens_est'])} tokens (est.)" if r["pack_tokens_est"] else "n/a")
        L.append(f"| {LABELS[m]} | {fmt(r['compaction_input_tokens'])} | {fmt(r['compaction_cost_usd'], 4)} | "
                 f"{fmt(r['compaction_wall_s'], 1)} | {fmt(r['compaction_internal_ms'])} | {size} |")
    L.append(f"\n## Violations by rule (variant A, round {rnd}, count over follow-up runs)\n")
    rules = list(next(iter(A["methods"].values()))["followup_violations_by_rule"])
    L.append("| Method | " + " | ".join(rules) + " |")
    L.append("|---|" + "---|" * len(rules))
    for m, r in A["methods"].items():
        L.append(f"| {LABELS[m]} | " + " | ".join(str(r["followup_violations_by_rule"].get(k, 0)) for k in rules) + " |")
    L.append(f"\n## Recall by the turn the fact was planted (variant A, round {rnd})\n")
    by_turn_table(L, A["methods"])
    if v2.get("H"):
        L.append("\n## Recall by planted turn, held-out facts (variant H, extractor v2)\n")
        by_turn_table(L, v2["H"]["methods"])
    B = v2.get("B") or v1.get("B")
    if B:
        a, bb = A["base"], B["base"]
        L.append("\n## Trimming ablation (base sessions)\n")
        L.append("| Variant | Session input tokens | Context at end of session | Session cost (USD) |")
        L.append("|---|---|---|---|")
        for name, x in (("A: trimming on", a), ("B: trimming off", bb)):
            L.append(f"| {name} | {fmt(x['input_tokens'])} (sd {fmt(x['input_tokens_sd'])}) | "
                     f"{fmt(x['context_at_end'])} | {fmt(x['cost_usd'], 3)} |")
        L.append("\n| Method | Recall, trimming on (A) | Recall, trimming off (B) | Follow-up tokens A | Follow-up tokens B |")
        L.append("|---|---|---|---|---|")
        for m, r in B["methods"].items():
            ra = A["methods"].get(m, {})
            L.append(f"| {LABELS[m]} | {fmt(ra.get('recall'), pct=True)} | {fmt(r['recall'], pct=True)} | "
                     f"{fmt(ra.get('followup_input_tokens'))} | {fmt(r['followup_input_tokens'])} |")
    for variant, sweep in sweeps.items():
        if not sweep or not sweep.get("coverage_mean"):
            continue
        L.append(f"\n## Offline: planted facts present in the pack, by budget (variant {variant}, no model call)\n")
        L.append("| Selector | " + " | ".join(f"{b:,}" for b in sweep["budgets"]) + " |")
        L.append("|---|" + "---|" * len(sweep["budgets"]))
        for name, cov in sweep["coverage_mean"].items():
            L.append(f"| {name} | " + " | ".join(fmt(cov[str(b)], pct=True) for b in sweep["budgets"]) + " |")
        kv: dict = {}
        for r in sweep["solver_value"]:
            kv.setdefault((r["run"], r["budget"]), {})[r["solver"]] = r["value"]
        ratios = [v["greedy"] / v["knapsack"] for v in kv.values() if v.get("knapsack") and "greedy" in v]
        if ratios:
            L.append(f"\nGreedy reached {min(ratios) * 100:.2f}% to {max(ratios) * 100:.2f}% of the knapsack's "
                     f"objective value (mean {mean(ratios) * 100:.2f}%) over {len(ratios)} run x budget cases.")
        if sweep.get("timing_ms"):
            t = sweep["timing_ms"]
            L.append(f"\nSelection at 6,000 tokens: graph build {fmt(mean([x['graph'] for x in t]), 1)} ms, "
                     f"relevance {fmt(mean([x['relevance'] for x in t]), 1)} ms, knapsack "
                     f"{fmt(mean([x['solver'] for x in t]), 1)} ms, for {fmt(mean([x['n_atoms'] for x in t]))} atoms, "
                     f"{fmt(mean([x['n_edges'] for x in t]))} edges, {fmt(mean([x['pinned_tokens'] for x in t]))} "
                     f"pinned tokens.")
    return "\n".join(L) + "\n"


def compare_table(s: dict) -> str:
    comp = s.get("offline_compare") or {}
    if len(comp) < 2:
        return ""
    L = ["\n## Offline: extractor versions on the same recorded sessions (graph, shortest path, knapsack)\n",
         "Share of the twelve planted facts whose statement is in the pack, every sentence word for word. No model call; "
         "deterministic given the recorded event logs.\n",
         "| Extractor | Scenario | 2,000 | 4,000 | 6,000 | 8,000 |", "|---|---|---|---|---|---|"]
    for name, by_v in comp.items():
        for v, cov in by_v.items():
            c = cov.get("graph, shortest path", {})
            L.append(f"| {name} | {'main (A)' if v == 'A' else 'held-out (H)'} | "
                     + " | ".join(fmt(c.get(b), pct=True) for b in ("2000", "4000", "6000", "8000")) + " |")
    L.append("\nPer fact at 6,000 tokens (share of runs whose pack contains it):\n")
    L.append("| Extractor | Scenario | " + " | ".join(f.id for f in sc.FACTS) + " |")
    L.append("|---|---|" + "---|" * len(sc.FACTS))
    for name, by_v in s["offline_compare_by_fact"].items():
        for v, facts in by_v.items():
            L.append(f"| {name} | {v} | " + " | ".join(fmt(facts[f.id], pct=True) for f in sc.FACTS) + " |")
    return "\n".join(L) + "\n"


FAMILY = {"full": 0, "recency": 0, "native": 1, "native_instr": 1, "assist": 1,
          "cg_dijkstra": 2, "cg_diffusion": 2, "cg_greedy": 2, "cg_2k": 2, "cg_no_user": 2}
FAMILY_NAMES = ["baselines", "native /compact", "context-graph /clear"]
COLORS = ["#52514e", "#2a78d6", "#eb6834"]  # muted ink, then categorical slots 1 and 2
SHORT = {"full": "no compaction", "native": "/compact", "native_instr": "/compact + instr.",
         "assist": "Assist", "recency": "recency", "cg_dijkstra": "graph (shortest path)",
         "cg_diffusion": "graph (diffusion)", "cg_greedy": "graph (greedy)", "cg_2k": "graph, 2k budget",
         "cg_no_user": "graph, no user share"}


def plots(ms: dict, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

    from matplotlib.ticker import FuncFormatter

    def place_labels(ax, pts):
        """Put each label next to its point without overlapping the others."""
        ax.figure.canvas.draw()
        to_px = ax.transData.transform
        boxes = []
        for x, y, text in sorted(pts, key=lambda p: (-p[1], p[0])):
            px, py = to_px((x, y))
            w, h = 5.6 * len(text) * 1.6, 11 * 1.6  # rough text box in pixels at dpi 150 / 100 * 1.6
            for dx, dy in ((8, 4), (8, -16), (8, 22), (-w - 8, 4), (-w - 8, -16), (8, 40), (8, -34),
                           (-w - 8, 22), (-w / 2, 26), (-w / 2, -30)):
                box = (px + dx, py + dy, px + dx + w, py + dy + h)
                if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in boxes):
                    break
            boxes.append(box)
            ax.annotate(text, (x, y), textcoords="offset pixels", xytext=(box[0] - px, box[1] - py),
                        fontsize=8, color="#52514e")

    def scatter(ax, xkey, xlabel):
        labels = []
        for fam in range(3):
            pts = [(m, r) for m, r in ms.items() if FAMILY[m] == fam and r.get(xkey) is not None]
            if not pts:
                continue
            ax.errorbar([r[xkey] for _, r in pts], [100 * r["recall"] for _, r in pts],
                        yerr=[100 * r["recall_sd"] for _, r in pts], fmt="o", ms=8, color=COLORS[fam],
                        ecolor=COLORS[fam], elinewidth=1, capsize=0, label=FAMILY_NAMES[fam],
                        markeredgecolor="#fcfcfb", markeredgewidth=2)
            labels += [(r[xkey], 100 * r["recall"], SHORT[m]) for m, r in pts]
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        xs = [p[0] for p in labels]
        span = (max(xs) - min(xs)) or max(xs)
        ax.set_xlim(max(0, min(xs) - 0.08 * span), max(xs) + 0.35 * span)
        ax.set_ylabel("probe recall (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, color="#e9e8e4", linewidth=0.8)
        ax.set_axisbelow(True)
        return labels

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    l0 = scatter(axes[0], "carried_tokens", "tokens carried over after compaction (measured)")
    l1 = scatter(axes[1], "followup_input_tokens", "input tokens for the follow-up task after compaction")
    handles, names = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, names, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    place_labels(axes[0], l0)
    place_labels(axes[1], l1)
    axes[0].set_title("Recall vs tokens carried over", loc="left", fontsize=11)
    axes[1].set_title("Recall vs tokens spent on the next task", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tokens_vs_recall.png"), dpi=150)
    plt.close(fig)

    rows = [m for m in ORDER if m in ms]
    grid = [[100 * ms[m]["recall_by_fact"][f.id] for f in sc.FACTS] for m in rows]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.4))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9ec5f0", "#2a78d6", "#0d3b75"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="#fcfcfb" if v >= 60 else "#0b0b0b")
    ax.set_xticks(range(len(sc.FACTS)))
    ax.set_xticklabels([f"t{f.turn}" for f in sc.FACTS])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT[m] for m in rows])
    ax.set_xlabel("turn the fact was planted (of 36); cell = % of runs that recalled it")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Recall by position in the session", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recall_by_turn.png"), dpi=150)
    plt.close(fig)

def event_dirs(raw: str) -> dict[str, str]:
    """For each variant, the round directory that holds its recorded base-session events."""
    out = {}
    for variant in ("A", "H"):
        for rnd in ("v1", "v2"):
            d = os.path.join(raw, rnd)
            if glob.glob(os.path.join(d, f"{variant}*", "base_events.jsonl.gz")):
                out[variant] = d
                break
    return out


def sweep_with(code_root: str, raw: str) -> dict:
    """Run the offline sweep with another checkout's plugin code, in a subprocess."""
    env = dict(os.environ, CG_CODE_ROOT=code_root)
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--raw", raw, "--sweep-only"],
                          env=env, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--work", help="benchmark work dir, to save base-session event logs for the offline sweep")
    ap.add_argument("--compare", action="append", default=[], metavar="NAME=PATH",
                    help="also run the offline sweep with the plugin code at PATH (e.g. v1=/tmp/cg-v1)")
    ap.add_argument("--sweep-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.sweep_only:
        res = {v: offline_sweep(d, v) for v, d in event_dirs(args.raw).items()}
        print(json.dumps({v: {"coverage_mean": r["coverage_mean"], "coverage": r["coverage"]} for v, r in res.items()}))
        return 0
    rounds = {}
    for rnd in ("v1", "v2"):
        d = os.path.join(args.raw, rnd)
        if os.path.isdir(d):
            if args.work:
                save_events(args.work, d)
            rounds[rnd] = load(d)
    if not rounds:
        print("no runs in", args.raw)
        return 1
    s = summarize(rounds)
    sweeps = {v: offline_sweep(d, v) for v, d in event_dirs(args.raw).items()}
    s["offline"] = {v: {k: x for k, x in sw.items() if k != "coverage"} for v, sw in sweeps.items()}
    compare = {"v2 (this checkout)": {v: {"coverage_mean": sw["coverage_mean"], "coverage": sw["coverage"]}
                                      for v, sw in sweeps.items()}}
    for item in args.compare:
        name, _, path = item.partition("=")
        compare[name] = sweep_with(path, args.raw)
    s["offline_compare"] = {n: {v: x["coverage_mean"] for v, x in c.items()} for n, c in compare.items()}
    s["offline_compare_by_fact"] = {
        n: {v: {f.id: mean([1.0 if h[f.id] else 0.0 for h in x["coverage"]["graph, shortest path"]["6000"]])
                for f in sc.FACTS} for v, x in c.items()} for n, c in compare.items()}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1)
    text = tables(s, sweeps) + compare_table(s)
    with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    A = s["rounds"].get("v2", {}).get("A") or s["rounds"]["v1"]["A"]
    plots(A["methods"], args.out)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
