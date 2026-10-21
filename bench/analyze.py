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
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
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
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "base.json")):
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
        "followup_violations_by_rule": viol,
        "followup_tests_pass": mean([1.0 if f["checks"]["tests_pass"] else 0.0 for f in fu]),
        "followup_refund_ran": mean([1.0 if f["checks"]["refund_ran"] else 0.0 for f in fu]),
    }


def summarize(runs: dict) -> dict:
    out: dict = {"runs": sorted(runs), "variants": {}, "methods": {}, "methods_B": {}, "base": {}}
    for variant in ("A", "B"):
        bases = [r["base"] for k, r in runs.items() if k.startswith(variant)]
        if not bases:
            continue
        models = sorted({m for b in bases for c in b["calls"] for m in c.get("models", [])})
        out["base"][variant] = {
            "n": len(bases), "trim": bases[0]["trim"], "turns": bases[0]["n_turns"], "models": models,
            "input_tokens": mean([b["input_tokens"] for b in bases]),
            "input_tokens_sd": sd([b["input_tokens"] for b in bases]),
            "output_tokens": mean([b["output_tokens"] for b in bases]),
            "cost_usd": mean([b["cost_usd"] for b in bases]),
            "wall_s": mean([b["wall_s"] for b in bases]),
            "context_at_end": mean([r["full"]["probe"]["carried_tokens"] for k, r in runs.items()
                                    if k.startswith(variant) and "full" in r]),
            "failed_calls": sum(1 for b in bases for c in b["calls"] if not c["ok"]),
        }
        key = "methods" if variant == "A" else "methods_B"
        for m in ORDER:
            recs = [r[m] for k, r in runs.items() if k.startswith(variant) and m in r]
            if recs:
                out[key][m] = method_row(recs)
    total = 0.0
    for r in runs.values():
        total += r["base"]["cost_usd"]
        for k, v in r.items():
            if k == "base":
                continue
            for part in ("compaction_call",):
                total += (v.get(part) or {}).get("cost_usd", 0) or 0
            total += ((v.get("probe") or {}).get("call") or {}).get("cost_usd", 0) or 0
            total += ((v.get("followup") or {}).get("call") or {}).get("cost_usd", 0) or 0
            total += (v.get("call") or {}).get("cost_usd", 0) or 0
    out["total_cost_usd"] = round(total, 2)
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


def offline_sweep(raw: str) -> dict:
    """Which planted facts are in the pack, by budget and method. No model calls."""
    from context_graph import config
    from context_graph.atoms import extract_atoms
    from context_graph.budget import Item, total_value
    from context_graph.engine import build_pack
    from context_graph.events import Event

    budgets = [1000, 2000, 3000, 4000, 6000, 8000, 12000]
    variants = {
        "graph_dijkstra": {"relevance": "dijkstra", "solver": "knapsack"},
        "graph_diffusion": {"relevance": "diffusion", "solver": "knapsack"},
        "graph_greedy": {"relevance": "dijkstra", "solver": "greedy"},
        "graph_no_user": {"relevance": "dijkstra", "solver": "knapsack", "user_share_replace": 0.0},
        "recency": {"selector": "recency"},
    }
    out: dict = {"budgets": budgets, "coverage": {}, "solver_value": [], "timing_ms": []}
    for path in sorted(glob.glob(os.path.join(raw, "A*", "base_events.jsonl.gz"))):
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
                hits = {f.id: sc.PLANTS[f.turn][:60] in text for f in sc.FACTS}
                out["coverage"].setdefault(name, {}).setdefault(str(b), []).append(hits)
                if name in ("graph_dijkstra", "graph_greedy") and sel.selector == "graph":
                    items = [Item(i, sel.scores[i]["value"], sel.scores[i]["tokens"]) for i in sel.scores
                             if sel.scores[i]["reason"] in ("knapsack", "greedy")]
                    out["solver_value"].append({"run": run, "solver": cfg["solver"], "budget": b,
                                                "value": total_value(items, [i.id for i in items])})
                if name == "graph_dijkstra" and b == 6000:
                    out["timing_ms"].append(dict(sel.timings_ms, n_atoms=sel.n_atoms, n_edges=sel.n_edges))
    cov = {}
    for name, by_b in out["coverage"].items():
        cov[name] = {b: mean([sum(h.values()) / len(h) for h in hs]) for b, hs in by_b.items()}
    out["coverage_mean"] = cov
    return out


# ------------------------------------------------------------------ outputs


def fmt(x, nd=0, pct=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{100 * x:.0f}%"
    return f"{x:,.{nd}f}"


def tables(s: dict, sweep: dict | None) -> str:
    L = []
    b = s["base"].get("A", {})
    L.append(f"Runs: {', '.join(s['runs'])}. Model(s): {', '.join(b.get('models', []))}. "
             f"Total API cost of everything below: ${s['total_cost_usd']}.\n")
    L.append("## Context kept and tokens after compaction (variant A, mean of runs)\n")
    L.append("| Method | Probe recall | Pinned-kind facts | Plain facts | Carried-over tokens | "
             "Follow-up input tokens | Constraint violations (total) | Follow-up tests pass |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m, r in s["methods"].items():
        L.append(f"| {LABELS[m]} | {fmt(r['recall'], pct=True)} (sd {fmt(r['recall_sd'], pct=True)}) | "
                 f"{fmt(r['recall_pinned_kind'], pct=True)} | {fmt(r['recall_plain_facts'], pct=True)} | "
                 f"{fmt(r['carried_tokens'])} | {fmt(r['followup_input_tokens'])} | "
                 f"{r['followup_violations_total']} | {fmt(r['followup_tests_pass'], pct=True)} |")
    L.append("\n## Cost of the compaction step itself (variant A, mean of runs)\n")
    L.append("| Method | Model input tokens | Cost (USD) | Wall time (s) | Summary or pack size |")
    L.append("|---|---|---|---|---|")
    for m, r in s["methods"].items():
        if m == "full":
            continue
        size = (f"{fmt(r['summary_chars'])} chars" if r["summary_chars"] else
                f"{fmt(r['pack_tokens_est'])} tokens (est.)" if r["pack_tokens_est"] else "n/a")
        L.append(f"| {LABELS[m]} | {fmt(r['compaction_input_tokens'])} | {fmt(r['compaction_cost_usd'], 4)} | "
                 f"{fmt(r['compaction_wall_s'], 1)} | {size} |")
    L.append("\n## Violations by rule (variant A, count over runs)\n")
    rules = list(next(iter(s["methods"].values()))["followup_violations_by_rule"])
    L.append("| Method | " + " | ".join(rules) + " |")
    L.append("|---|" + "---|" * len(rules))
    for m, r in s["methods"].items():
        L.append(f"| {LABELS[m]} | " + " | ".join(str(r["followup_violations_by_rule"].get(k, 0)) for k in rules) + " |")
    L.append("\n## Recall by the turn the fact was planted (variant A)\n")
    L.append("| Method | " + " | ".join(f"{f.id} t{f.turn}" for f in sc.FACTS) + " |")
    L.append("|---|" + "---|" * len(sc.FACTS))
    for m, r in s["methods"].items():
        L.append(f"| {LABELS[m]} | " + " | ".join(fmt(r["recall_by_fact"][f.id], pct=True) for f in sc.FACTS) + " |")
    if s["base"].get("B"):
        a, bb = s["base"]["A"], s["base"]["B"]
        L.append("\n## Trimming ablation (base session, mean of runs)\n")
        L.append("| Variant | Session input tokens | Context at end of session | Session cost (USD) |")
        L.append("|---|---|---|---|")
        for name, x in (("A: trimming on", a), ("B: trimming off", bb)):
            L.append(f"| {name} | {fmt(x['input_tokens'])} (sd {fmt(x['input_tokens_sd'])}) | "
                     f"{fmt(x['context_at_end'])} | {fmt(x['cost_usd'], 3)} |")
        L.append("\n| Method | Recall, trimming on (A) | Recall, trimming off (B) |")
        L.append("|---|---|---|")
        for m, r in s["methods_B"].items():
            ra = s["methods"].get(m, {}).get("recall")
            L.append(f"| {LABELS[m]} | {fmt(ra, pct=True)} | {fmt(r['recall'], pct=True)} |")
    if sweep and sweep.get("coverage_mean"):
        L.append("\n## Offline: planted facts present in the pack, by budget (no model call)\n")
        L.append("| Selector | " + " | ".join(f"{b:,}" for b in sweep["budgets"]) + " |")
        L.append("|---|" + "---|" * len(sweep["budgets"]))
        for name, cov in sweep["coverage_mean"].items():
            L.append(f"| {name} | " + " | ".join(fmt(cov[str(b)], pct=True) for b in sweep["budgets"]) + " |")
        kv = {}
        for r in sweep["solver_value"]:
            kv.setdefault((r["run"], r["budget"]), {})[r["solver"]] = r["value"]
        ratios = [v["greedy"] / v["knapsack"] for v in kv.values() if v.get("knapsack") and "greedy" in v]
        if ratios:
            L.append(f"\nGreedy reached {fmt(min(ratios) * 100, 2)}% to {fmt(max(ratios) * 100, 2)}% of the "
                     f"knapsack's objective value (mean {fmt(mean(ratios) * 100, 2)}%) over "
                     f"{len(ratios)} run x budget cases.")
        if sweep.get("timing_ms"):
            t = sweep["timing_ms"]
            L.append(f"\nSelection time at 6,000 tokens: graph build {fmt(mean([x['graph'] for x in t]), 1)} ms, "
                     f"relevance {fmt(mean([x['relevance'] for x in t]), 1)} ms, knapsack "
                     f"{fmt(mean([x['solver'] for x in t]), 1)} ms, for {fmt(mean([x['n_atoms'] for x in t]))} atoms "
                     f"and {fmt(mean([x['n_edges'] for x in t]))} edges.")
    return "\n".join(L) + "\n"


FAMILY = {"full": 0, "recency": 0, "native": 1, "native_instr": 1, "assist": 1,
          "cg_dijkstra": 2, "cg_diffusion": 2, "cg_greedy": 2, "cg_2k": 2, "cg_no_user": 2}
FAMILY_NAMES = ["baselines", "native /compact", "context-graph /clear"]
COLORS = ["#52514e", "#2a78d6", "#eb6834"]  # muted ink, then categorical slots 1 and 2
SHORT = {"full": "no compaction", "native": "/compact", "native_instr": "/compact + instr.",
         "assist": "Assist", "recency": "recency", "cg_dijkstra": "graph (shortest path)",
         "cg_diffusion": "graph (diffusion)", "cg_greedy": "graph (greedy)", "cg_2k": "graph, 2k budget",
         "cg_no_user": "graph, no user share"}


def plots(s: dict, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    ms = s["methods"]

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
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    place_labels(axes[0], l0)
    place_labels(axes[1], l1)
    axes[0].set_title("Recall vs tokens carried over", loc="left", fontsize=11)
    axes[1].set_title("Recall vs tokens spent on the next task", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tokens_vs_recall.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    show = [m for m in ("full", "native", "native_instr", "recency", "cg_dijkstra", "cg_diffusion") if m in ms]
    palette = {"full": "#52514e", "native": "#2a78d6", "native_instr": "#6250d6", "recency": "#1baf7a",
               "cg_dijkstra": "#eb6834", "cg_diffusion": "#eda100"}
    turns = [f.turn for f in sc.FACTS]
    for m in show:
        ys = [100 * ms[m]["recall_by_fact"][f.id] for f in sc.FACTS]
        ax.plot(turns, ys, marker="o", ms=5, lw=2, color=palette[m], label=SHORT[m])
    ax.set_xlabel("turn the fact was planted (of 36)")
    ax.set_ylabel("recall after compaction (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(turns)
    ax.grid(True, color="#e9e8e4", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    ax.set_title("Recall by position in the session", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recall_by_turn.png"), dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--work", help="benchmark work dir, to save base-session event logs for the offline sweep")
    args = ap.parse_args()
    if args.work:
        save_events(args.work, args.raw)
    runs = load(args.raw)
    if not runs:
        print("no runs in", args.raw)
        return 1
    s = summarize(runs)
    sweep = offline_sweep(args.raw)
    s["offline"] = {k: v for k, v in sweep.items() if k != "coverage"}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1)
    with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(tables(s, sweep))
    plots(s, args.out)
    print(tables(s, sweep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
