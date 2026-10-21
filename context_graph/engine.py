"""Glue: event log -> atoms -> selection -> pack, plus the selection log."""

from __future__ import annotations

import time
from typing import Any

from . import pack as packmod
from .atoms import Atom, extract_atoms
from .select import Selection, select
from .store import Thread
from .tokens import estimate


def thread_atoms(thread: Thread, cfg: dict[str, Any]) -> list[Atom]:
    cwd = thread.meta().get("cwd", "")
    return extract_atoms(thread.events(), cwd=cwd, chars_per_token=float(cfg["chars_per_token"]))


def build_pack(atoms: list[Atom], cfg: dict[str, Any], mode: str, ctx: str = "ctx",
               budget: int | None = None) -> tuple[str, Selection]:
    cpt = float(cfg["chars_per_token"])
    if budget is None:
        budget = int(cfg["budget_assist"] if mode == "assist" else cfg["budget_replace"])
    reserved = packmod.header_tokens(mode, cfg.get("selector", "graph"), ctx, cpt)
    share = float(cfg["user_share_assist"] if mode == "assist" else cfg["user_share_replace"])
    sel = select(atoms, cfg, budget, reserved_tokens=reserved, user_share=share)
    text = packmod.render(sel, atoms, mode=mode, ctx=ctx, cpt=cpt)
    # The budget is hard: if rounding left the rendered pack over it, drop the
    # least valuable chosen atom until it fits. Pins are never dropped.
    pinned = set(sel.pinned)
    while estimate(text, cpt) > budget:
        chosen = [i for i in sel.kept if i not in pinned]
        if not chosen:
            sel.overflow = True
            break
        worst = min(chosen, key=lambda i: float(sel.scores[i].get("value", 0) or 0)
                    if sel.selector == "graph" else sel.scores[i]["turn"])
        sel.kept.remove(worst)
        sel.dropped.append(worst)
        sel.chosen_tokens -= sel.scores[worst]["tokens"]
        text = packmod.render(sel, atoms, mode=mode, ctx=ctx, cpt=cpt)
    return text, sel


def selection_record(sel: Selection, atoms: list[Atom], text: str, cfg: dict[str, Any], mode: str,
                     trigger: str, elapsed_ms: float) -> dict[str, Any]:
    by_id = {a.id: a for a in atoms}

    def row(aid: str) -> dict[str, Any]:
        r = dict(sel.scores.get(aid, {}))
        r["id"] = aid
        r["render"] = by_id[aid].render[:300] if aid in by_id else ""
        return r

    return {
        "trigger": trigger,
        "mode": mode,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {k: cfg.get(k) for k in ("selector", "relevance", "solver", "half_life_turns",
                                           "diffusion_time", "focus_turns", "chars_per_token",
                                           "user_share_replace", "user_share_assist")},
        "budget": sel.budget,
        "pack_chars": len(text),
        "pack_tokens_est": estimate(text, float(cfg["chars_per_token"])),
        "n_atoms": sel.n_atoms,
        "n_edges": sel.n_edges,
        "pinned_tokens": sel.pinned_tokens,
        "overflow": sel.overflow,
        "focus": sel.focus,
        "elapsed_ms": round(elapsed_ms, 2),
        "timings_ms": sel.timings_ms,
        "kept": [row(i) for i in sel.kept],
        "dropped": sorted((row(i) for i in sel.dropped), key=lambda r: -float(r.get("value", 0) or 0)),
        "pack": text,
    }


def select_and_log(thread: Thread, cfg: dict[str, Any], mode: str, trigger: str,
                   ctx: str = "ctx") -> tuple[str, dict[str, Any]]:
    t0 = time.perf_counter()
    atoms = thread_atoms(thread, cfg)
    text, sel = build_pack(atoms, cfg, mode, ctx=ctx)
    rec = selection_record(sel, atoms, text, cfg, mode, trigger, (time.perf_counter() - t0) * 1000)
    rec["log_path"] = thread.write_selection(rec)
    return text, rec
