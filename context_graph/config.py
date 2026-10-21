"""Settings, read from (later wins):

  1. built-in defaults below
  2. <store home>/config.json
  3. <project>/.context-graph.json
  4. environment variables CONTEXT_GRAPH_<KEY> for top-level keys
     (for example CONTEXT_GRAPH_RELEVANCE=diffusion) and CONTEXT_GRAPH_TRIM=0|1
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

DEFAULTS: dict[str, Any] = {
    "relevance": "dijkstra",  # dijkstra | diffusion
    "solver": "knapsack",  # knapsack | greedy
    "selector": "graph",  # graph | recency (the recency-truncation baseline)
    "budget_replace": 6000,  # tokens carried over after /clear
    "budget_assist": 2000,  # tokens added on top of the /compact summary
    "user_share_replace": 0.35,  # share of the budget kept for the user's own prompts after /clear
    "user_share_assist": 0.0,  # the /compact summary already lists the user's messages
    "inject_on_clear": True,
    "inject_on_compact": True,
    "half_life_turns": 8.0,
    "diffusion_time": 2.0,
    "focus_turns": 2,
    "chars_per_token": 2.2,  # measured on claude-haiku-5-5, see tokens.py
    "bucket_tokens": 10,
    "trim": {
        "enabled": True,
        "tools": ["Bash"],
        "max_lines": 80,  # outputs longer than this are trimmed
        "max_chars": 8000,
        "head_lines": 25,
        "tail_lines": 25,
        "max_error_lines": 20,
    },
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> None:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


def _coerce(raw: str, like: Any) -> Any:
    if isinstance(like, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(float(raw))
    if isinstance(like, float):
        return float(raw)
    return raw


def load(home: str, cwd: str = "", env: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    cfg = copy.deepcopy(DEFAULTS)
    for path in (os.path.join(home, "config.json"), os.path.join(cwd, ".context-graph.json") if cwd else ""):
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    _merge(cfg, json.load(fh))
            except (OSError, ValueError):
                pass
    for key, like in DEFAULTS.items():
        raw = env.get("CONTEXT_GRAPH_" + key.upper())
        if raw is None:
            continue
        if key == "trim":
            cfg["trim"]["enabled"] = _coerce(raw, True)
        else:
            cfg[key] = _coerce(raw, like)
    return cfg
