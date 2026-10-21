"""Measure characters per token for the session model, from API usage counts.

Each sample is sent as a prompt in a fresh session with tools off; the input
tokens minus those of an empty prompt are the sample's tokens. Samples: English
prose (the MIT license text, repeated), Python source (context_graph/atoms.py),
and a real pack from the benchmark if one exists.

Usage: python3 bench/token_ratio.py [--model haiku]   -> results/token_ratio.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import claude_cli as cc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    args = ap.parse_args()
    if not cc.have_key():
        print(f"{cc.KEY_VAR} is not set: nothing measured.")
        return 0
    samples = {
        "english (MIT license x8)": open(os.path.join(ROOT, "LICENSE"), encoding="utf-8").read() * 8,
        "python (atoms.py, first 9000 chars)": open(os.path.join(ROOT, "context_graph", "atoms.py"),
                                                    encoding="utf-8").read()[:9000],
    }
    packs = sorted(glob.glob(os.path.join(ROOT, "results", "raw", "A*", "cg_dijkstra.json")))
    if packs:
        with open(packs[0], encoding="utf-8") as fh:
            samples[f"pack ({os.path.relpath(packs[0], ROOT)})"] = json.load(fh).get("pack", "")
    work = tempfile.mkdtemp(prefix="cg-tok-")
    env = cc.child_env(os.path.join(work, "claude"), os.path.join(work, "home"), {"CONTEXT_GRAPH_DISABLE": "1"})

    def tokens(text: str) -> tuple[int, list[str]]:
        c = cc.run("Reply with ok.\n\n" + text, work, env, model=args.model, tools="")
        return c.input_tokens, c.models

    base, models = tokens("")
    out = {"model_arg": args.model, "models": models, "empty_prompt_tokens": base, "samples": []}
    for name, text in samples.items():
        n, _ = tokens(text)
        n -= base
        out["samples"].append({"sample": name, "chars": len(text), "tokens": n,
                               "chars_per_token": round(len(text) / n, 3)})
        print(f"{name}: {len(text)} chars, {n} tokens, {len(text) / n:.2f} chars/token")
    with open(os.path.join(ROOT, "results", "token_ratio.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
