# Results: how each number was measured

Everything here is produced by scripts in `bench/` and can be regenerated:

    BENCH_ANTHROPIC_API_KEY=... python3 bench/run_bench.py --runs A1 A2 A3 B1 B2 B3 --parallel 6
    python3 bench/analyze.py --work /tmp/cg-bench --compare v1=<worktree of commit c6b9b26>
    BENCH_ANTHROPIC_API_KEY=... python3 bench/token_ratio.py
    BENCH_ANTHROPIC_API_KEY=... python3 bench/e2e.py

Raw data is split by extractor version. `raw/v1/` holds every method on the main scenario
(A, B) run with the extractor before user-stated facts were pinned (plugin code at commit
`c6b9b26`, identical in `context_graph/` to the code the first runs used). `raw/v2/` holds the
held-out sessions (H) run with the current extractor (commit `ed54fb3`). Each table states how
many runs a number comes from; a method with fewer than three runs has fewer recorded runs. One
line was removed from the `/compact` summary in `raw/v1/A3/native_instr.json` because it quoted
a git setting of the benchmark machine.

| File | What it is |
|---|---|
| `raw/<version>/<run>/base.json` | every call of the scripted base session: prompt, result text, usage, cost, wall time |
| `raw/<version>/<run>/baseline_probe.json` | the probe asked in a fresh, empty session (the fixed overhead) |
| `raw/<version>/<run>/<method>.json` | one compaction method on that session: the compaction call, compaction metadata or selection log, the pack or summary, the probe answer and its scores, the follow-up call and its checks |
| `raw/<version>/<run>/base_events.jsonl.gz` | the event log context-graph captured during the base session (input to the offline sweep) |
| `raw/meta-*.json` | Claude Code version, model argument, run ids |
| `summary.json` | every aggregate below, computed by `bench/analyze.py` |
| `tables.md` | the tables quoted in the top-level README |
| `tokens_vs_recall.png`, `recall_by_turn.png` | the plots |
| `token_ratio.json` | characters per token measured for the session model (measured during the pilot; see its note) |
| `e2e/e2e-report.json` | the end-to-end plugin check |

## Setup

* **Sessions.** `bench/scenario.py` holds a 36-turn script of ordinary coding requests on
  `bench/demo_repo` (a small Python ledger with a CLI and a known bug). Each turn is one
  `claude -p --resume <id>` call with tools Bash, Read, Edit, Write, Glob and Grep, the
  plugin loaded through `--plugin-dir`, and a fresh `CLAUDE_CONFIG_DIR` per run so no user
  settings apply. The model argument is `haiku`; the model id the API reports is recorded per call.
* **Planted facts.** Twelve facts are stated by the user at turns 2, 5, 8, ..., 35, each
  inside an ordinary request. Seven are rule-like (constraints, decisions, a rejected
  approach, a root cause), the kind context-graph pins. Five are plain facts (a host and
  port, a contact name, a ticket number, a command name, a fiscal-year date) that the pin
  rules do not match.
* **Variants.** A runs the base session with tool-output trimming on, B with it off. A
  gets all ten methods, B gets no compaction, `/compact` and the graph method.
* **Methods.** Each method starts from a copy of the finished base session (transcripts and
  graph store copied, repository reset to the end-of-session commit), so all methods see the
  identical session. `/compact` methods run `claude -p --resume <id> "/compact"`; Replace
  methods run `"/clear"`, which starts a new session that the SessionStart hook fills with
  the pack. The recency baseline is the same `/clear` with the pack replaced by the most
  recent stretch of the session that fits the same 6,000 token budget.

## Measures

* **Probe recall.** One call, forked from the post-compaction session with tools off
  (`--tools ""`), asks all twelve questions. An answer counts as correct when every regular
  expression listed for that fact in `scenario.py` matches it and it is not "unknown".
  Scoring is automatic, no model judges it. Recall by planted turn is the same score split
  by the turn the fact was stated.
* **Constraint violations.** One follow-up coding task, forked from the post-compaction
  session on a reset repository, asks for faster balances and a refund command with
  currency conversion. `bench/checks.py` then counts violations of the planted rules:
  memoizing balances (the rejected approach), any change under `legacy/`, `float` added in
  `ledger/`, a new non-standard-library import or requirements file, and non-CSV output
  from the new command. It also runs the repository's tests.
* **Tokens carried over.** The input tokens of the probe call after compaction minus the
  input tokens of the same probe in a fresh empty session. This counts whatever the method
  leaves in context (summary, preserved messages, pack) with the model's own tokenizer.
* **Tokens spent after compaction.** The input tokens of the follow-up task call: every
  input token the API processed across all its model calls (uncached, cache writes and
  cache reads, from `usage` in `--output-format json`, which sums over the calls of one turn).
* **Session tokens.** The same sum over the 36 base-session calls.
* **Compaction cost.** For `/compact`: the `preTokens` of the `compact_boundary` record in
  the transcript (the tokens the summarizer read), the call's `total_cost_usd`, its wall
  time and the summary length. The JSON `usage` of a `/compact` call is empty in Claude
  Code 2.1.306, so the cost and `preTokens` are the measures. For `/clear` methods: no model
  is called (cost 0); the time is the selection time from the selection log plus the wall
  time of the `/clear` call.
* **Offline sweep.** `analyze.py` re-runs selection on the saved event logs at budgets from
  1,000 to 12,000 tokens without any model call, and checks whether the first 60 characters
  of each planted statement appear in the pack. It also compares the knapsack's objective
  value with greedy's on the same items.
* **Offline extractor comparison.** `analyze.py --compare v1=<path>` runs the same pack
  check with the plugin code from another checkout, on the same recorded sessions.
* **Characters per token.** `token_ratio.py` sends samples as prompts and subtracts the
  input tokens of an empty prompt.

## Next measurements

Live probes and follow-up tasks for the current extractor (v2), on the main scenario and for
the graph methods on the held-out scenario:

    python3 bench/run_bench.py --runs A1 A2 A3 --parallel 3 --out results/raw/v2
    python3 bench/run_bench.py --runs H1 H2 H3 --parallel 3 --out results/raw/v2
    python3 bench/analyze.py --work /tmp/cg-bench --compare v1=<worktree of commit c6b9b26>

## Caveats

* The sessions are scripted, so the user side is identical across runs but the model's
  side is not: tool calls, outputs and replies differ from run to run, which is why each
  number is a mean over runs with its standard deviation where shown.
* Three runs per variant is a small sample. Differences of one or two facts out of twelve
  between methods are within run-to-run noise.
* Some facts can also be recovered from the repository by reading code (for example, that
  the CLI prints CSV). The probe call has no tools, so it measures only what is in context;
  the follow-up task has tools, as a real session would.
* The session model reports a 1,000,000 token context window, so these sessions (tens of
  thousands of tokens of context) are compacted on purpose, far before Claude Code would
  do it on its own.
