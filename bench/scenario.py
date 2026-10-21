"""The scripted session: work turns, planted facts, probes and the follow-up task.

Twelve facts are planted at known turns (every third turn from 2 to 35), so
recall can be read by position, as in "Lost in the Middle" (Liu et al. 2024).
Seven are the kind context-graph pins (constraints, decisions, a rejected
approach, a root cause). Five are plain facts the pin rules do not match, so
they survive only if the graph and the budget keep them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Fact:
    id: str
    turn: int
    kind: str  # constraint | decision | rejected | root_cause | fact
    question: str
    expect: list[str]  # every regex must match the answer (case-insensitive)


FACTS = [
    Fact("F1", 2, "constraint", "What is the rule about how money amounts are represented in this codebase?",
         [r"cent", r"float|integer|int\b"]),
    Fact("F2", 5, "fact", "Which host and port does the production ledger database live on?",
         [r"ledger-db-07", r"6432"]),
    Fact("F3", 8, "decision", "What storage backend did we decide on, and why was the alternative rejected?",
         [r"sqlite", r"concurren|corrupt"]),
    Fact("F4", 11, "constraint", "Which directory must we not modify, and which team owns it?",
         [r"legacy", r"payments"]),
    Fact("F5", 14, "fact", "Who is the finance contact for report questions?", [r"priya"]),
    Fact("F6", 17, "root_cause", "What was the root cause of the missing entries in the monthly report?",
         [r"last day|31", r"<|less than|exclusive|compar|bound|exclud|drop"]),
    Fact("F7", 20, "rejected", "Which approach to speeding up get_balance was tried and rejected, and why?",
         [r"lru_cache|cach|memoi", r"stale"]),
    Fact("F8", 23, "fact", "What is the ticket number for the report work?", [r"FIN-2291"]),
    Fact("F9", 26, "decision", "What name did we choose for the new export command?", [r"export-csv"]),
    Fact("F10", 29, "constraint", "What format must all CLI output use?", [r"csv", r"header"]),
    Fact("F11", 32, "fact", "On what date does our fiscal year start?", [r"april\s*1|1\s*april|apr(il)?[ -]0?1"]),
    Fact("F12", 35, "constraint", "What is the rule about adding dependencies?",
         [r"standard library|stdlib|third[- ]party|no (new |external )?dependenc"]),
]

PLANTS = {
    2: "Rule for this codebase: never use float for money. Amounts are integer cents end to end.",
    5: "For context, the production ledger database lives on the host ledger-db-07.internal, port 6432.",
    8: ("Decision: we'll store ledgers in SQLite through the sqlite3 module. The JSON-file storage idea is "
        "rejected because it corrupted data on concurrent writes."),
    11: "Don't touch anything under legacy/. The payments team owns it and their CI breaks when we change it.",
    14: "By the way, the finance contact for report questions is Priya Raman.",
    17: ("I traced the missing March entries. The root cause is that monthly_report compares entry dates with < "
         "against the last day of the month, so entries dated on the last day are dropped."),
    20: ("We tried caching get_balance with functools.lru_cache last month and it returned stale balances after "
         "add(). Do not use lru_cache or any memoization for balances."),
    23: "This work is tracked in the ticket FIN-2291.",
    26: "We'll call the new export command export-csv, not dump.",
    29: "All CLI output must stay CSV with a header row. The finance team parses it with a script.",
    32: "Our fiscal year starts on April 1, so Q1 runs from April to June.",
    35: "Don't add third-party dependencies to this project; standard library only.",
}

WORK = {
    1: "Look around this repository and tell me briefly what it does. Then run the test suite.",
    2: "Add a format_cents function to ledger/money.py that renders 1050 as '10.50' and -5 as '-0.05', with tests.",
    3: "Run the full test suite. Leave the report failure alone for now and tell me which tests fail.",
    4: "Add input validation to parse_amount: reject more than two decimal places and non-numeric input with "
       "ValueError. Add tests.",
    5: "Show me the git log and a diff stat of the working tree.",
    6: "Show me the full contents of ledger/cli.py and ledger/jsonfile.py.",
    7: "Add an accounts command to the CLI that prints every account name, one per line after a header row.",
    8: "Implement a SQLite-backed store in ledger/sqlite_store.py with add, entries and get_balance, plus tests "
       "using an in-memory database.",
    9: "Run the tests verbosely with python3 -m unittest discover -s tests -v and show me the output.",
    10: "Write docstrings for every public function and class in ledger/store.py and ledger/sqlite_store.py.",
    11: "Grep the codebase for every place that reads or writes amounts and list them with line numbers.",
    12: "Add a --db option check: if the --db path's directory does not exist, exit with a clear error message.",
    13: "Print every Python file under ledger/ with cat so I can review the whole package.",
    14: "Run the tests again and tell me the current state.",
    15: "Add type hints to ledger/money.py and ledger/report.py.",
    16: "Show me the report output for a small sample: add three entries for March 2026 to a temp db with the CLI "
        "and run the report command.",
    17: "Fix that bug and add a regression test for the last day of the month.",
    18: "Refactor ledger/report.py so filtering and formatting are separate functions. Keep the tests green.",
    19: "Add validation for the report month argument: it must look like YYYY-MM, otherwise exit with an error.",
    20: "Run python3 -m unittest discover -s tests -v and summarize the results.",
    21: "Add a test for get_balance with many accounts and many entries.",
    22: "List every TODO or FIXME comment in the repository.",
    23: "Add a memo length limit of 200 characters in the CLI add command, with a test.",
    24: "Show me git diff --stat and then the full diff of ledger/report.py.",
    25: "Write a short CONTRIBUTING.md with how to run the tests.",
    26: "Implement that export command: it writes every entry as CSV to stdout, oldest first.",
    27: "Add tests for the export command.",
    28: "Write scripts/gen_sample.py that creates a sample ledger with 300 entries across 5 accounts in "
        "March 2026, run it, then run the report for 2026-03 on that db and show me the output.",
    29: "Make the report command print a total line at the end.",
    30: "Review ledger/cli.py for error handling problems and fix the worst one.",
    31: "Run all tests and show me the full output.",
    32: "Add a quarter option to the report command that reports a whole quarter.",
    33: "Add tests for the quarter option.",
    34: "Show me the whole current ledger/cli.py.",
    35: "Run the full test suite one more time.",
    36: "Next I want to work on balance performance and refunds. Before that, summarize the CLI commands we have "
        "now in a short list.",
}

N_TURNS = 36


def turn_prompt(turn: int) -> str:
    plant = PLANTS.get(turn)
    work = WORK[turn]
    return f"{plant} {work}" if plant else work


PROBE_INSTRUCTIONS = (
    "Answer these questions from what you know about this session so far. Do not use any tools. "
    "If you do not know an answer, write unknown. Answer each on one line, starting with its number, "
    "for example 'Q1: ...'."
)


def probe_prompt() -> str:
    qs = "\n".join(f"Q{i}: {f.question}" for i, f in enumerate(FACTS, 1))
    return f"{PROBE_INSTRUCTIONS}\n\n{qs}"


def score_probe(answer: str) -> dict[str, bool]:
    lines: dict[int, str] = {}
    current = None
    for raw in answer.splitlines():
        m = re.match(r"^\s*\**\s*Q(\d+)\s*\**\s*[:.)-]\s*(.*)", raw)
        if m:
            current = int(m.group(1))
            lines[current] = m.group(2)
        elif current is not None:
            lines[current] += " " + raw.strip()
    out = {}
    for i, f in enumerate(FACTS, 1):
        text = lines.get(i, "")
        unknown = bool(re.fullmatch(r"\W*unknown\W*", text.strip(), re.I))
        out[f.id] = (not unknown) and all(re.search(p, text, re.I) for p in f.expect)
    return out


FOLLOWUP = (
    "New task. get_balance is too slow on big ledgers, so make balance lookups fast. Then add a refund command "
    "to the CLI, used like `python3 -m ledger.cli --db DB refund ACCOUNT AMOUNT --currency EUR`, which records "
    "a refund of that amount converted to USD with the rates in legacy/fx.py; that conversion rounds badly, so "
    "make refunds convert correctly. Keep all tests passing and add tests for the new parts."
)
