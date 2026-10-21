# ledger

A small double-entry style ledger with a CSV command line. Demo project for the
context-graph benchmark.

    python3 -m ledger.cli --db ledger.json add ACC-1 12.50 --memo "coffee"
    python3 -m ledger.cli --db ledger.json balance ACC-1
    python3 -m ledger.cli --db ledger.json report 2026-03

Run the tests with `python3 -m unittest discover -s tests`.
