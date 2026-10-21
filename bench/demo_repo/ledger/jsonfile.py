"""Load and save a ledger as a JSON file."""

import datetime
import json
import os

from .store import Entry, Ledger


def load(path):
    if not os.path.exists(path):
        return Ledger()
    with open(path) as fh:
        raw = json.load(fh)
    return Ledger(
        Entry(datetime.date.fromisoformat(r["date"]), r["account"], int(r["amount"]), r.get("memo", ""))
        for r in raw
    )


def save(ledger, path):
    rows = [
        {"date": e.date.isoformat(), "account": e.account, "amount": e.amount, "memo": e.memo}
        for e in ledger.entries
    ]
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)
