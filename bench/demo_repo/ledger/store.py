"""In-memory ledger."""

import datetime
from dataclasses import dataclass


@dataclass
class Entry:
    date: datetime.date
    account: str
    amount: int  # cents
    memo: str = ""


class Ledger:
    def __init__(self, entries=None):
        self.entries = list(entries or [])

    def add(self, entry):
        self.entries.append(entry)

    def get_balance(self, account):
        total = 0
        for entry in self.entries:
            if entry.account == account:
                total += entry.amount
        return total

    def accounts(self):
        return sorted({e.account for e in self.entries})
