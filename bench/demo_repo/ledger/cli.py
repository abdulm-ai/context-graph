"""Command line: add, balance, report. Output is CSV."""

import argparse
import datetime
import sys

from . import jsonfile
from .money import parse_amount
from .report import monthly_report
from .store import Entry


def main(argv=None):
    p = argparse.ArgumentParser(prog="ledger")
    p.add_argument("--db", default="ledger.json")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("account")
    a.add_argument("amount")
    a.add_argument("--memo", default="")
    a.add_argument("--date", default=None)
    b = sub.add_parser("balance")
    b.add_argument("account")
    r = sub.add_parser("report")
    r.add_argument("month")
    args = p.parse_args(argv)

    ledger = jsonfile.load(args.db)
    if args.cmd == "add":
        date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()
        entry = Entry(date, args.account, parse_amount(args.amount), args.memo)
        ledger.add(entry)
        jsonfile.save(ledger, args.db)
        print("date,account,amount_cents,memo")
        print(f"{entry.date.isoformat()},{entry.account},{entry.amount},{entry.memo}")
    elif args.cmd == "balance":
        print("account,balance_cents")
        print(f"{args.account},{ledger.get_balance(args.account)}")
    elif args.cmd == "report":
        for line in monthly_report(ledger, args.month):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
