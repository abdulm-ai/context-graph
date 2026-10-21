"""Monthly reports as CSV lines."""

import calendar
import datetime


def month_bounds(month):
    """'2026-03' -> (first day, last day) of that month."""
    year, mon = (int(x) for x in month.split("-"))
    first = datetime.date(year, mon, 1)
    last = datetime.date(year, mon, calendar.monthrange(year, mon)[1])
    return first, last


def monthly_report(ledger, month):
    first, last = month_bounds(month)
    rows = [e for e in ledger.entries if first <= e.date < last]
    rows.sort(key=lambda e: (e.date, e.account))
    lines = ["date,account,amount_cents,memo"]
    for e in rows:
        lines.append(f"{e.date.isoformat()},{e.account},{e.amount},{e.memo}")
    return lines
