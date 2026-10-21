import datetime
import unittest

from ledger.report import monthly_report
from ledger.store import Entry, Ledger


class ReportTest(unittest.TestCase):
    def test_header_and_rows(self):
        led = Ledger([Entry(datetime.date(2026, 3, 5), "ACC-1", 100, "x")])
        self.assertEqual(monthly_report(led, "2026-03"), ["date,account,amount_cents,memo", "2026-03-05,ACC-1,100,x"])

    def test_includes_last_day(self):
        led = Ledger([Entry(datetime.date(2026, 3, 31), "ACC-1", 100, "eom")])
        self.assertEqual(len(monthly_report(led, "2026-03")), 2)


if __name__ == "__main__":
    unittest.main()
