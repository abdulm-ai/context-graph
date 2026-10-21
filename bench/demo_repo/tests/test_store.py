import datetime
import unittest

from ledger.store import Entry, Ledger

D = datetime.date(2026, 3, 2)


class LedgerTest(unittest.TestCase):
    def test_balance(self):
        led = Ledger([Entry(D, "ACC-1", 1000), Entry(D, "ACC-2", 50), Entry(D, "ACC-1", -250)])
        self.assertEqual(led.get_balance("ACC-1"), 750)
        self.assertEqual(led.get_balance("ACC-9"), 0)

    def test_accounts(self):
        led = Ledger([Entry(D, "B", 1), Entry(D, "A", 1), Entry(D, "B", 1)])
        self.assertEqual(led.accounts(), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
