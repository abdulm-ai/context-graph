import unittest

from ledger.money import parse_amount


class ParseAmountTest(unittest.TestCase):
    def test_whole_and_fraction(self):
        self.assertEqual(parse_amount("12.50"), 1250)
        self.assertEqual(parse_amount("3"), 300)
        self.assertEqual(parse_amount("0.07"), 7)

    def test_negative(self):
        self.assertEqual(parse_amount("-4.25"), -425)


if __name__ == "__main__":
    unittest.main()
