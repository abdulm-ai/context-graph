"""Currency conversion used by the payments service. Owned by the payments team."""

RATES = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}


def to_usd(amount, currency):
    return round(amount * RATES[currency], 2)
