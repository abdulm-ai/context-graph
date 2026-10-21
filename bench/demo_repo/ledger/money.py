"""Money helpers. Amounts are integer cents."""


def parse_amount(text):
    """Parse a decimal string such as '12.50' or '-3' into cents."""
    text = text.strip()
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    whole, _, frac = text.partition(".")
    frac = (frac + "00")[:2]
    cents = int(whole or "0") * 100 + int(frac)
    return -cents if negative else cents
