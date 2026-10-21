"""Token estimate used for the budget.

The plugin is standard library only, so it cannot run the model's tokenizer.
It estimates tokens from characters. The default ratio, 2.2 characters per
token, was measured with API usage counts on claude-haiku-5-5 for a rendered
pack (2.21), Python source (2.37) and English prose (2.10); see
results/README.md. Set chars_per_token in the config for another model.
"""

from __future__ import annotations

import math

DEFAULT_CHARS_PER_TOKEN = 2.2


def estimate(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))
