"""Token estimate used for the budget.

The plugin is standard library only, so it cannot run the model's tokenizer.
It estimates tokens from characters. The default ratio is deliberately on the
low side (more tokens per character than plain English) because session text
is code, paths and command output. The benchmark measures the real number of
carried-over tokens from API usage and reports the estimate's error.
"""

from __future__ import annotations

import math

DEFAULT_CHARS_PER_TOKEN = 3.5


def estimate(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))
