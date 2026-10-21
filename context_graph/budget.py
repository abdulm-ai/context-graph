"""Fit atoms into the token budget.

Exact 0/1 knapsack by dynamic programming: maximise total value subject to
total tokens <= capacity. Token counts are rounded up to buckets (10 tokens by
default) so the table has capacity/bucket columns; with a 6,000 token budget
and a few hundred atoms that is well under a second in pure Python.

Greedy by value density (value / tokens) is the usual fast alternative. It is
optimal for the fractional problem but not for 0/1, and the benchmark measures
how much that costs here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Item:
    id: str
    value: float
    tokens: int


def _buckets(tokens: int, bucket: int) -> int:
    return max(1, math.ceil(tokens / bucket))


def knapsack(items: list[Item], capacity_tokens: int, bucket: int = 10) -> list[str]:
    cap = capacity_tokens // bucket
    if cap <= 0:
        return []
    cand = [it for it in items if it.value > 0 and _buckets(it.tokens, bucket) <= cap]
    n = len(cand)
    best = [0.0] * (cap + 1)
    take = [bytearray(cap + 1) for _ in range(n)]
    for i, it in enumerate(cand):
        w = _buckets(it.tokens, bucket)
        v = it.value
        row = take[i]
        for c in range(cap, w - 1, -1):
            alt = best[c - w] + v
            if alt > best[c]:
                best[c] = alt
                row[c] = 1
    chosen = []
    c = cap
    for i in range(n - 1, -1, -1):
        if take[i][c]:
            chosen.append(cand[i].id)
            c -= _buckets(cand[i].tokens, bucket)
    chosen.reverse()
    return chosen


def greedy(items: list[Item], capacity_tokens: int, bucket: int = 10) -> list[str]:
    cap = capacity_tokens // bucket
    order = sorted(
        (it for it in items if it.value > 0),
        key=lambda it: it.value / _buckets(it.tokens, bucket),
        reverse=True,
    )
    chosen, used = [], 0
    for it in order:
        w = _buckets(it.tokens, bucket)
        if used + w <= cap:
            chosen.append(it.id)
            used += w
    return chosen


def total_value(items: list[Item], chosen: list[str]) -> float:
    pick = set(chosen)
    return sum(it.value for it in items if it.id in pick)
