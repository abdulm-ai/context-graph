"""First-order decay with rehearsal.

dN/dt = -lambda N, so N(t) = exp(-lambda t), with t counted in user turns since
the atom was last used and lambda = ln 2 / half_life. Using an atom again (a
later atom links to it by file, symbol or error-fix) resets t to zero, the
rehearsal effect in Ebbinghaus's forgetting curve, as used for agent memory in
Generative Agents (Park et al. 2023) and MemoryBank (Zhong et al. 2024).
"""

from __future__ import annotations

import math


def decay(now_turn: int, last_used: int, half_life_turns: float) -> float:
    if half_life_turns <= 0:
        return 1.0
    age = max(0, now_turn - last_used)
    return math.exp(-math.log(2) / half_life_turns * age)
