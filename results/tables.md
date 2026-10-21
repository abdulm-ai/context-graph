Model(s): claude-haiku-5-5. Base sessions per variant: 3. Total API cost of every recorded call: $27.83.

## Main scenario (variant A), extractor v1 (before pinning user-stated facts)

| Method | Runs | Probe recall | Rule-like facts | Plain facts | Carried-over tokens | Follow-up input tokens | Constraint violations (total) | Follow-up tests pass |
|---|---|---|---|---|---|---|---|---|
| no compaction | 3 (3 follow-up) | 100% (sd 0%) | 100% | 100% | 93,787 | 1,162,041 | 0 | 100% |
| /compact | 3 (3 follow-up) | 100% (sd 0%) | 100% | 100% | 10,877 | 491,490 | 1 | 100% |
| /compact + instructions | 3 (3 follow-up) | 100% (sd 0%) | 100% | 100% | 12,497 | 535,708 | 0 | 100% |
| /compact + pins (Assist) | 3 (3 follow-up) | 100% (sd 0%) | 100% | 100% | 14,202 | 499,224 | 0 | 100% |
| /clear + recency truncation | 3 (3 follow-up) | 44% (sd 5%) | 50% | 33% | 6,039 | 845,704 | 0 | 100% |
| /clear + graph, shortest path, knapsack | 3 (3 follow-up) | 92% (sd 0%) | 100% | 75% | 5,755 | 943,321 | 0 | 100% |
| /clear + graph, diffusion, knapsack | 3 (3 follow-up) | 89% (sd 5%) | 100% | 67% | 5,349 | 793,925 | 0 | 100% |
| /clear + graph, shortest path, greedy | 3 (2 follow-up) | 92% (sd 0%) | 100% | 75% | 5,766 | 865,142 | 0 | 100% |
| /clear + graph, 2k budget | 2 (1 follow-up) | 71% (sd 6%) | 94% | 25% | 1,752 | 761,112 | 1 | 100% |
| /clear + graph, no user-prompt share | 1 (0 follow-up) | 75% (sd 0%) | 100% | 25% | 5,966 | n/a | 0 | n/a |

## Held-out facts (variant H), extractor v2

| Method | Runs | Probe recall | Rule-like facts | Plain facts | Carried-over tokens | Follow-up input tokens | Constraint violations (total) | Follow-up tests pass |
|---|---|---|---|---|---|---|---|---|
| no compaction | 3 (3 follow-up) | 100% (sd 0%) | 100% | 100% | 84,924 | 1,029,934 | 0 | 100% |
| /compact | 3 (3 follow-up) | 75% (sd 43%) | 75% | 75% | 11,288 | 749,492 | 0 | 100% |
| /compact + instructions | 3 (2 follow-up) | 100% (sd 0%) | 100% | 100% | 11,504 | 358,436 | 0 | 100% |
| /compact + pins (Assist) | 2 (0 follow-up) | 100% (sd 0%) | 100% | 100% | 11,666 | n/a | 0 | n/a |

## Cost of the compaction step itself (variant A, round v1)

| Method | Model input tokens | Cost (USD) | Wall time (s) | Time inside (ms) | Summary or pack size |
|---|---|---|---|---|---|
| /compact | 114,587 | 0.1910 | 24.8 | 22,080 | 12,743 chars |
| /compact + instructions | 114,587 | 0.1954 | 31.9 | 29,180 | 15,275 chars |
| /compact + pins (Assist) | 114,587 | 0.2443 | 34.7 | 31,435 | 15,487 chars |
| /clear + recency truncation | 0 | 0.0000 | 3.0 | 75 | 5,936 tokens (est.) |
| /clear + graph, shortest path, knapsack | 0 | 0.0000 | 3.0 | 90 | 5,959 tokens (est.) |
| /clear + graph, diffusion, knapsack | 0 | 0.0000 | 3.3 | 121 | 5,963 tokens (est.) |
| /clear + graph, shortest path, greedy | 0 | 0.0000 | 2.8 | 121 | 5,957 tokens (est.) |
| /clear + graph, 2k budget | 0 | 0.0000 | 3.0 | 105 | 1,980 tokens (est.) |
| /clear + graph, no user-prompt share | 0 | 0.0000 | 3.2 | 84 | 5,999 tokens (est.) |

## Violations by rule (variant A, round v1, count over follow-up runs)

| Method | lru_cache | legacy | float | dependency | csv |
|---|---|---|---|---|---|
| no compaction | 0 | 0 | 0 | 0 | 0 |
| /compact | 0 | 0 | 1 | 0 | 0 |
| /compact + instructions | 0 | 0 | 0 | 0 | 0 |
| /compact + pins (Assist) | 0 | 0 | 0 | 0 | 0 |
| /clear + recency truncation | 0 | 0 | 0 | 0 | 0 |
| /clear + graph, shortest path, knapsack | 0 | 0 | 0 | 0 | 0 |
| /clear + graph, diffusion, knapsack | 0 | 0 | 0 | 0 | 0 |
| /clear + graph, shortest path, greedy | 0 | 0 | 0 | 0 | 0 |
| /clear + graph, 2k budget | 0 | 0 | 1 | 0 | 0 |
| /clear + graph, no user-prompt share | 0 | 0 | 0 | 0 | 0 |

## Recall by the turn the fact was planted (variant A, round v1)

| Method | F1 t2 | F2 t5 | F3 t8 | F4 t11 | F5 t14 | F6 t17 | F7 t20 | F8 t23 | F9 t26 | F10 t29 | F11 t32 | F12 t35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no compaction | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| /compact | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| /compact + instructions | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| /compact + pins (Assist) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| /clear + recency truncation | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 33% | 100% | 100% | 100% | 100% |
| /clear + graph, shortest path, knapsack | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| /clear + graph, diffusion, knapsack | 100% | 0% | 100% | 100% | 67% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| /clear + graph, shortest path, greedy | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| /clear + graph, 2k budget | 100% | 0% | 100% | 100% | 0% | 100% | 100% | 0% | 50% | 100% | 100% | 100% |
| /clear + graph, no user-prompt share | 100% | 0% | 100% | 100% | 0% | 100% | 100% | 0% | 100% | 100% | 100% | 100% |

## Recall by planted turn, held-out facts (variant H, extractor v2)

| Method | F1 t2 | F2 t5 | F3 t8 | F4 t11 | F5 t14 | F6 t17 | F7 t20 | F8 t23 | F9 t26 | F10 t29 | F11 t32 | F12 t35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no compaction | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| /compact | 100% | 67% | 67% | 67% | 67% | 67% | 67% | 67% | 100% | 67% | 100% | 67% |
| /compact + instructions | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| /compact + pins (Assist) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

## Trimming ablation (base sessions)

| Variant | Session input tokens | Context at end of session | Session cost (USD) |
|---|---|---|---|
| A: trimming on | 5,399,239 (sd 958,474) | 93,787 | 1.781 |
| B: trimming off | 4,630,680 (sd 311,830) | 89,522 | 1.446 |

| Method | Recall, trimming on (A) | Recall, trimming off (B) | Follow-up tokens A | Follow-up tokens B |
|---|---|---|---|---|
| no compaction | 100% | 100% | 1,162,041 | 1,045,615 |
| /compact | 100% | 100% | 491,490 | 581,839 |
| /clear + graph, shortest path, knapsack | 92% | 89% | 943,321 | 1,013,261 |

## Offline: planted facts present in the pack, by budget (variant A, no model call)

| Selector | 1,000 | 2,000 | 3,000 | 4,000 | 6,000 | 8,000 | 12,000 |
|---|---|---|---|---|---|---|---|
| graph, shortest path | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| graph, diffusion | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| graph, greedy | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| graph, no user-prompt share | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| recency truncation | 8% | 8% | 17% | 17% | 25% | 39% | 64% |

Greedy reached 99.21% to 100.00% of the knapsack's objective value (mean 99.89%) over 17 run x budget cases.

Selection at 6,000 tokens: graph build 3.3 ms, relevance 1.3 ms, knapsack 4.6 ms, for 255 atoms, 1,124 edges, 1,054 pinned tokens.

## Offline: planted facts present in the pack, by budget (variant H, no model call)

| Selector | 1,000 | 2,000 | 3,000 | 4,000 | 6,000 | 8,000 | 12,000 |
|---|---|---|---|---|---|---|---|
| graph, shortest path | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| graph, diffusion | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| graph, greedy | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| graph, no user-prompt share | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| recency truncation | 8% | 8% | 17% | 17% | 31% | 42% | 67% |

Greedy reached 94.40% to 100.00% of the knapsack's objective value (mean 99.47%) over 18 run x budget cases.

Selection at 6,000 tokens: graph build 5.8 ms, relevance 1.2 ms, knapsack 4.8 ms, for 239 atoms, 1,023 edges, 991 pinned tokens.

## Offline: extractor versions on the same recorded sessions (graph, shortest path, knapsack)

Share of the twelve planted facts whose statement is in the pack, every sentence word for word. No model call; deterministic given the recorded event logs.

| Extractor | Scenario | 2,000 | 4,000 | 6,000 | 8,000 |
|---|---|---|---|---|---|
| v2 (this checkout) | main (A) | 100% | 100% | 100% | 100% |
| v2 (this checkout) | held-out (H) | 100% | 100% | 100% | 100% |
| v1 | main (A) | 53% | 67% | 92% | 92% |
| v1 | held-out (H) | 44% | 69% | 89% | 86% |

Per fact at 6,000 tokens (share of runs whose pack contains it):

| Extractor | Scenario | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 (this checkout) | A | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| v2 (this checkout) | H | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| v1 | A | 100% | 0% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| v1 | H | 100% | 0% | 100% | 67% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
