# Seed 355: retained native Agent comparison

This is the first full run of the fixed protocol, after small plumbing preflights. It used `deepseek-flash` with an explicitly mutable provider alias. The experiment ran on upstream base `ebe8eb2bee19f7665ce759c8fa8d53de5f5b2b75` plus the feature source hashes in `summary.json`; the unrelated upstream task-generation change was incorporated afterward. See [the protocol](../../README.md) for split construction, selection policy and limitations.

| Retained version | State tracking | Logical deduction |
| --- | ---: | ---: |
| Initial baseline | 23 / 24 | 22 / 24 |
| After state-tracking feedback | 24 / 24 | 22 / 24 |
| After logical-deduction feedback | 24 / 24 | 24 / 24 |

Each column is **12 distinct held-out tasks × 2 repeats**, not 24 independent tasks. Both model-generated updates passed the current-family validation gate. The first update's logical-deduction total stayed 22/24, while two task means improved and two regressed. Aggregate accuracy therefore hid task-level turnover. This observation does not by itself distinguish prompt effects from provider variation.

For the final release versus baseline, task-cluster bootstrap intervals were `[0, 12.5]` percentage points for state tracking and `[0, 20.83]` for logical deduction. Both contain zero; this run does not establish a statistically significant learning gain. There is one split seed, no immutable provider revision, and possible public-benchmark pretraining contamination.

The run completed 144/144 held-out episodes with 22 tool calls. Total provider attempts were 214 (166 held-out, 41 optimization/validation, 7 changed-suite), with 178,464 prompt tokens and 59,087 completion tokens reported. No provider failure occurred. Three completed checkpoints were reused byte-for-byte; evaluation left the served head and scenario commit history unchanged. Currency cost was not inferred.

Files:

- `summary.json`: release map, family comparisons, source/data hashes and phase usage.
- `case-results.jsonl`: all 144 per-case outcomes, usage and retained-observation hashes.
- `updates.json`: the two actual reflection responses and validation scores.
- `protocol.json`: settings fixed before model calls.
- `report.md`: the complete comparison report.

The raw private run also contains task text, reflection inputs, traces, local storage and provider response counters. It is not published here. Checksums make the reviewed raw export identifiable; rerun the provided driver to produce your own inspectable trajectories. This committed bundle claims measurement and reproducibility support, not a new RSI algorithm or a full BBH benchmark result.
