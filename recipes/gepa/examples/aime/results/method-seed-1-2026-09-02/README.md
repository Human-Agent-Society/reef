# Reef GEPA method, seed 1 (2026-09-02)

The second seed of the method through this example, the same protocol as
[`../method-seed-0-2026-09-02/`](../method-seed-0-2026-09-02/README.md): same
models, split, seed prompt, and 150-call budget, 128 workers. A different seed
draws different parents and training problems, so this run is not expected
to walk the official seed-0 path; it is the method's second data point.

| iteration | parent, problems | outcome |
| --- | --- | --- |
| 0 | 0, [2, 9, 43] | accepted, candidate 1 (15/45) |
| 1 | 1, [33, 5, 26] | accepted, candidate 2 (14/45, not served) |
| 2 | 1, [12, 11, 39] | accepted, candidate 3 (17/45, served) |

Four candidates, 198 metric calls, no minibatch rejection.

## Result

| | validation seed | validation selected | test frozen | test selected | gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| official, seed 0 | 7/45 (15.6%) | 20/45 (44.4%) | 40/150 (26.67%) | 58/150 (38.67%) | +12.0 pp |
| this run, seed 1 | 9/45 (20.0%) | 17/45 (37.8%) | 36/150 (24.00%) | 54/150 (36.00%) | +12.0 pp |

The selected prompt carries a zero-padding rule, which the exact-containment
scorer punishes on the 45 short-answer test problems (14 to 9 there) while
the 105 three-digit problems went from 22 to 45; the net gain still equals
the official run's.

## Stored files

As for seed 0: `summary.json`, and `archive.json` with the reflection prompts
and the model's minibatch answers removed.
