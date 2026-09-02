# Upstream GEPA, seeds 0 and 1 (2026-09-02)

The upstream optimizer run fresh on the same day as the method's own runs, so
the two arms can be compared without a month of drift between them. Both used
`gepa.optimize` from the pinned release, `gpt-4.1-mini-2025-04-14` for tasks,
`gpt-5-2025-08-07` for reflection, the pinned 45/45/150 AIME split, the same
seed prompt, a 150-call budget, and 128 workers. Scores are on the sealed
150-problem AIME-2025 split.

| seed | frozen | selected | improvement | candidates / calls |
| ---: | ---: | ---: | ---: | --- |
| 0 | 31.33% (47/150) | 42.67% (64/150) | +11.33 pp | 4 / 198 |
| 1 | 26.67% (40/150) | 40.67% (61/150) | +14.00 pp | 4 / 198 |

Read these beside `../method-seed-0-2026-09-02/` and `../method-seed-1-2026-09-02/`.

Two things worth noting. The gains here, +11.33 and +14.00 points, sit close to
the +12.00 of the older retained run in `../quickstart-seed-0-2026-09-01/`, so
what the upstream arm achieves is stable. The frozen column is not: the same
seed prompt on the same problems scored 47/150 at seed 0 and 40/150 at seed 1,
which is the run-to-run noise of the task model, measured directly.

Each file here is the driver's own summary with the model responses removed,
plus the configuration the run booted from.
