# Reef GEPA method, seed 0 (2026-09-02)

The method under `recipes/gepa/` run through this example at seed 0, against
the official quickstart record in
[`../quickstart-seed-0-2026-09-01/`](../quickstart-seed-0-2026-09-01/README.md).
Same models (`gpt-4.1-mini-2025-04-14` for tasks, `gpt-5-2025-08-07` for
reflection), same 45/45/150 AIME split, same seed prompt, same 150-call
budget, 128 workers; 37 minutes end to end.

## The search walked the official run's path

The archive plans each iteration from one seeded generator in upstream's
order, so the parent it reflected from and the training problems it showed
the reflection model are the official run's own:

| iteration | official parent, problems | this run | outcome here |
| --- | --- | --- | --- |
| 0 | 0, [1, 27, 35] | 0, [1, 27, 35] | accepted, candidate 1 |
| 1 | 0, [0, 23, 37] | 0, [0, 23, 37] | accepted, candidate 2 |
| 2 | 2, [14, 12, 7] | 2, [14, 12, 7] | accepted, candidate 3 |

Both runs produced four candidates and stopped at 198 metric calls. The
official run served candidate 3; here candidate 3 validated below candidate 2,
so candidate 2 stayed served.

## Result

| | validation seed | validation selected | test frozen | test selected | gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| official | 7/45 (15.6%) | 20/45 (44.4%) | 40/150 (26.67%) | 58/150 (38.67%) | +12.0 pp |
| this run | 12/45 (26.7%) | 18/45 (40.0%) | 40/150 (26.67%) | 70/150 (46.67%) | +20.0 pp |

The frozen test score is identical to the official one. On the 45 test
problems with answers under 100 the selected prompt scored 28 (official 18);
on the 105 three-digit problems it scored 42 (official 40). It carries no
zero-padding rule; candidates 1 and 3, which do, lost on validation.

## Stored files

`summary.json` is the driver's own summary. `archive.json` is the method's
archive with the reflection prompts and the model's minibatch answers
removed: every candidate's text, parent, validation vector, and minibatch
scores; the per-iteration plans; each reflection's scores and verdict; the
metric-call count.
