# Paired GEPA quickstart seed 0 (2026-09-01)

This retained result compares the pinned official GEPA quickstart with Reef's
rules-only harness evolution path. Both arms used optimizer seed 0, the same
45/45/150 AIME split, `gpt-4.1-mini-2025-04-14` for tasks,
`gpt-5-2025-08-07` for reflection, the same seed prompt, and a 150-call search
budget. The 150 held-out examples are the 30 AIME-2025 problems repeated five
times.

This is a one-seed implementation conformance result. It is not a reproduction
of the GEPA paper's larger DSPy `ChainOfThought` experiment or its absolute
benchmark score.

## Result

- The official GEPA arm improved held-out accuracy from 40/150 (26.67%) to
  58/150 (38.67%), a gain of 12 percentage points.
- The Reef rules-only arm improved from 41/150 (27.33%) to 56/150 (37.33%), a
  gain of 10 percentage points.
- Both arms selected candidate 3 from four candidates after 198 recorded metric
  calls. Official validation improved from 7/45 (15.56%) to 20/45 (44.44%);
  Reef validation improved from 11/45 (24.44%) to 18/45 (40.00%).
- Before search, the 450-rollout baseline gate measured 22.00% for the direct
  official path and 20.67% through Reef. The predeclared non-inferiority check
  passed with no failed Reef episodes.
- The official arm took 15,371.5 seconds and cost an estimated $5.458627. The
  Reef arm took 2,301.3 seconds and cost an estimated $3.642371.

The frozen scores differ by 0.67 percentage points and the selected scores by
1.33 points. Reef therefore reproduced the official quickstart's improvement
pattern closely for seed 0. Seeds 1 and 2 and the multi-node extension were not
run, so this result does not estimate across-seed variance or complete the
four-cell study.

## Baseline provenance

The baseline gate was paid under Reef commit
[`1748a3a2`](https://github.com/Human-Agent-Society/reef/commit/1748a3a2c3e9952482ad64e31a74bf82109a7d3d)
and run identity `0ffbc8e3`. It was then imported into the seed-0 optimization
run at
[`d450f0f4`](https://github.com/Human-Agent-Society/reef/commit/d450f0f4c53336f3074bb9058acdf05e3367aa61).
That intervening commit added a worker-concurrency override without changing
the model, dataset, prompt, request envelope, or sampling settings. The
baseline path used 10-way concurrency and checkpoint batches for both arms;
the 32 Reef and 16 held-out values in its run identity were inactive defaults
for optimization cells. Both seed-0 optimization arms used 128 workers. The
manifest retains both source commits, both run identities, and the original
baseline report hash.

## Incorrect-response retention fix

The completed official held-out run used upstream `ContainsAnswerEvaluator` on
AIME-2025 rows that omit `additional_context`. Correct responses were retained
normally. For incorrect responses, upstream computed the zero score and then
raised `KeyError` while constructing optional feedback, so the checkpoint kept
the correct zero but lost the response text. This affected 110 frozen and 92
selected zero-score checkpoints; it did not change either aggregate score.

Commit
[`e8094ce6`](https://github.com/Human-Agent-Society/reef/commit/e8094ce6840ac77e8b8ab25ec9d2c0ce30ca2796)
normalizes the optional context before evaluation and adds a regression test.
The paid run was not repeated; the known diagnostic limitation remains part of
this retained record.

## Stored artifacts

`manifest.json` records the immutable source, dependency, model, dataset, and
run identities; aggregate outcomes; usage and cost; and hashes for the
aggregate reports committed in this directory. These include the original
baseline result and run identity plus each optimization arm's config, summary,
and compact GEPA run log. Dataset contents, prompts produced during search,
model responses and reasoning, checkpoints, credentials, and local paths are
not committed.
