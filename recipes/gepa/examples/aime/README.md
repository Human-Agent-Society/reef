# GEPA on AIME: validating the method

[GEPA](https://github.com/gepa-ai/gepa) is reflective prompt evolution: keep an
archive of candidate prompts, Pareto-sample a parent from it, evaluate that
parent on a small training minibatch, ask a stronger model to rewrite one
component in light of what went wrong, keep the child only if it beats its
parent on that minibatch, then score it on the whole validation set and serve
the best. `recipes/gepa/` is that algorithm written as a Reef harness-evolution
method - `propose` and `selection` under the `evolution:` contract, with the
archive as the method's own state on disk - and this example is what validates
it. The benchmark is the upstream AIME quickstart: 45 training problems, 45
validation problems, a sealed 150-problem AIME-2025 test split, a 150-call
search budget, seed 0. Nothing here imports `gepa`; the two places the example
reproduces upstream text (the scorer's feedback wording and the epoch-shuffled
minibatch order) say so in `harness/aime.py`.

## What maps to what

| GEPA | Reef |
| --- | --- |
| Candidate prompt | a composition of harness nodes; the evolvable one here is `rules`, Pi's `AGENTS.md` |
| Pareto-sample a parent from the archive | `Archive.select_parent` over per-problem fronts, dominated candidates pruned |
| Evaluate the parent on a train minibatch, with traces | the served composition's own recorded traffic: the driver runs the minibatch through the service, so `propose` gets the transcripts free |
| Reflect on one component and propose a rewrite | `models["reflection"]` (gpt-5) with GEPA's own prompt, over `Inputs` / `Generated Outputs` / `Feedback` records |
| Accept the child iff it beats the parent on the minibatch | the proposer runs its own episodes and returns `None` on a reject, which skips the step |
| Full validation pass, then Pareto update | the mechanism's `evolution.tasks` is the validation set; `GEPASelector.decide` reads the per-task scores it produced |
| Serve the argmax-mean candidate | select on a strict mean improvement over the served composition, which publishes the tree for `GET /reef/harness` |

The request envelope is reproduced by seed nodes rather than by a custom Pi
command line: a `config` node writes `defaultTools: []` into `settings.json`,
and a fixed `code_extension` makes the rendered rules text Pi's entire system
prompt at `before_agent_start` and flattens a single-part user message to the
plain string upstream sends. Both are non-evolvable. The extension reads
`AGENTS.md` at startup, so the rules node underneath it can evolve freely.

## Setup and run

```bash
pip install -e .                       # datasets, reef-client
REEF_PI_BINARY=/path/to/pi ./run.sh --dry-run
```

A dry run verifies the Pi binary is `0.84.2`, loads and hash-checks the pinned
splits, boots the recipe from `gepa.yaml` with the validation set filled in,
and prints the plan. It makes no model call. Git LFS is not needed: the
embedded service keeps its artifacts in memory under `./work`.

```bash
OPENAI_API_KEY=... REEF_PI_BINARY=/path/to/pi ./run.sh
```

A live run is roughly 150 search episodes plus the mechanism's validation
passes, then 300 test episodes. Episodes are independent, so `REEF_GEPA_WORKERS`
(default 128) run at once everywhere: the mechanism's validation pass through
`evolution.episode_workers`, the driver's minibatch, and the test passes. It
changes wall time only. `--budget` lowers the search budget for a
shorter run. `REEF_GEPA_MULTI=1` adds an `aime-solver` skill node to the seed
and evolves it alongside the rules node, which is the extension case, not the
comparison. The credential is read once, handed to the embedded service as its
upstream, and never written into a candidate, a checkpoint, or a published
tree; the Pi episodes authenticate to the local service with a placeholder.

One round is: pull the served tree, draw the next epoch-shuffled minibatch of
three training problems, run each as a Pi episode against that tree with a
model binding pointed back at the embedded service, score it, find its recorded
request by problem text, and report the score against it. The third report
closes the batch (`data.batch_size: 3`) and schedules the training step, which
is where the method proposes, evaluates, and publishes. Rounds stop when the
archive reaches the metric-call budget.

## What a run retains

Everything under `./work`, and a rerun resumes from it:

- `gepa/<scenario>.json` - the archive: candidates and their texts, parents,
  per-problem validation vectors, fronts, the round-robin cursors, and the
  metric-call count.
- `rounds.json` - the driver's round counter, so a resumed run continues the
  training order instead of redrawing minibatches it already paid for.
- `heldout/{frozen,selected}/example-NNNN.json` - one checkpoint per test
  episode, keyed by the problem and the composition; a checkpoint written for
  a different composition is refused, never folded into the score.
- `reef-data/`, `artifacts/` - the scenario's records and its release chain.
- `summary.json` - validation seed and selected means from the archive, both
  test scores, the candidate count, the metric calls, and the retained official
  numbers beside them.

## The validation contract

The target is the retained record in
[`results/quickstart-seed-0-2026-09-01/`](results/quickstart-seed-0-2026-09-01/README.md):
the official GEPA quickstart at seed 0 improved sealed AIME-2025 accuracy from
**26.67%** (40/150) to **38.67%** (58/150), selecting candidate 3 of 4 after
198 metric calls, with validation moving from 15.56% to 44.44%. A method run
that lands in that neighbourhood - a frozen score near 27% and a selected score
near 38%, with a handful of candidates and a strictly improving validation mean
- has reproduced the quickstart. One seed of a stochastic search is a
conformance check, not an estimate of variance; seeds 1 and 2 remain follow-up
work.

The deterministic half of the contract is `tests/test_gepa_aime_harness.py`,
which runs with no model and no Pi binary: the scorer, the feedback wording,
the dataset drift refusal, the minibatch order against a sequence computed from
upstream, and the driver's record lookup and report against a real embedded
service with a stubbed inference backend.

## Deviations

- The mechanism re-evaluates the served composition on the validation set at
  every evaluated step, so one accepted proposal costs `2 x 45` episodes where
  upstream GEPA pays 45. The archive counts the method's own metric calls, so
  the budget still means what it means upstream; the wall-clock and spend do
  not compare directly.
- The training minibatch is real recorded traffic through the service rather
  than a direct evaluation call, which is the point of the exercise but means
  a minibatch problem is scored by the driver and re-read by the method from
  the transcript, not scored twice.
- `evolution.tasks` is data, not configuration, so `gepa.yaml` ships it empty
  and `run.py` fills it from the pinned split before `build_recipe`.
