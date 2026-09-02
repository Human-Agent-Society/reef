# GEPA through Reef harness evolution

Upstream [GEPA](https://github.com/gepa-ai/gepa) is the search engine and
Reef is the harness runtime. GEPA proposes named text components and keeps
candidate ancestry and Pareto frontiers; Reef maps each component to a
declarative node, renders the whole composition for Pi, runs isolated
episodes, and publishes the selected tree as a release. The experiment is
the upstream AIME quickstart run four ways:

1. `reference` - the pinned upstream quickstart, unchanged;
2. `frozen` - the Reef seed composition scored on the test split, no search;
3. `rules` - one Reef rules node that Pi receives as its exact system prompt,
   the single-component conformance arm; and
4. `multi` - a fixed-topology composition whose rules and skill nodes GEPA
   evolves independently, in Pi's complete harness.

The same prompt is GEPA's `system_prompt` component in the direct arm and
Reef's `rules` node in the Pi arms. In the rules arm Pi runs with tools,
skills, prompt templates, and context-file discovery disabled, and an
extension loaded only for that arm makes the rendered text Pi's final system
prompt and flattens the single text user message to the upstream request
shape.

Before any search cell, a baseline gate scores that seed prompt through both
paths - ten repetitions of the 45 validation problems per arm, provider
default sampling, the upstream scorer, the same output cap - and stops the
run unless a one-sided 95% lower bound shows Reef within ten points of the
direct path with no failed Reef episode. A broken bridge therefore fails
before it can spend on search or look like a search result.

## Directory layout

```text
aime_harness_evolve/
  harness/
    adapter.py       GEPA text components as Reef nodes, evaluated through Pi
    reference.py     the pinned upstream adapter on a tracked model
    baseline.py      the pre-search alignment gate
    search.py        upstream search, validation-only promotion, sealed test
    heldout.py       per-example checkpoints for the test passes
    models.py        usage ledgers, the spend cap, tracked model clients
    publication.py   the selected composition as a Reef Git-LFS release
    reporting.py     per-cell and aggregate result files
    config.py        every pin and constant a report names
    data.py          the pinned splits and the seed compositions
  results/           the retained seed-0 record
  run.py             the four-cell driver
  run.sh             launcher
  pyproject.toml     the upstream release and the two runtime dependencies
```

It sits under `recipes/gepa/examples/` because upstream GEPA owns the method
and this directory is the runnable experiment; it defines no Reef `Recipe`.
SkillClaw is flat at `recipes/skillclaw/` because it ships its method package.

## Pins

`harness/config.py` is the single source: GEPA `0.1.2` (the
[`v0.1.2`](https://github.com/gepa-ai/gepa/releases/tag/v0.1.2) tag,
commit `92dadfff`), Pi `0.84.2`, `gpt-4.1-mini-2025-04-14` for tasks and `gpt-5-2025-08-07` for
reflection at provider-default sampling, a 150-call search budget, seed 0 plus
1 and 2, and the upstream
[`aime.init_dataset()`](https://github.com/gepa-ai/gepa/blob/92dadfffbe98c8ecf508179a1cab09c1bb85cd32/src/gepa/examples/aime.py)
split (45/45/150, the 30 AIME 2025 problems five times) at pinned Hugging Face
revisions with its full-split SHA-256. The runner refuses any other GEPA
or Pi version and records the exact Reef commit it ran from.

## Setup and run

```bash
pip install -e .                       # gepa==0.1.2, datasets, litellm
REEF_PI_BINARY=/path/to/pi ./run.sh --dry-run
```

A dry run validates the pins and Git LFS, prints the plan with the nominal
task-evaluation count, and loads no data. Live runs read the credential named
by `API_KEY_ENV` (`OPENAI_API_KEY`) after pin validation and hand it only to
transient model bindings; it is never written to a candidate, checkpoint,
report, or published tree. `--max-observed-cost-usd` is required: it persists
completed-call estimates in `observed-cost.json` and starts no new call past
the cap. Calls in flight can overshoot, and a Pi episode can hold several, so
set the account-side project budget as the real ceiling.

Run the gate first, then the cells, into the same directory:

```bash
OPENAI_API_KEY=... REEF_PI_BINARY=/path/to/pi \
  ./run.sh --baseline-only --max-observed-cost-usd <cap> --output-dir outputs/exact
OPENAI_API_KEY=... REEF_PI_BINARY=/path/to/pi \
  ./run.sh --cell all --seeds 0 1 2 --max-observed-cost-usd <cap> --output-dir outputs/exact
```

The full command schedules 5,400 task evaluations plus reflection calls;
`--cell` and `--seeds` narrow it. `--workers` overrides evaluation
concurrency only. `--smoke` uses two examples per split, an eight-call
budget, one worker, and reflection even on a perfect minibatch: a plumbing
check with no authority.

A reused output directory resumes. `run-identity.json` is written once and a
later invocation with a different budget, model, pin, worker count, or smoke
setting is refused; cells, seeds, and the cap may be staged. A cell with
`done.json` is skipped, GEPA resumes from its own checkpoint under `search/`, the test passes
resume from `heldout-checkpoints/` one example at a time, and every usage file
is updated after each call so reports count work done before a restart. The
gate's result is kept once it passes.

## What a run retains

Each search cell keeps GEPA's checkpoint and run log, `summary.json` (the
promotion decision, frozen and selected test scores, Pareto specialists,
metric calls, wall time, token usage, and an estimated cost with the price
snapshot that produced it), `config.json`, the raw result, held-out outputs,
the selected candidate, a learning curve, and the Graphviz candidate tree.
The Reef cells add `publication.json` with the content, release, and parent-release
identities of the published provider-free tree. After every requested cell
and seed finishes, `results.json` compares cells: mean scores, mean delta,
sample deviation, promotion rate, spend, and wall time.

Costs use standard-processing prices observed on 2026-08-30
([GPT-4.1 Mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
$0.40/M input, $0.10/M cached, $1.60/M output;
[GPT-5](https://developers.openai.com/api/docs/models/gpt-5) $1.25/M, $0.125/M,
$10/M); the rates travel with every report.

## Contract and deviations

The reference follows the pinned upstream
[`gepa.optimize` quickstart](https://github.com/gepa-ai/gepa/blob/92dadfffbe98c8ecf508179a1cab09c1bb85cd32/README.md):
`DefaultAdapter`, one `system_prompt` component, Pareto selection over
instance frontiers, `skip_perfect_score=True`, no evaluation cache, 150 metric
calls, seed 0. Only the model aliases are replaced, by dated snapshots. The
Reef cells run the identical search settings over Reef nodes. Stricter than
the short upstream example:

- both arms score with upstream `ContainsAnswerEvaluator`, whose expected
  string includes the exact `### <answer>` marker;
- the baseline gate gives both arms Pi's 16,384-token output cap, while the
  reference search leaves the cap unset as upstream does;
- validation alone decides promotion (a strict aggregate improvement over the
  seed), and only then are the seed and the selection scored on the sealed
  test split, which GEPA never sees;
- the dataset revisions and split hash are verified before any paid call.

The multi-node cell is an extension, not an upstream comparison: it checks
that GEPA can select and reflect on separate named components while Reef
evaluates and publishes the whole tree. The experiment evolves text inside a
fixed topology; adding nodes and evolving executable extensions are separate
questions.

## Deterministic validation

`tests/test_gepa_harness.py` runs without a model call. Fake Pi episodes
drive the real pinned optimizer, and the suite covers the rules mapping,
round-robin evolution of rules and skill, per-instance Pareto specialists,
the strict promotion gate, test sealing, GEPA and held-out resume, usage and
spend accounting, the baseline gate's statistics, and Reef release
publication.

## Results

The retained
[`quickstart-seed-0-2026-09-01`](results/quickstart-seed-0-2026-09-01/README.md)
record pairs the pinned upstream reference with the Reef rules arm under the
same seed, data, models, prompt, and budget. Upstream improved sealed
AIME-2025 accuracy from 26.67% to 38.67%; Reef improved it from 27.33% to
37.33%, with the baseline gate at 22.00% and 20.67%. That is one stochastic
seed of the quickstart, not the larger DSPy experiment in the GEPA paper; the
frozen and multi-node cells and seeds 1 and 2 remain follow-up robustness
work.
