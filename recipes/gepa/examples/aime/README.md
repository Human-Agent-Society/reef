# GEPA on AIME: validating the method

[GEPA](https://github.com/gepa-ai/gepa) is reflective prompt evolution: keep an
archive of candidate prompts, Pareto-sample a parent from it, evaluate that
parent on a small training minibatch, ask a stronger model to rewrite one
component in light of what went wrong, keep the child only if it beats its
parent on that minibatch, then score it on the whole validation set and serve
the best. `recipes/gepa/` is that algorithm written as a Reef harness evolution
method - `recipes.gepa:GEPA` under the `evolution:` contract, with the archive
in the method's `state/archive.json`, which Reef commits with each step - and
this example is what validates it. The benchmark is the upstream AIME
quickstart: 45 training problems, 45 validation problems, a sealed 150-problem
AIME-2025 test split, a 150-call search budget, seed 0. Nothing here imports `gepa`; the two places the example
reproduces upstream text (the scorer's feedback wording and the epoch-shuffled
minibatch order) say so in `harness/aime.py`.

## What maps to what

| GEPA | Reef |
| --- | --- |
| Candidate prompt | a harness tree; the evolvable file here is `rules.md`, rendered into Pi's package as `plugin/rules.md` |
| Pareto-sample a parent from the archive | `Archive.select_parent` over per-problem fronts, dominated candidates pruned |
| Evaluate the parent on a train minibatch, with traces | the served tree's own recorded traffic: the driver runs the minibatch through the service, so the method gets the transcripts free; any other parent runs as runs through `services.evaluate` |
| Reflect on one component and propose a rewrite | `models["reflection"]` (gpt-5) with GEPA's own prompt, over `Inputs` / `Generated Outputs` / `Feedback` records |
| Accept the child iff it beats the parent on the minibatch | the method runs the child through `services.evaluate` and leaves `harness/` unchanged on a reject, so the step publishes nothing and commits the archive |
| Full validation pass, then Pareto update | `evolution.evaluation.tasks` is the validation set: the step's evaluation scores the child on it, and the next step reads the per-task scores from the history into the archive |
| Serve the argmax-mean candidate | the `compare` policy with `statistic: mean` publishes a strict mean improvement over the served tree's validation scores, which the score cache keeps, so the served tree stays the archive's best; it publishes the tree for `GET /reef/harness` |

The request envelope is reproduced by seed files rather than by a custom Pi
command line: `seed/config/primary.json` sets `defaultTools: []`, which Reef's own
extension applies, and the fixed extension `seed/extensions/reef-quickstart-system.ts`
makes the rendered rules text Pi's entire system prompt at
`before_agent_start` and flattens a single-part user message to the plain
string upstream sends. Neither is evolvable (`components: [rules]`). The
extension reads the package's `rules.md` at startup, so `rules.md`
underneath it can evolve freely.

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

A live run is roughly 150 search runs plus the evaluation's validation
passes, then 300 test runs. Runs are independent, so `REEF_GEPA_WORKERS`
(default 128) run at once everywhere: the evaluation's validation pass through
`execution.evolution.workers`, the driver's minibatch, and the test passes. It
changes wall time only. `--budget` lowers the search budget for a
shorter run. `REEF_GEPA_MULTI=1` adds an `aime-solver` skill to the seed tree
and evolves it alongside `rules.md`, which is the extension case, not the
comparison. The credential is read once, handed to the embedded service as its
upstream, and never written into a candidate, a checkpoint, or a published
tree; the Pi runs authenticate to the local service with a placeholder.

The validation pool defaults to `auto`: one worker selects `uni`, multiple
workers select `mp` (or Ray in an existing multi-worker placement group).
The former `local` thread-pool executor has been removed. The retained
concurrency default of 128 now means 128 persistent Python worker processes;
reduce `REEF_GEPA_WORKERS` to fit host memory and provider rate limits, for
example `REEF_GEPA_WORKERS=4 ./run.sh`. Set `REEF_GEPA_EXECUTOR=ray` for Ray
actors, or `REEF_GEPA_EXECUTOR=uni REEF_GEPA_WORKERS=1` for one in-process
worker. After it builds the recipe,
the driver replaces the configured scorer and feedback classes with one
`AIMEScorer` built from a serializable snapshot of the answer and context
tables. Workers do not rely on driver globals, and later
registrations do not change an existing recipe. Labels stay in scorer state,
not in the task prompts or rendered harness files. Run subprocesses and
temporary directories remain isolated. Custom hooks are preserved; they must
carry their own serializable state when used across processes.

For example, a two-worker local Ray run:

```bash
pip install 'ray[default]'
REEF_GEPA_EXECUTOR=ray REEF_GEPA_WORKERS=2 ./run.sh
```

To use an existing cluster, also set `RAY_ADDRESS` to its connection address
(for example `ray://head-node:10001`). Use compatible Python/Ray versions and
install the same Reef revision, Node.js, and Pi `0.84.2` on every worker node.
`REEF_PI_BINARY` must resolve on every node (use `pi` on PATH or a common
absolute installation path). The driver uploads only this example's
`harness/` Python package through Ray's runtime environment, not the work
directory or credentials. If embedding the driver in an already initialized
Ray runtime, provision that package there yourself; the driver neither
reconfigures nor shuts down a runtime it does not own.

Ray affects **only the evaluation's worker pool**, which runs the validation
passes and the method's own runs (every child, and a parent that never
served). The
driver's minibatch and held-out test passes still use local threads. Validation
workers call the configured upstream model directly; they do not need access
to the embedded loopback service. They do need network access to the provider,
and receive the model credential in their transient run binding, so use
only trusted Ray nodes. The embedded service remains bound to `127.0.0.1`.

Size the Ray pool to available resources and provider rate limits: each worker
requests one logical CPU and zero GPUs by default, so the default 128 workers
would require 128 available logical CPUs for the whole pool to become ready.
Override `execution.evolution.resources.cpus_per_worker` only when appropriate
for the workload. Worker count is fixed for a run, not autoscaling.
The driver still forwards top-level `execution` and `executors`, including
named profiles and Ray actor options such as `runtime_env`. `--dry-run`
validates the recipe but does not launch workers or check remote dependencies.

One round is: pull the served source tree (`GET /reef/harness?form=source`),
read the archive from the latest commit, settle it from the history, and draw
the iteration's plan as the method will (its parent choice and its three
training problems), render the tree with a model binding pointed back at the
embedded service, run each problem as a Pi run, score it, find its
recorded request by problem text, and report the score against it. The third
report fills the step's job data (`evolution.batch_size: 3`) and schedules the
training step, which is where the method proposes, the evaluation validates,
and a win publishes. Rounds stop when the archive reaches the metric-call
budget.

## What a run retains

Everything under `./work`, and a rerun resumes from it:

- `reef-data/` - the scenario's records and commit history. Each commit
  carries the method's `archive.json`: candidates and their texts, parents,
  per-problem validation vectors, the round-robin cursors, the metric-call
  count, each reflection's minibatch scores and result, and the seeded
  generator's state - which parent and which training problems each round
  uses, drawn in upstream's own order, so seed 0 here shows the reflection
  model the same problems the official seed-0 run showed it.
- `harness/` - the scenario's history (every candidate tree, every scored
  run, every decision) and its step records. The history is the score
  cache, and the next step settles the archive from it. Each step record's
  `calls.json` keeps the reflection's prompt and reply.
- `artifacts/` - the release chain.
- `heldout/{frozen,selected}/example-NNNN.json` - one checkpoint per test
  run, keyed by the problem and the tree it ran on; a checkpoint written
  for a different tree is refused, never folded into the score.
- `seed/`, `bootstrap/`, `served/`, `selected/` - the seed tree the recipe
  boots from, the base release's tree, and the source trees the last round
  and the test pass pulled.
- `summary.json` - validation seed and selected means from the archive, both
  test scores, the candidate count, the metric calls, and the retained official
  numbers beside them.

## The validation contract

Four runs on 2026-09-02, two of the upstream optimizer and two of this method,
across two seeds, with the models, split, budget, scorer, and worker count
held fixed. Every run produced four candidates and consumed 198 metric calls.
The runs pair by seed: at a given seed both arms reflect on the same training
problems in the same order. The method's two runs predate the current driver;
a fresh run on it is pending.

| seed | arm | frozen | selected | improvement |
| ---: | --- | ---: | ---: | ---: |
| 0 | upstream GEPA | 31.33% | 42.67% | +11.33 pp |
| 0 | this method | 26.67% | 46.67% | +20.00 pp |
| 1 | upstream GEPA | 26.67% | 40.67% | +14.00 pp |
| 1 | this method | 24.00% | 36.00% | +12.00 pp |
| | upstream mean | | | +12.67 pp |
| | this method mean | | | +16.00 pp |

The means are across seeds because one run of this search has no stable number
to report. GEPA is a stochastic optimization: the reflection model writes a
different prompt each time, and the task model scores a given prompt
differently from run to run. This method is 8.67 points above upstream at seed
0 and 2.00 below it at seed 1, and a difference that changes sign between seeds
is sampling. The frozen column is the yardstick, being the same seed prompt in
all four runs, and it scores between 36 and 47 of 150. The records are in
[`results/`](results/).

The deterministic half of the contract is `tests/recipes/gepa/test_gepa_aime_harness.py`,
which runs with no model and no Pi binary: the scorer, the feedback wording,
the dataset drift refusal, and the driver's record lookup and report against a
real embedded service with a stubbed inference backend. Where the upstream
package is installed, `tests/recipes/gepa/test_gepa_fidelity.py` drives
`gepa.optimize` and this method side by side on a synthetic task from the
same seed, the method through whole steps on the real backend, and requires
the same candidates, parents, minibatches, and validation means at every
iteration.

## Deviations

- Every run goes through the score cache. A (tree, problem) pair that the
  history holds a score for is not run again, except a run that could not
  run, which runs again: the served tree's validation is its stored one, as
  upstream keeps it, and a parent that ran a minibatch problem before reuses
  that score where upstream would sample it again. The archive charges every
  metric call that upstream charges, cached or not, so the budget means what
  it means upstream. A change of the evaluation context starts a new cache.
  The context is the served model and the named models, the adapter and its
  version, the harness binary (`evolution.binary`), the scorer, the run
  settings (`task_timeout_s`, `forbid_residue`), and the run executor
  (`evolution.executor`, `evolution.sandbox`).
- The training minibatch is real recorded traffic through the service rather
  than a direct evaluation call, which is the point of the exercise but means
  a minibatch problem is scored by the driver and re-read by the method from
  the transcript, not scored twice.
- `evolution.evaluation.tasks` is data, not configuration, so `gepa.yaml`
  ships it empty and `run.py` fills it from the pinned split before
  `build_recipe`.
