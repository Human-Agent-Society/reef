# Meta-Harness on Terminal-Bench

Run the [Meta-Harness recipe](../../README.md) on the same pinned 30-task
Terminal-Bench 2 hard subset as the [retained comparison](../../RESULTS.md).
The seed is vanilla Terminus 2, represented by a no-op `Agent(Terminus2)`
module. The proposer rewrites that module, retains every valid candidate, and
selects only strict improvements over the best recorded mean score.

This example uses the current shared Reef recipe and Terminus adapter. It is
a runnable continuation of the method reproduction, not the internal runner
that produced the historical numbers. The differences are listed below;
`RESULTS.md` and its selected harness remain in the package directory.

## Setup and run

Live code evolution requires Linux, Python 3.12+, Git LFS, bubblewrap, an E2B
key, and model endpoints. Put Reef, its Python environment, and the task
checkout under `/usr` or a writable `/opt` prefix: those paths are visible
inside Reef's sandbox. The Python interpreter behind the virtual environment
must also be installed under one of those prefixes.

From the repository root, using that Python interpreter:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[terminus]'
git lfs install

git clone https://github.com/harbor-framework/terminal-bench-2.git /opt/terminal-bench-2
git -C /opt/terminal-bench-2 checkout --detach 69671fbaac6d67a7ef0dfec016cc38a64ef7a77c

cd recipes/meta_harness/examples/terminal_bench
pip install -e .
./run.sh --tasks-root /opt/terminal-bench-2 --dry-run
```

The dry run verifies the task checkout's revision, refuses changes or extra
files in the selected task directories, renders the seed without importing
candidate code, and prints the task and episode counts. It does not build a
sandbox, start a trial, or call a model, so it also works on macOS without E2B
credentials. `--tasks-root` defaults to `REEF_TERMINAL_BENCH_DIR`, or
`/opt/terminal-bench-2` when unset.

Configure a Chat Completions endpoint for Terminus and a Responses endpoint
for the proposer. Base URLs must omit `/v1`; keys are supplied separately:

```bash
export REEF_UPSTREAM_URL=https://your-model-endpoint.example
export REEF_UPSTREAM_API_KEY=...
export REEF_MODEL=openai/gpt-5.6-luna
export REEF_PROPOSER_URL=https://your-model-endpoint.example
export REEF_PROPOSER_API_KEY=...
export REEF_PROPOSER_MODEL=gpt-5.6-sol
export E2B_API_KEY=...

# Small live wiring check: one task, one repeat, one candidate.
./run.sh --task cancel-async-tasks --iterations 1 --repeats 1

# The 30-task suite, two repeats, up to four new candidates.
./run.sh
```

The model names match the retained experiment and require an endpoint that
serves them; replace both names with models available at your endpoint when
needed. The target name includes LiteLLM's `openai/` provider prefix.
`REEF_PROPOSER_URL` and `REEF_PROPOSER_API_KEY` default to the target endpoint
and key. Endpoints without authentication can leave the keys unset.
`REEF_UPSTREAM_URL` defaults to `https://api.openai.com`.

`REEF_META_HARNESS_WORKERS` defaults to 4 concurrent gate episodes. Lower it
to fit endpoint rate limits and E2B capacity. Each episode has a 9,000-second
outer timeout, including setup and verification. The sandbox forwards only
the E2B key and environment selector; model credentials are bound into the
temporary episode config, not the evolved or published composition.
The sandbox's `egress_hosts` setting enables networking but does not enforce
a hostname firewall, as documented by the shared recipe.

## Implementation

`run.sh` loads deployment defaults and starts `run.py`. The driver embeds
Reef's dispatcher with the recipe in `terminal_bench.yaml`, a Git LFS artifact
repository, and SQLite scenario storage. It does not start an HTTP listener:
Terminus calls the configured model endpoint, and the driver imports its
completed trajectory and verifier reward into Reef's records.

Each round runs one task through the served composition, rotating through the
suite by committed step number. That real rollout is one inference record
containing its episode transcript, paired with a report referencing that
record. `data.batch_size: 1` makes it one training batch. The shared recipe
then proposes a composition, runs candidate and incumbent on the entire
suite, and commits the selection, population, and published tree together.
Both the rollout and the gate use Reef's configured episode executor,
timeout, residue policy, and finite-score checks. Failed gate episodes count
as zero; a rollout that cannot launch stops the driver before a proposal.
No infrastructure failure is automatically replaced with another trial.

Four accepted candidate proposals cost at most 480 gate episodes
(`4 x 2 sides x 30 tasks x 2 repeats`), plus one feedback rollout per proposal
attempt. Invalid and duplicate proposals can consume attempts without a
gate; the default limit is eight attempts and therefore eight feedback
rollouts. Here "accepted" means admitted for evaluation: a candidate need
not win selection to count. A small run gets proportionally smaller budgets.
Retries after a failed, uncommitted step can incur additional provider spend;
these are logical search budgets, not billing caps.

## Resume and output

The printed directory is `work/<campaign-id>/`, or under `REEF_WORK` when
set. The id hashes the task paths, seed, model settings, and search
configuration, so a different suite or budget gets separate state. Repeat
the same command to resume. Run only one driver against a campaign directory
at a time.

Reef's committed algorithm state is the source of truth for the population,
served candidate, attempts, and gate episode count. Pending rollout records
and reports replay after an interruption; a saved rollout does not need
another model call just because its report was interrupted. A failed scenario
commit cannot advance the search or publish a summary for that step.

- `reef-data/`: SQLite records and the durable scenario history.
- `artifacts.git`, `artifact-work/`, `artifact-cache/`: published harnesses and
  their release chain, recoverable by a new process.
- `population/<scenario-hash>.json`: the recipe's post-commit population
  mirror, including candidate code, parents, scores, and attempt audit hashes.
- `summary.json`: committed candidate score vectors and means, served id,
  proposer calls, and gate episode count. A restart rebuilds stale mirrors
  from Reef's committed state.

## Distance from the recorded protocol

- The dataset revision and 30-task list match the retained comparison. The
  [manifest](harness/tasks.json) cites the upstream script that defines the
  subset. The task files are fetched from upstream, not copied into Reef.
- The historical runner measured the baseline once and each new candidate
  once. This example uses Reef's paired gate: the first gate supplies the
  baseline measurement, and later gates remeasure the incumbent even though
  selection uses its previously admitted score. The first proposal sees a
  feedback rollout but no full baseline score vector.
- This example adds one recorded feedback rollout per attempt. The proposer
  sees those transcripts and the complete retained population through the
  recipe's standard prompt; it does not run the upstream tool-using coding
  agent or inspect the internal experiment's raw trial directories.
- The target uses the shared adapter's Chat Completions/LiteLLM path. The
  proposer uses Responses with provider-default reasoning settings. The
  retained local experiment used Responses and `xhigh` effort; its request,
  verifier, and infrastructure-replacement adaptations are not installed by
  this example.
- The Python package pins match the recorded Harbor, LiteLLM, OpenAI, and E2B
  versions. The seed has no behavioral override; only `code_extension`
  evolves. The historical selected harness is linked from
  [RESULTS.md](../../RESULTS.md), not used as the seed.
- There is no automatic winner remeasurement or held-out pass. New runs are
  method demonstrations, not reproductions of the exact historical scores.

## Verification

From the repository root:

```bash
pytest tests/test_meta_harness_terminal_bench.py \
  tests/reef_service/test_meta_harness.py \
  tests/reef_service/test_meta_harness_artifact_transactions.py
pre-commit run --all-files
```

The campaign tests replace model calls and episode launches while exercising
the real recipe, scoring, selection, SQLite commit log, artifact publication,
failed commits, and restart recovery. They do not establish benchmark scores
or replace a live Linux/E2B smoke run.
