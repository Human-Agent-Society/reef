# Meta-Harness on Terminal-Bench

Run the [Meta-Harness recipe](../../README.md) on the full pinned
Terminal-Bench 2 suite. See [Results](#results) for the comparison and
[RESULTS.md](../../RESULTS.md) for the evaluation configuration and scope.
The seed is vanilla Terminus 2, represented by a no-op `Agent(Terminus2)`
module. The proposer rewrites that module, retains every valid candidate, and
selects only strict improvements over the best recorded mean score.

The example uses Reef's shared Meta-Harness recipe and Terminus adapter.
[Deviations](#deviations) documents the differences between the runnable setup
and the comparison.

## Results

The Terminal-Bench comparison starts from vanilla Terminus 2 and evaluates a
baseline plus four full-history iterations with the following settings:

| Setting | Value |
| --- | --- |
| Task | Fixed Terminal-Bench evaluation subset at revision `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`; suite and scope in [RESULTS.md](../../RESULTS.md#configuration-and-scope) |
| Upstream | `stanford-iris-lab/meta-harness@44b9942127847f7421db70d8c7e48407f09a3c70` |
| Models | Target `gpt-5.6-luna`; proposer `gpt-5.6-sol` |
| Requests | Responses API, `xhigh` reasoning effort |
| Seed | Vanilla Terminus 2 |
| Budget | Baseline plus four full-history iterations per arm, two repeats per measurement |
| Selection | Strict improvement in mean score; ties keep the current choice |
| Runtime | Python 3.12.14, Harbor 0.20.0, LiteLLM 1.99.0, OpenAI 2.54.0, E2B 2.46.4 |

Scores count passing trials per measurement:

| Measurement | Reef | Upstream |
| --- | ---: | ---: |
| Baseline | 20/60 | 24/60 |
| Best selected score | 23/60 | 24/60 |

The baseline harness is vanilla Terminus 2 in both arms, measured independently.
Reef selects iteration 1; upstream retains its baseline. The
[full iteration table](../../RESULTS.md) records each candidate's score and
selection decision.

Replaying the same completed score histories through Reef's selector and
upstream's `update_frontier` produced identical choices on all eight candidate
decisions, including Reef's tie. This checks the selection rule given the same
observations; independent proposals and scores can differ.

Remeasuring the selected harnesses with two fresh repeats on the same tasks
gave **22/60 (36.67%) for Reef's iteration 1** and **21/60 (35.00%) for
upstream's baseline**. These measurements did not feed back into search and
are not a held-out task evaluation. One infrastructure loss per arm was
replaced; Reef includes a terminal-loss zero under the shared scoring policy.
The [selected Reef harness](../../results/reef_harness.py) and
[full report](../../RESULTS.md) retain the implementation and scoring details.

## Setup and run

Live runs need Linux, Python 3.12+, Git LFS, bubblewrap, an E2B key, and model
endpoints. Keep Reef, its virtual environment and base Python interpreter, and
the task checkout under `/usr` or a writable `/opt` prefix so the sandbox can
read them. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
git lfs install

git clone https://github.com/harbor-framework/terminal-bench-2.git /opt/terminal-bench-2
git -C /opt/terminal-bench-2 checkout --detach 69671fbaac6d67a7ef0dfec016cc38a64ef7a77c

cd recipes/meta_harness/examples/terminal_bench
pip install -e .
./run.sh --tasks-root /opt/terminal-bench-2 --dry-run
```

The dry run checks the pinned revision and clean task directories, renders the
seed, and prints the run size. It makes no model calls and works on macOS
without E2B. `--tasks-root` defaults to `REEF_TERMINAL_BENCH_DIR`, then
`/opt/terminal-bench-2`.

Use Chat Completions for Terminus and Responses for the proposer. Base URLs
must omit `/v1`:

```bash
export REEF_UPSTREAM_URL=https://your-model-endpoint.example
export REEF_UPSTREAM_API_KEY=...
export REEF_MODEL=openai/gpt-5.6-luna
export REEF_PROPOSER_URL=https://your-model-endpoint.example
export REEF_PROPOSER_API_KEY=...
export REEF_PROPOSER_MODEL=gpt-5.6-sol
export E2B_API_KEY=...

# Small live wiring check: one task, one repeat, one candidate.
./run.sh --task extract-elf --iterations 1 --repeats 1

# The full suite, two repeats, up to four new candidates.
./run.sh
```

Repeat `--task NAME` to select tasks from the [manifest](harness/tasks.json);
omit it for the full suite. Use `--iterations` and `--repeats` for smaller runs.
Replace model names as needed, retaining the target's `openai/` provider prefix.
The proposer URL and key default to the target's; the target URL defaults to
`https://api.openai.com`. Unauthenticated endpoints can omit keys.

`REEF_META_HARNESS_WORKERS` defaults to 4; lower it for endpoint or E2B limits.
Episodes time out after 9,000 seconds, including setup and verification.
Model keys stay in temporary episode configs, not published harnesses.
Sandbox `egress_hosts` enables networking, not hostname filtering; see the
[sandbox configuration](../../README.md#terminus-2-code-evolution).

## Implementation

`run.sh` starts `run.py`, which embeds the configured recipe, Reef dispatcher,
SQLite storage, and Git LFS artifacts without an HTTP listener. Terminus calls
the model endpoint directly.

Each round records one served-harness rollout and its verifier report, rotating
tasks by committed step. The recipe proposes a candidate, evaluates it and the
incumbent on the selected suite, then commits the population and serving state
together. Rollouts and gates use Reef's executor, timeout, residue policy, and
finite-score checks. Failed gate episodes score zero; rollout launch failures
stop the run. Infrastructure failures are not automatically replaced.

Defaults allow four evaluated candidates and eight proposal attempts, including
invalid or duplicate proposals: at most 1,424 gate episodes
(`4 x 2 sides x suite size x 2 repeats`) plus eight feedback rollouts. Smaller
runs scale the gate budget. Retrying uncommitted work can cost extra; these are
search limits, not billing caps.

## Resume and output

Repeat the same command to resume `work/<campaign-id>/` (or under `REEF_WORK`).
Tasks, seed, models, and search settings determine the id. Run only one driver
per directory.

Committed Reef state is authoritative. Restarts reuse recorded rollouts, replay
pending reports, and rebuild stale JSON mirrors. Failed commits cannot advance
search or its summary.

- `reef-data/`: SQLite records and scenario history.
- `artifacts.git`, `artifact-work/`, `artifact-cache/`: published harnesses and
  their release chain.
- `population/<scenario-hash>.json`: post-commit mirror of candidates, parents,
  scores, and attempt audit hashes.
- `summary.json`: committed scores, served id, proposer calls, and gate count.

## Deviations

- **Suite:** full pinned dataset by default; reported scores use a fixed subset
  at the same revision.
- **Schedule:** paired gates remeasure the incumbent but select against its
  previously admitted score. The first baseline score vector arrives after the
  first proposal. The comparison measures the baseline and each candidate once.
- **Proposals:** one feedback rollout and the complete population feed the
  recipe's prompt, not an upstream tool-using coding agent inspecting trial files.
- **Requests:** Chat Completions/LiteLLM for the target; Responses with default
  reasoning for the proposer. The comparison uses Responses and `xhigh`, plus
  request and verifier adaptations not installed here.
- **Final evaluation:** no automatic winner remeasurement or held-out pass.
  The report's remeasurement uses the same tasks, not a held-out set.

The command therefore does not reproduce the exact comparison protocol.

## Verification

From the repository root:

```bash
pytest tests/test_meta_harness_terminal_bench.py \
  tests/reef_service/test_meta_harness.py \
  tests/reef_service/test_meta_harness_artifact_transactions.py
pre-commit run --all-files
```

Tests mock model calls and episode launches, checking the real recipe,
scoring, commits, publication, and recovery. They do not measure live benchmark
performance or replace a Linux/E2B smoke run.
