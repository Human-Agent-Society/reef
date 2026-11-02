# Meta-Harness on Terminal-Bench

Run the [Meta-Harness method](../../README.md) on the full pinned
Terminal-Bench 2 suite. See [Results](#results) for the comparison and
[RESULTS.md](../../RESULTS.md) for the evaluation configuration and scope.
The seed is vanilla Terminus 2, represented by a no-op `Agent(Terminus2)`
module at `seed/extensions/agent.py`. The proposer rewrites that module,
the search retains every valid candidate, and Reef selects only strict
improvements over the best recorded mean score.

The example uses the Meta-Harness method, Reef's harness evolution recipe, and
the campaign's own [terminus adapter](#the-terminus-adapter).
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

# Small live wiring check: one task, one repeat, one step.
./run.sh --task extract-elf --iterations 1 --repeats 1

# The full suite, two repeats, four steps.
./run.sh
```

Repeat `--task NAME` to select tasks from the [manifest](harness/tasks.json);
omit it for the full suite. `--iterations` sets the steps the search takes in
all (default 4), and `--repeats` the repeats of each task.
Replace model names as needed, retaining the target's `openai/` provider prefix.
The proposer URL and key default to the target's; the target URL defaults to
`https://api.openai.com`. Unauthenticated endpoints can omit keys.

`REEF_META_HARNESS_WORKERS` defaults to 4; lower it for endpoint or E2B limits.
Runs time out after 9,000 seconds, including setup and verification.
Model keys stay in temporary run configs, not published harnesses.
Sandbox `egress_hosts` enables networking, not hostname filtering; see the
[sandbox configuration](../../README.md#terminus-2-code-evolution).

## Implementation

`run.sh` starts `run.py`, which embeds the configured recipe, a Reef dispatcher,
SQLite storage, and Git LFS artifacts without an HTTP listener. Terminus calls
the model endpoint directly.

`terminal_bench.yaml` trains in `manual` mode, so a training request starts
each step, as `POST /reef/train` does in a deployment. For each step,
`run.py` queues one request with the dispatcher's `accept_instruction`, the
call behind that route, and commits the step's job. The method proposes a
candidate from the search history. Reef evaluates it on the selected suite
against the incumbent's cached scores, adds it to the history, and commits
the step with the serving state.

Runs and evaluations use Reef's executor, timeout, residue policy, and
finite-score checks. Failed evaluation runs score zero; run launch
failures stop the run. Infrastructure failures are not automatically replaced.

Each step costs one proposer call and, for a valid new candidate, one
evaluation: 178 runs on the full suite with 2 repeats (89 tasks). The first
evaluation also runs the seed. A refused proposal, invalid or a repeat, still takes
a step, as in upstream, and evaluates nothing. A failed commit keeps its runs
in the history, and its retry reuses them; a candidate that a retry proposes
in place of the failed one costs extra.

## The terminus adapter

`reef_terminus/` is this campaign's adapter for Harbor's Terminus 2. Reef does
not bundle it. `pip install -e .` in this directory registers it as
`terminus` on the `reef.harness_adapters` entry point group, and installs the
`reef-terminus` runner that its runs launch. A deployment then names it
with `evolution.adapter: terminus`, as `terminal_bench.yaml` does. The runner
writes one trial file per run, and `reef_terminus.trajectory.read_trials`,
the adapter's `sessions` reader, reads it back.

Terminus 2 is a Harbor agent class, not a CLI. `reef-terminus` reads the
tree from `REEF_TERMINUS_DIR`, configures Harbor's `terminus-2` agent
through `reef-eval`, and runs the one Harbor task (a directory or a
registry id) that `--task` names. It writes the verifier reward and the
ATIF trajectory under `REEF_TERMINUS_SESSION_DIR`.

- `config` supplies constructor arguments; unknown keys are refused.
  `rules` supplies an `extra_instruction_paths` entry. `skill` and
  `command` are two `AgentConfig.skills` roots, and Harbor does not mark
  commands as commands.
- One extension file, `extensions/<name>.py`, may define
  `Agent(Terminus2)`. Rendering checks its syntax without running it.
- The binding keeps the served name in `model_name` and sets
  `llm_kwargs.custom_llm_provider` to `litellm_proxy`, so litellm sends
  the name unchanged, whatever its vendor prefix, and passes
  `reasoning_effort`, the thinking budget, and `llm_call_kwargs` in the
  body. Harbor looks up the context limit under `model_name`; a name that
  litellm does not list gets 1,000,000 tokens.
- Routes refused: `llm_kwargs` fields that the binding does not write;
  `llm_call_kwargs` arguments that select an endpoint, provider,
  credential, model, fallback, or logging callback (such as `base_url`,
  `model`, and `fallbacks`); and `model` or `models` in
  `llm_call_kwargs.extra_body`.

To run a published tree yourself, save the `files` of `GET /reef/harness`
under a tree root. Evaluation runs call the upstream directly, but your
own run through Reef needs the scenario header, or Reef answers HTTP 400.
Write `terminus/config.json` with the service's `REEF_TOKEN` (any text
when the service has no token), then run one task:

``json
{
  "model_name": "<served model>",
  "api_base": "http://127.0.0.1:8901/v1",
  "llm_kwargs": {
    "api_key": "<REEF_TOKEN>",
    "custom_llm_provider": "litellm_proxy",
    "extra_headers": {"x-reef-scenario": "<scenario>"}
  }
}
``

``bash
TREE_ROOT="$HOME/reef-harness/terminus"
REEF_TERMINUS_DIR="$TREE_ROOT" \
  REEF_TERMINUS_SESSION_DIR="$TREE_ROOT/terminus/sessions" \
  REEF_TERMINUS_TRIALS_DIR="$HOME/reef-trials" \
  reef-terminus --task /path/to/harbor/task
``

The trial record under `$TREE_ROOT/terminus/sessions` holds the rewards
and the trajectory. Docker must share the trials directory with the host;
with colima on macOS, use a path under your home directory. A trial with
neither reward nor verifier output is reported as a possible mount problem.

### Docker or a remote environment

Tasks run the Harbor task in local Docker by default. The run keeps
the service's `DOCKER_HOST`, `DOCKER_CONTEXT`, and `DOCKER_CONFIG`
(default `~/.docker`), because `HOME` points into the run. On macOS
the run root is under `~/.reef/runs`. Elsewhere it is in the temp
directory; if Docker's VM does not share it, set `TMPDIR` for the service.

To run tasks in a remote Harbor environment, use the sandbox executor and
pass the switch and the credentials into it. For E2B:

``yaml
evolution:
  executor: sandbox
  sandbox:
    egress_hosts: [api.e2b.dev]
    env_from: [REEF_TERMINUS_ENVIRONMENT, E2B_API_KEY]
``

Start the service with `REEF_TERMINUS_ENVIRONMENT=e2b` and
`E2B_API_KEY`. For Modal, allow `api.modal.com`, pass
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, and set
`REEF_TERMINUS_ENVIRONMENT=modal`. The runtime needs Linux, bubblewrap,
Python 3.12+, and Harbor with the environment's extra (`harbor[e2b]`, or
the `modal` package). The interpreter and local task directories must be
visible inside the sandbox, for example under `/opt`. `egress_hosts`
does not yet enforce a hostname firewall.

Reef refuses before launch: local Docker under the sandbox executor, a
remote environment without `egress_hosts` or its credentials in
`env_from`, and an `extension` file under any executor other than
`sandbox`. Such a tree still renders; the refusal depends on how the
run runs, not on what the tree holds.

## Resume and output

Repeat the same command to resume `work/<campaign-id>/` (or under `REEF_WORK`).
Tasks, seed, models, and search settings determine the id; the step count
does not, so a larger `--iterations` continues the same search. Run only one
driver per directory.

Committed Reef state is authoritative. Each request has the ID of its step,
so a restart finds the request of an unfinished step instead of queueing a
second one, and finishes the job that a previous process started. Failed
commits cannot advance search or its summary.

- `harness/`: the search history (every candidate tree, its runs and their
  records, and each step's decision) and one record per step, which holds the
  proposer's prompt and reply in `calls.json`.
- `reef-data/`: SQLite records and the scenario's commits. Each commit holds
  the method state: the replies that proposed no candidate.
- `artifacts.git`, `artifact-work/`, `artifact-cache/`: published harnesses and
  their release chain.
- `summary.json`: rebuilt from the latest commit and the history of the steps
  it covers: each candidate's parent, outcome, and scores, the served id, the
  candidates generated, and the evaluation runs spent.

## Deviations

- **Suite:** full pinned dataset by default; reported scores use a fixed subset
  at the same revision.
- **Schedule:** each evaluation pairs the candidate with the incumbent's
  cached scores, so the incumbent does not run again, except a run that
  could not run, which runs again. The baseline's scores
  arrive with the first candidate's evaluation. As in the comparison, the
  baseline and each candidate run once.
- **Proposals:** the complete population and bounded excerpts of its
  evaluation run records feed the method's prompt, not an upstream tool-using
  coding agent that reads the trial files. [`agent/`](../../agent/README.md)
  runs that coding agent as the proposer.
- **Requests:** Chat Completions/LiteLLM for the target; Responses with default
  reasoning for the proposer. The comparison uses Responses and `xhigh`, plus
  request and verifier adaptations not installed here.
- **Final evaluation:** no automatic winner remeasurement or held-out pass.
  The report's remeasurement uses the same tasks, not a held-out set.

The command therefore does not reproduce the exact comparison protocol.

## Verification

From the repository root:

```bash
pytest tests/recipes/meta_harness/test_meta_harness_terminal_bench.py \
  tests/recipes/meta_harness/test_meta_harness.py \
  tests/recipes/meta_harness/test_meta_harness_artifact_transactions.py
pre-commit run --all-files
```

Tests mock model calls and run launches, checking the real method, recipe,
scoring, commits, publication, and recovery. They do not measure live benchmark
performance or replace a Linux/E2B smoke run.
