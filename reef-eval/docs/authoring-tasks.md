# Authoring tasks & benchmarks

Tasks are **100% stock Harbor tasks**, validated against Harbor's
`TaskConfig` by `tests/test_task_suite.py`; reef-eval adds conventions around
the format, never fields inside it. Start from
[`tasks/_template`](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/_template), a working
placeholder task (maximize `x` in `[0, 1]`) that passes the suite before
you change anything. `harbor trial start -p <dir>` runs it standalone.

An **autoresearch task** ships its own judge, which is most of this page.
A **continual-learning task** is any stock Harbor task; see
[below](#a-continual-learning-task).

## Anatomy of an autoresearch task

Two containers: the agent's, and the judge's. All scoring lives with the
judge; the agent only sees scores come back over HTTP.

```
my-task/
├── task.toml            # stock Harbor config; network allows only the judge
├── instruction.md       # the problem + the submission protocol
├── environment/
│   ├── Dockerfile           # the AGENT's container (its data, no scoring)
│   ├── Dockerfile.judge     # the JUDGE's container
│   ├── docker-compose.yaml  # wires the judge in as a sidecar, sets $JUDGE_URL
│   ├── judge_server.py      # generic HTTP server; never edited per task
│   ├── score.py             # THE scoring rule: grade(path) → {"reward", "reason"}
│   ├── final.py             # optional final judge (hidden tests); see below
│   └── judge_config.json    # {"max_submissions": N, "min_interval_sec": s}
├── tests/
│   ├── test.sh · grade.py   # generic verifier: asks the judge for /final
│   └── grader_tests.json    # unit tests for score.py (and final.py)
└── solution/solve.sh        # reference solution; submits once, proves the pipeline
```

## The judge protocol

The agent gets `$JUDGE_URL` and a submission budget:

| Request | Who calls it | What happens |
|---|---|---|
| `POST /submit` (body = the solution file) | the agent, at will | `score.py` grades it and records the result; over budget → 429 |
| `GET /status` | the agent | submissions used / remaining, best so far |
| `GET /final` | nobody, on this port | **403**: finalization is a verifier-only capability (see below) |

Finalization runs on a separate verifier port (`VERIFIER_PORT`, default
`PORT + 1`) behind a per-session token generated at startup. The token is
written to `{DATA_DIR}/.verifier_token`, inside the judge's own
filesystem, never in the agent's environment. The local executor reads it
from there; the container verifier fetches it via `GET /token` on the
verifier port (which the agent should not be able to reach; the network
policy keeps the agent on the agent port only).

| Request | Who calls it | What happens |
|---|---|---|
| `GET /final` | the verifier, once, at the end (verifier port + token) | the final grade + the full submission log. **Terminal**: the first call locks the session; later submissions are refused, repeat calls return the cached grade. |
| `GET /token` | the verifier (verifier port) | returns the per-session verifier token |

Scoring code and data belong only in the judge image, and the agent's
network reaches only the judge's agent port (`network_mode =
"allowlist"`, `allowed_hosts = ["judge"]`); see
[design](design.md#reward-hacking-scoring-runs-outside-the-agents-container).

At the end the verifier calls `GET /final`, the judge freezes the best
submission by session score, runs `final.py` on it if present, and locks
the session, so the agent never observes the final evaluation.

## The submission budget

`judge_config.json` sets the cap. With a public metric and cheap scoring
(circle-packing, tsp-tour), a generous cap of around 1000 just bounds
judge compute. When scoring is expensive (string-compression runs the
agent's program), set it tighter. When the metric has a secret, keep the
session feedback on public data and put the secret in the final judge
below, so the cap carries no secrecy burden.

## The final judge (`final.py`, optional)

Same signature as `score.py`, run exactly once, on the best submission,
when the verifier finalizes. This is where hidden tests live: held-out
data, stricter checks, anything the session score must not leak. In
[symbolic-regression](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/autoresearch/first-party/symbolic-regression)
the session scores on training points and the final judge scores once on
held-out points, so no submission budget can probe them. Without
`final.py`, the final grade is the best session score.

## The scoring contract

`score.py` (and `final.py`) expose one function:

```python
def grade(artifact: Path | None) -> dict:
    # {"reward": float, "reason": str}; invalid input → 0.0 with a reason
```

Rules: recompute everything from the submitted file, never trust anything
inside it; validate conservatively (exact arithmetic where floats can be
gamed; reject, don't round); malformed input scores `0.0` with a reason,
never an exception. Data files sit next to the script and are read via
`Path(__file__).parent`, so the same file works in the judge image and in
`--local` runs.

## Testing your task: `tests/grader_tests.json`

Each case is one sentence: *this solution must score this reward,
because…* Write one case for your reference solution and one for **each
rule your scorer enforces** (expected reward `0.0`). The template's file:

```json
{
  "solve_sh_scores": 0.5,
  "cases": [
    {"solution": {"x": 0.5}, "reward": 0.5, "why": "the reference solution scores exactly this"},
    {"solution": {"x": 1.5}, "reward": 0.0, "why": "out-of-bounds values must be rejected"},
    {"solution": {"nonsense": true}, "reward": 0.0, "why": "malformed input must score zero, not crash"}
  ]
}
```

- `cases` exercise `score.py` offline, in milliseconds, on every CI run;
  add `final_cases` (same shape) when the task has a final judge.
- `solve_sh_scores` drives the containerized E2E gate: what the reference
  solution must earn end to end, either a number or `{"min": …, "max": …}`
  where the live run legitimately varies.
- Optional per-case `"tolerance"` for float comparison (default 1e-9).

`pytest tests/test_task_suite.py` picks up any folder with this file. In
containers, `--agent oracle` must reproduce `solve_sh_scores` (the E2E
gate) and `--agent nop` must score 0; if it doesn't, the environment leaks
the answer.

## Network policy

The default is the tightest thing that still works: the agent may reach
the judge and nothing else.

```toml
[environment]
network_mode = "allowlist"
allowed_hosts = ["judge"]

[agent]                            # optional phase override, e.g. for
network_mode = "allowlist"         # in-container LLM loops
allowed_hosts = ["judge", "api.openai.com"]
```

Never give the agent open internet on a task whose answers can be looked
up.

## A GPU task

Declare the requirement in `task.toml` (`[environment]` `gpus = 1`),
which Harbor validates and its cloud backends honor. For local Docker,
add the standard nvidia device reservation
(`deploy.resources.reservations.devices`) to the `main` service in the
task's `environment/docker-compose.yaml`; give the judge service the
reservation too when scoring itself needs the GPU. For kernel-timing
tasks, wall-clock speedups on shared hosts are noisy: score against a
reference implementation run in the same container and session, and record
the GPU model as a tag so curves never mix hardware.

## A continual-learning task

Any stock Harbor task runs in a [stream](get-started.md#streams)
unchanged: reef-eval mounts the carried memory directory itself, and a
pass/fail verifier is a fine score (a pass is a 0-or-1 reward). Two
conventions from the committed benchmarks are worth copying:

- **Hidden state lives in a sidecar.** When something must stay out of
  the agent's reach at runtime (CL-Bench's metered database, its poker
  deck), reuse the judge pattern: an HTTP sidecar holds the secret and
  enforces the interaction budget.
- **Scoring is the upstream metric.** A converted benchmark keeps its
  published scorer, ported verbatim where possible, so numbers stay
  comparable with the source paper.

## Define a benchmark

A benchmark is a directory whose immediate children are task folders.
That is the whole format:

```
my-bench/
├── task-01/    # a stock Harbor task: task.toml, instruction.md, ...
└── task-02/
```

A path runs as-is, and a folder placed in the catalog resolves by name:

```bash
reef-eval run path/to/my-bench --agent oracle       # any directory of tasks
reef-eval run my-bench --agent oracle               # once it sits in tasks/<regime>/my-bench
```

In a checkout, `tests/test_task_suite.py` picks up every task under
`tasks/`, so a new benchmark is validated the moment the folder exists.
To distribute one, publish the directory in any git repository and
register the pin, the way gym environments register:

```python
from reef_eval import fetch

fetch.register("my-bench", "https://github.com/me/my-bench.git", "<commit or tag>")
tasks = fetch.benchmark("my-bench")  # downloads on first use, then cached
```

`subdir=` points inside the repo when the tasks are not at its root. The
registry is per process, the way gym's is: a script that calls `register`
can pass `fetch.benchmark("my-bench") / "task-01"` straight to `Lab.run`,
while the shell `reef-eval` command sees only the built-ins, so give it the
fetched path. Ship the `register` call in your package's import;
registering an existing name replaces it, which is how a fork takes over
a built-in.

## A benchmark converter

A converter turns a published external format into a folder of task
dirs. It lives beside the benchmark it maintains, next to a `fetch.py`
pinned to an upstream commit so the committed tasks can be regenerated.
Check one real upstream spec into `tests/fixtures/` and validate the
emitted task under Harbor's `TaskConfig`;
[`edgebench/convert.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef-eval/tasks/autoresearch/edgebench/convert.py)
is the reference.
