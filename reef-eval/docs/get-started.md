# Get started

This page goes from installing reef-eval to a real score, in both task
regimes. For the ideas behind the pipeline see [design](design.md); for
evaluating your own agent see [running agents](running-agents.md).

## Install

```bash
pip install "reef-eval[harbor]"    # benchmark tasks download on first use
# or from source, with every task already in tasks/:
git clone https://github.com/Human-Agent-Society/reef && cd reef/reef-eval
pip install -e ".[harbor]"
```

Docker runs the tasks. One caveat: most tasks set `network_mode =
"allowlist"`, so the container reaches only the hosts the task names and
nothing else on the internet. Harbor enforces that with an nftables
sidecar, and Docker Desktop older than ~4.30 lacks the kernel support, so
it will refuse the run (see [troubleshooting](#troubleshooting)).

## Run a task

The `oracle` agent runs a task's reference solution in-container, with
no credentials and no internet, so a failure there points at the setup
rather than at the agent. Start there:

```bash
reef-eval list                                 # what's runnable
reef-eval run cl-bench/bsm-s01 --agent oracle  # should score exactly 1.0
```

Then a real agent. Its CLI runs inside the container, so it needs
credentials there and its hosts added to the allowlist; see
[running agents](running-agents.md):

```bash
reef-eval run frontier-cs/frontier-cs-algorithm-1 --agent claude-code --model anthropic/claude-opus-5 --budget 2h
```

## Budgets

A run is bounded by whichever resource is scarce; set that one and leave
the rest unset:

| Axis | CLI flag | `Budget` field | Enforcement |
|---|---|---|---|
| time | `--budget` (`2h`, `30m`, `90s`; bare = hours) | `time_h` | hard: the container timeout; reaching it ends the run normally and the verifier still grades what exists |
| evals | `--max-evals` | `max_submissions` (the flag and the field differ) | hard at the task's own ceiling (the judge returns 429 past it); a lower per-run cap is signalled |
| tokens | `--max-tokens` (`500k`, `2m`) | `max_tokens` | soft: signalled, actual spend recorded |

Each axis reaches the container as a `REEF_EVAL_*` environment variable and is
tagged on the episode (`budget`, `budget_max_tokens`, ...) so runs group
by it. Actual spend comes back as `used_*` columns: submission counts
from the judge's log and tokens from the harness's usage report.

The eval axis needs a judge, so it applies to autoresearch tasks. A
stream task is graded by its verifier after the episode and has nothing
to submit to, so `--max-evals` does nothing there and reef-eval warns when a
run sets it on such a task. Time and tokens work in both regimes. In a
stream the budget applies to each task on its own, and it is part of the
stream's identity: run the same tasks under a different budget and you
get a separate stream with its own state.

Some CL-Bench domains meter the agent themselves, such as the 15 SQL
queries a dbx question allows. Those limits come from the task and its
sidecar, so they hold whatever you pass on the command line.

## Streams

A stream is an ordered task list under one agent, with a state directory
carried between tasks and mounted into every container as
`$REEF_EVAL_STATE_DIR`. reef-eval never reads its contents. The scores show
whether carrying it helped.

```bash
reef-eval stream cl-bench --agent claude-code --model anthropic/claude-opus-5
reef-eval stream terminal-bench cl-bench --shuffle 1 --agent claude-code --model anthropic/claude-opus-5
```

Targets come first, exactly as in `reef-eval run`, and they decide the order.
A benchmark expands to every task inside it sorted by path, and several
targets run in the order you typed them. `reef-eval stream cl-bench` runs
`bsm-s01`, `bsm-s02`, ... then `code-i01`, `cohort-...`, `dbx-q01`, and
so on: domain by domain, and inside a domain the upstream sequence,
because the converted names are zero-padded. `reef-eval stream terminal-bench
cl-bench` runs all of terminal-bench and then all of cl-bench.

`tasks("cl-bench")` returns that same list in Python, so filtering,
reordering, or repeating entries is ordinary list work before you hand it
to `Stream`, which runs exactly the list it is given. `--shuffle SEED`
shuffles the list deterministically and records the seed as a tag, so
each seed is its own stream. AgentStream's scenarios map onto this:
sequential is the target order, interleaved is a seeded shuffle, and
isolated is one stream per benchmark, with no state shared between them.

Re-running the same command resumes; a different agent, tags, budget, or
task list is automatically a separate stream with its own state. `--name`
labels the stream (it becomes the `stream` tag and the state directory);
without it the label is derived from the targets. Pass a new `--name` to
run the same tasks again from empty memory, the way `--tag attempt=2`
gives `reef-eval run` a fresh attempt.

Around each task, the live state directory is reset from the previous
snapshot before the run and snapshotted after, so a crashed stream picks
up where it left off and every step's memory can be audited later:

```
<lab>/streams/<name>-<variant>/
  state/                    # mounted into the current task
  snapshots/init            # the state before position 0 (seed a memory here)
  snapshots/000-<digest>    # each task's ending state; the digest covers the
  snapshots/001-<digest>    # task list up to here and matches the episode key
```

Appending tasks extends a finished stream; editing an earlier position
re-runs everything after it.

## The Python API

A `Lab` is a directory holding one results table. Each `run` is one
episode (one Harbor trial); `df` returns everything recorded so far as a
pandas DataFrame:

```python
from reef_eval import Budget, Lab, Stream, metrics, tasks

lab = Lab("runs/exp1")
row = await lab.run(  # asyncio: inside an async function or a notebook
    "tasks/autoresearch/frontier-cs/frontier-cs-algorithm-1",
    agent={"name": "claude-code", "model_name": "anthropic/claude-opus-5"},
    budget=Budget(time_h=2),  # or max_tokens=..., max_submissions=...
    tags={"prompt": "v2"},  # free-form; each key becomes a df() column
)
row.rewards  # the trusted score
row.uri  # the trial directory, for auditing

stream = Stream("cl-bench", tasks("cl-bench"))  # every task in the benchmark
await stream.run(lab, agent={"name": "claude-code"}, budget="30m")

df = lab.df("episode")
df.groupby(["model", "task"])["reward"].mean()
```

`tasks()` resolves what the CLI resolves: a task directory, a folder of
tasks, a benchmark name (downloaded on first use), or a Harbor registry id,
which passes through as-is. It returns the references as a list of strings
in the CLI's order, so
`tasks("cl-bench")` is the list `reef-eval stream cl-bench` runs, and printing it
is how you see what a target covers:

```python
order = tasks("cl-bench")
len(order)  # 301
order[:3]  # the first three, as paths

Stream("first-20", order[:20])  # a slice
Stream("poker-only", [t for t in order if "poker" in t])  # a filter
Stream("revisit", [*order[:10], order[0]])  # a repeat, which measures forgetting
```

This is where the Python API goes past the CLI. `reef-eval stream` runs the
resolved list as it comes, with `--shuffle SEED` for a deterministic
reshuffle and nothing else; `Stream` runs exactly the list it is given, so
any other order, subset, or repetition is yours to build. The list is part
of the stream's identity, so each of the streams above keeps its own state
and resumes on its own.

Every run, from any script and any day, appends to the same table, so
comparing agents is a query over it and [metrics](metrics.md) are
functions over it. `lab.run_many([...])` runs a batch with bounded
concurrency; `lab.df("trace")` returns the judge's per-submission log.

## Resume

Every episode gets a key derived from the task path, the agent, the tags,
and any overrides. A budget is recorded as tags, so it is part of the key
too. The keys live in the results table inside the lab directory: when
that table already has a row for the key, `run` returns the stored row
and executes nothing, and otherwise the episode runs.

Re-running the same script or the same `reef-eval stream` command therefore
picks up where it left off, and no daemon or job file has to remember
anything between runs.

A recorded failure counts as recorded. An episode whose trial raised
stores a row carrying an `error` tag, and a later call skips it like any
other row. A crash that killed the process mid-episode stores nothing,
so that episode runs again from the start.

To run something again instead of resuming it:

| You want | Do |
|---|---|
| another attempt beside the first | vary a tag: `--tag attempt=2`, or `tags={"attempt": 2}` |
| an empty table | a new lab directory: `--lab runs/exp2` |
| the same tasks from empty memory | a new stream name: `--name wk2` |
| one episode again, in place | `lab.store.delete_prefix(key)` (the key is the `key` column of `lab.df()`), which drops the episode row and its trace rows |

Resume is episode-granular: a half-finished episode starts over, because
a run stitched from checkpoints is not comparable to a clean budget.

## No Docker? Local and fake runs

`--local` starts the task's own judge as a local process and runs your
command against it:

```bash
reef-eval run autoresearch/first-party/circle-packing --local \
  --command "python examples/random_search.py" --budget 30s
```

The judge code is the same as the container sidecar's, but nothing is
isolated (even hidden tests are readable on your machine), so local rows
carry a `local://` uri and are never trusted results. Use local runs
while developing and report the numbers from container runs. `--fake`
and `python examples/quickstart.py` need no setup at all; their scores
are simulated.

## Where the data is stored

A `Lab` is a directory (`--lab`, default `runs/cli`):

```
runs/cli/
├── results.sqlite                  # one table: episode rows + trace rows
└── trials/<task>__<id>/
    ├── agent/trajectory.json       # every step, with tokens and duration
    ├── verifier/reward.json        # the final score
    ├── verifier/submissions.jsonl  # every judge-scored submission, with t
    └── result.json, config.json, trial.log
```

`reef-eval report` summarizes the store; every row's `uri` points back at its
trial directory, so any number can be traced to the files behind it.

## Troubleshooting

- **`Cannot connect to the Docker daemon`**: start Docker first.
- **`network_mode='allowlist' is not supported by EnvironmentType.DOCKER`**:
  your Docker VM kernel lacks nftables FIB rules (Docker Desktop <= ~4.30).
  Upgrade Docker; until then `--local` develops against the real judge.
- **An agent CLI fails during setup with npm/apt errors**: its install
  hosts are missing from the allowlist, or the setup timeout is too
  small; see [running agents](running-agents.md#network-access).
- **An agent CLI fails immediately with 401**: no credentials; see
  [running agents](running-agents.md#credentials).
