# Running agents

An "agent" is anything Harbor can run against the task container. Three
integration levels follow, cheapest first; one integration runs in both
regimes, a single task or a [stream](get-started.md#streams). All three
share the same task, scoring, and results store, so numbers stay
comparable across methods.

## The autoresearch contract

Every autoresearch task gives your agent two environment variables and
one protocol:

| | |
|---|---|
| `$JUDGE_URL` | the judge's agent-port HTTP address |
| `$BUDGET_SEC` | your time budget (local runs; containers use the task timeout) |
| `POST $JUDGE_URL/submit` | body = your solution file → `{"score", "best", "remaining", ...}`; over budget → 429 |
| `GET $JUDGE_URL/status` | submissions used / remaining, best so far |

Your final score is your best submission (re-scored by a final judge with
hidden tests, if the task has one). The final evaluation runs on a
**separate verifier port** behind a per-session token; the agent cannot
trigger or observe it. Everything else (how you search, what you
evaluate locally, whether you build your own scorer) is up to you.

If the run set a [budget](get-started.md#budgets) beyond time, the container
also carries `REEF_EVAL_MAX_SUBMISSIONS` and/or `REEF_EVAL_MAX_TOKENS`. Read them
if your method should pace itself; reef-eval records the actual spend either
way.

## The continual-learning contract

A [stream](get-started.md#streams) episode is an ordinary Harbor task graded
by its verifier; the judge endpoint exists only on autoresearch tasks.
The one addition is
`$REEF_EVAL_STATE_DIR`: a directory mounted into every container of the
stream and carried from task to task. Read it at the start, write
whatever your future self should know; reef-eval snapshots it after every
task and never reads it. The format is yours: notes, a skill library, an
evolved harness.

reef-eval only sets the variable. Making the agent use it is part of your
method: a custom harness reads it directly, and a supported harness
needs the run's instruction or system prompt to point at it.

Carrying state also works outside streams. `lab.run(task, agent=...,
state_dir="path/to/memory")` mounts the same directory into a single
episode, so a method can accumulate state inside one autoresearch
problem.

## Level 1: a supported harness (zero code)

`claude-code`, `codex`, `aider`, `cursor-cli`, `terminus-2`, … plus
`oracle` (submits the task's reference solution; proves the pipeline) and
`nop` (does nothing; catches leakage):

```bash
reef-eval run frontier-cs/frontier-cs-2-0-vllm-llm-serving-optimization --agent claude-code --model anthropic/claude-opus-5
reef-eval run frontier-cs/frontier-cs-algorithm-1 --agent codex --budget 2h
reef-eval stream cl-bench --agent claude-code --model anthropic/claude-opus-5
```

The instruction tells the harness the submission protocol; `--budget` sets
the timeout and a `budget` tag. Extra `AgentConfig` fields pass via
`--agent-arg key=value`.

### Credentials

Installed harnesses run their CLI *inside* the task container and need
credentials injected:

- **codex**: `export OPENAI_API_KEY=...`, or reuse your local
  `codex login` session with `CODEX_FORCE_AUTH_JSON=1` (Harbor uploads
  `~/.codex/auth.json` into the container);
- **claude-code**: `ANTHROPIC_API_KEY` (or its own login flow).

### Network access

Locked-down tasks (first-party autoresearch, CL-Bench) set
`allowed_hosts` in `task.toml`, and the container reaches nothing else.
An agent whose CLI runs in the container needs hosts added in both of
Harbor's phases: the **setup phase**, where the CLI installs itself over
`apt` and `npm`, and the **agent phase**, where it calls the model
endpoint. Widen the allowlist per run, with no task edits, through the
`environment.extra_allowed_hosts` override:

```python
import asyncio
from reef_eval import Lab

INSTALL_HOSTS = [  # setup phase: apt / nvm / node / npm
    "deb.debian.org",
    "security.debian.org",
    "raw.githubusercontent.com",
    "github.com",
    "codeload.github.com",
    "nodejs.org",
    "registry.npmjs.org",
]
API_HOSTS = [  # agent phase: codex talking to OpenAI
    "chatgpt.com",
    "auth.openai.com",
    "api.openai.com",
]


async def main():
    lab = Lab("runs/codex")
    row = await lab.run(
        "tasks/continual-learning/cl-bench/bsm-s01",
        agent={
            "name": "codex",
            "model_name": "openai/gpt-5.6-sol",
            "override_setup_timeout_sec": 1200,  # npm install can be slow
        },
        environment={"extra_allowed_hosts": INSTALL_HOSTS + API_HOSTS},
    )
    print(row.rewards)  # the verifier's score for this codex run


asyncio.run(main())
```

The agent-phase-only equivalent, `agent.extra_allowed_hosts`, is
reachable from the CLI as `--agent-arg extra_allowed_hosts='[...]'` but
does not cover setup. On tasks whose network is already open (the
`frontier-cs` folders), the plain CLI needs no overrides.

## Level 2: your own harness (one class)

Implement Harbor's `BaseAgent`, reference it by import path. Your code
runs on the host; the container (where `$JUDGE_URL` lives) is reached via
`environment.exec` / `upload_dir` / `download_file`:

```python
from harbor.agents.base import BaseAgent


class MyAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "my-agent"

    def version(self) -> str | None:
        return "0.1"

    async def setup(self, environment) -> None:
        await environment.exec(command="pip install my-harness")

    async def run(
        self, instruction, environment, context
    ) -> None: ...  # your loop; submit from inside via $JUDGE_URL
```

```python
row = await lab.run(
    "tasks/autoresearch/frontier-cs/frontier-cs-algorithm-1",
    agent={"import_path": "my_pkg.my_agent:MyAgent", "model_name": "..."},
    tags={"harness": "my-harness", "budget": 1},
)
```

Two runnable versions:
[`examples/minimal_harness.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef-eval/examples/minimal_harness.py)
is a ~25-line adapter around a random-search loop, no LLM and no keys, so
it runs anywhere Docker does.
[`examples/llm_harness.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef-eval/examples/llm_harness.py)
adds the part that makes it autoresearch: it asks a model for a candidate,
submits it, and puts the judge's score into the next prompt. Any
OpenAI-compatible endpoint works.

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=...
python examples/llm_harness.py --model deepseek/deepseek-v4-flash
```

Two placements for your method, both fine:

- **Host-side loop**: your method keeps its own GPUs / inference server /
  orchestrator and submits through the container (`environment.exec`, the
  way `llm_harness.py` does it). Fits external learners (TTT-style
  training loops) and host-side multi-agent systems (CORAL-style). The
  container's network policy stays closed and your API key never enters
  it.
- **In-container**: upload your method and launch it; it talks to the
  judge directly. LLM calls from inside need the task's network policy
  opened: add your API host to the `[agent]` phase allowlist; see
  [network policy](authoring-tasks.md#network-policy).

## Level 3: your method isn't an "agent" at all

An evolutionary search, a solver portfolio, a bare sampling loop. The
whole integration is: read `$JUDGE_URL`, POST candidates worth scoring,
stop at 429. The minimal version is ~20 lines
([`examples/random_search.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef-eval/examples/random_search.py)).

An OpenEvolve-style loop plugs in the same way: its `evaluate()` function
POSTs the candidate to `$JUDGE_URL/submit` and returns the judge's score.
Mind the submission budget: evolutionary methods burn evaluations fast,
so give cheap candidates a local pre-filter (your own evaluator, written
from the instruction) and spend submissions on survivors.

## Ready-to-run adapters

[`examples/run_harness.py`](https://github.com/Human-Agent-Society/reef/blob/main/reef-eval/examples/run_harness.py)
runs version-pinned OpenEvolve, Codex and CORAL adapters against a task,
for comparison against your own. They share the task's budgets and record
their token usage like any other agent. Commands, versions and
credentials:
[harness README](https://github.com/Human-Agent-Society/reef/blob/main/reef-eval/examples/harnesses/README.md).

## Comparing fairly

- **Scores come from the task's judge.** A different scoring rule means a
  new task, built from
  [`tasks/_template`](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/_template).
- **Compare methods at the same `budget` tag**, on the same tasks. All
  scores are judge-computed, so the curve comparison is as trustworthy as
  the endpoint comparison. `oracle` and `nop` bracket the plausible range.
- **In streams, compare against the stateless baseline**: the same tasks
  through plain `lab.run`; `metrics.transfer` is the gain the carried
  memory bought.
- **Sanity-check cheaply**: `--agent oracle` proves the task and
  `--agent nop` proves it leaks no answer.
