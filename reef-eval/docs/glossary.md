# Glossary

Every term reef-eval uses, in one place. Terms link to the page that owns them.

## Tasks and runs

| Term | Meaning |
|---|---|
| task | A stock Harbor task directory: `task.toml`, `instruction.md`, `environment/`, `tests/`, and a reference `solution/`. |
| episode | One run of one task under one agent; reef-eval's unit of measurement. Equals one Harbor trial. |
| trial | Harbor's name for the same thing. Every episode's `uri` points at its trial directory, so results stay auditable. |
| agent, harness | Whatever does the work inside the task container: `claude-code`, `codex`, your own `BaseAgent`, or a plain script. See [running agents](running-agents.md). |
| oracle | Harbor's built-in agent that runs a task's reference `solution/`. Used to prove a task's pipeline end to end. |
| autoresearch | One open-ended optimization problem with a continuous score, worked at for a whole budget, with a judge scoring every submission. One of the two regimes. See [design](design.md). |
| session | One task's run against its judge, from the first submission to finalization. The final judge locks it, after which submissions are refused. |
| regime | The shape of the work being measured: autoresearch (one open-ended problem) or a stream of tasks. Either regime works with any form of persisted state. See [design](design.md). |
| self-evolving, continual learning | Something the agent learned persists past the run that produced it, as memory, skills, an evolved harness, or weights. It can show up in either regime, and reef-eval measures its effect on the scores. |

## Scoring

| Term | Meaning |
|---|---|
| judge | In autoresearch tasks, the HTTP sidecar that holds all scoring code and data and scores every submission. The agent can reach it and nothing else. |
| submission | One candidate solution POSTed to the judge. Each task caps how many are allowed. |
| final judge | An optional `final.py` on the judge with hidden tests. Runs once, on the best submission, and locks the session. |
| verifier | Harbor's scoring step at the end of a trial: for judge tasks it asks the judge for the final grade; for pass/fail tasks it runs the task's `tests/`. |
| reward | The trusted score of an episode, as reported by the verifier. |
| reward hacking | Raising the score without doing the task: reading the scoring code, editing the log, or overfitting to a scorer the agent can query without limit. The judge holds all scoring code and data in its own container, the submission budget caps how often the agent can probe it, and `final.py` grades the best submission once on hidden tests. See [design](design.md#reward-hacking-scoring-runs-outside-the-agents-container). |
| trace | Per-submission scores from the judge, stored next to the episode row; the raw material of the anytime curve. |
| budget | What an episode may spend: time, evals, or tokens. Evals need a judge, so that axis applies to autoresearch tasks. Set on the run, delivered as `REEF_EVAL_*` env vars, recorded as `budget_*` tags with actuals in `used_*` columns. See [budgets](get-started.md#budgets). |

## Results

| Term | Meaning |
|---|---|
| Lab | A directory holding the results store; `Lab.run` executes one episode into it. See [get started](get-started.md#the-python-api). |
| store | The append-only SQLite table behind a Lab. Two row kinds: `episode` and `trace`. |
| tags | Free-form dimensions on every row (model, suite, budget, stream). Any key you pass becomes a column; metrics document the columns they expect. |
| key | An episode's stable id, derived from (task, agent, tags, overrides) or passed explicitly. A key that already has a row is skipped, which is how re-running resumes. |

## Streams

| Term | Meaning |
|---|---|
| stream | An ordered task list run under one agent with a state directory carried between tasks. See [streams](get-started.md#streams). |
| position | An episode's index within its stream, recorded as a tag. |
| state directory | The carried directory, mounted into every task's container as `$REEF_EVAL_STATE_DIR`. The agent writes whatever it wants its future self to know; reef-eval never reads it. |
| snapshot | The state directory saved after each position. The next position starts from it, which makes starting states deterministic and resume clean. |
| variant | A digest of a stream's setup (agent, tags, budget, overrides). Together with the name and the task list it decides which state and keys a stream gets, so two streams that differ in any of them stay separate. |

## Metrics

| Term | Meaning |
|---|---|
| anytime | The property that a run has a usable answer at every moment, improving as it goes. The anytime curve is the best-so-far score over time. See [metrics](metrics.md). |
| anytime score | The anytime curve's time-average, from `metrics.auc`. Comparable across runs only when they are averaged over the same window. |
| learning curve | Score over stream position, the continual-learning progress curve. Not the training-set-size curve the term means elsewhere in ML. |
| transfer | What carrying state was worth: a stream task's score minus the same task run on its own. CL-Bench's gain metric, not the forward transfer of the continual-learning literature. |
| forgetting | How much a revisited task degraded: the best of its earlier visits minus its last. Positive means the agent forgot. |

## Environment

| Term | Meaning |
|---|---|
| allowlist | `network_mode = "allowlist"` in `task.toml`: the container reaches the hosts in `allowed_hosts` and nothing else. The usual setting is the judge alone. |
| allowed_hosts | The hosts a task permits. `extra_allowed_hosts` widens the set for one run without editing the task. See [running agents](running-agents.md#network-access). |
| sidecar | A second container beside the agent's, in the same task. The judge is one, and so is the process that enforces the allowlist. |
