# autoenvscaling

**Status: beta.** AutoEnvScaling remains under `recipes/beta/autoenvscaling/` until complete, reproducible learning results are published. The tests validate the generator, the authors, the checks and the band against stand ins; they do not run pi, Claude Code or a model, and they do not establish learning performance. See [issue #422](https://github.com/Human-Agent-Society/reef/issues/422) for this classification.

AutoEnvScaling (preprint forthcoming) as a Reef task generator: a coding agent rebuilds recorded coding-agent sessions as Harbor tasks. The agent writes one task in a workspace from one session and the repository the session changed. It runs the task's oracle and nop check itself through a tool, and revises until the check passes. A pass-rate band marks the tasks that the Reasoning Agent can reach. With `author: prompt`, the same generator runs the generator service's one-reply prompt Designer on the same sessions, as the baseline ([the prompt Designer baseline](#the-prompt-designer-baseline)).

## Results

These results measure the generated tasks. They are not learning results: no model has trained on these tasks yet.

In every run, the agent and the prompt Designer use `gpt-6-sol`, `gpt-6-luna` runs the tasks as a strong solver, and the tasks build and run in E2B. The runs are from 2026-09-30 to 2026-10-13.

**Agent mode and the prompt Designer.** 77 SWE-Together sessions, 3 rebuilds each, so 231 tasks for each mode:

| | `author: prompt` | `author: agent` |
|---|---|---|
| Valid: the oracle passes and nop fails | 30% | **100%** |
| Faithful: SWE-Together's fix passes the rebuilt verifier, and the rebuilt solution passes SWE-Together's tests | 6% | **24%** |

**The task checks.** Agent mode on SWE-chat sessions. "Without checks" has no behaviour rule and no test review. "With checks" has the behaviour rule, the review in the agent's check tool, and `review: refuse` ([the task checks](#the-task-checks)):

| | Without checks | With checks |
|---|---|---|
| Sessions | 20 | 84 |
| Tasks that ran and that `gpt-6-luna` solves in 1 or more of 2 rollouts (fair tasks) | 6 of 19 (32%) | **41 of 51 (80%)** |
| Sessions that give a fair task | 30% | **49%** |
| Qwen3.5-9B rollouts of the fair tasks that solve them (4 rollouts for each task) | 3 of 23 (13%) | **36 of 91 (40%)** |

On the 15 sessions that both arms ran, the arms give 5 and 6 fair tasks. The checks refuse more tasks, but the tasks they admit are much more often solvable, by the strong solver and by a 9B model.

## Layout

```text
beta/autoenvscaling/
  sources.py        the sources file (session records and their repositories), the session rendered for the agent
  author.py         the author contract; the agent: pi in an E2B sandbox behind a gateway that records its calls; the prompt Designer
  author_tools.ts   the agent's task_check tool
  claude_author.py  Claude Code as the agent, signed in with its owner's OAuth token, behind a gateway that serves only the checks
  task_check.py     Claude Code's task_check command
  checks.py         the task checks: Harbor's oracle and nop agents, the contamination check, the test review and the solver calibration
  generator.py      AutoEnvScalingGenerator: the session cursor, admission, the task checks, the band, the parents and the seeds
  tmax.py           the command that converts extracted TMax tasks into seed tasks
  tmax_verifier.sh  the verifier of a converted TMax task: the fraction of its tests that pass
  examples/tinker/serve.yaml   a Tinker deployment with the generator service and the generated-task recipe
```

AutoEnvScaling is a task generator. The deployment selects it in `pipeline.record2sample.config.generator`. The built-in generated-task recipe (`reef.recipe.generated_tasks:GeneratedTaskRecipe`) hands the tasks to the training component, which runs them and trains on them. Its stages and their settings are described in [generated tasks](../../../docs/reference/configuration.rst#generated-tasks).

## The generator

`AutoEnvScalingGenerator` implements `reef.recipe.generated_tasks.TaskGenerator`. One round of the recipe is one generation:

1. `generate` takes the next `count` sessions of the sources file. The author rebuilds each session as a task (see [the author](#the-author)).
2. Admission holds each task to the generator service's parser and authoring checks, the rules a prompt Designer's reply is held to. Its Dockerfile gets a last step that keeps the repository's history up to the base commit only (below).
3. `validate` writes the task under the generator service's tasks root, runs the task checks there, and removes the task again. The first check is Harbor's oracle and nop agents: the reference solution must score 1, and doing nothing must score below 1. With `contamination` on, the contamination check comes next ([contamination](#contamination)). The test review follows ([the task checks](#the-task-checks)). With `calibration` on, the current solver's rollouts come last ([solver calibration](#solver-calibration)). A duplicate task is invalid.
4. The recipe saves each valid task with its train/eval split. Later the training component runs `rollout.rollouts_per_task` rollouts of each training task in a task job; the recipe runs no eval tasks.
5. When every training task of the generation has a result, `record_results` computes the pass rate of each task that ran and its place against the band, and writes the generation's report. A task that could not run is an outage, not a band measurement. It sends no Designer reports, because the Designer here is not trained.

Until `record_results` closes a generation, the generator keeps its sessions and check results in `open-<generation>.json` under its state directory, named by the recipe's generation number, so a restart can still close it.

An accepted task names two source records: the author's last recorded call, the one that wrote the task, and the session's record. A task that Claude Code wrote names one source record, the session's, and its transcript ([Claude Code as the author](#claude-code-as-the-author)). A task rebuilt from a repository gets 4 CPUs, 4 GB of memory, 10 GB of storage, 20 minutes to build, an hour for the solver and 7.5 minutes for the verifier.

The last step of every task's Dockerfile keeps the solver from reading the change it must make out of the repository's history. It removes every remote and every ref but HEAD's, and prunes the rest, so no commit after the base commit is left in the image.

### The session cursor

The generator chooses its own sessions. It does not use the request's description, and it refuses a request that carries source records. Leave the record stage's `source_records_per_round` at 0, its default.

`cursor.json` in the state directory keeps the next generation number and the next position in the sources file. The generations walk the whole file and then start again. A session counts as taken when its author finishes, whatever the result, so a restart continues with the next session. A run that a reload cancels takes no session. Task names hold the generation number and the session's index, not the scenario, so give each scenario its own generator service and tasks root: two scenarios on one service write tasks with the same names, and the service refuses the second one.

### Sessions without a task

The generation's report gives each session the stage where it stopped:

| Stage | Meaning |
| --- | --- |
| `source` | the session's record is gone or holds no session, or its repository or base commit is not there (deleted, private or a wrong URL) |
| `authoring` | the author made no recorded call, or admission refused its task |
| `outage` | the author could not run; the generation ends here |
| `credential` | the task of Claude Code as the author held its token; the generation ends here |
| `check` | a check found a defect in the task, or the task is a duplicate |
| `unchecked` | a refusing check could not run, so `validate` refused the task |
| `calibration` | with `calibration: refuse`, the solver's calibration rollouts put the task below or above the band, or their rewards vary too little |
| `run` | the recipe could not run the task |
| `band` | the task ran, and its pass rate is outside the band |
| `held_out` | the valid task is in the eval split, so the recipe did not run it and it has no band position |
| `unreached` | the round ended before it ran the task |

Three kinds of failure end differently:

- **An outage** is a failure the author does not cause: the session or the repository could not be fetched, the sandbox did not start, or a model call failed while the run's budget still had calls. The generation ends, and its report keeps the outage as its error. Validated tasks made before it stay in the task store, and its training tasks still run. The outage's session counts as taken.
- **The agent's own limits** are not outages: a run that spent `max_model_calls`, ran out of time or hit its reply budget leaves its task as it stands, and admission checks it like any other task.
- **An oracle check that cannot run**, because the generator service failed, ends the round. `GET /reef/status` shows it as the round's last error, the generation's report names it, and the task store keeps the tasks that the round published before the failure. So does a rollout request of a refusing calibration that the generator service fails ([solver calibration](#solver-calibration)). A test review that cannot run is recorded instead (see [the task checks](#the-task-checks)).

### The report and the task store

`state_dir/<scenario>/generation-<n>.json` keeps each session (its position, source record, Designer record, task name, stage, refusal and check results) and each task that ran (its path under the tasks root, rollout rewards, pass rate, band position and check results). The recipe saves each valid task and its train/eval split in the scenario's task store before its rollouts run. Related source sessions keep the same split. The training band does not filter the store; the calibration in `refuse` mode does.

The generator's `config` keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `generator_url` | required | the generator service, `${endpoints.generator}` |
| `state_dir` | required | where each scenario keeps its cursor, reports, agent runs and repository snapshots |
| `sources_path` | required | the sources file; optional with `seed_tasks_path` ([seed tasks](#seed-tasks)) |
| `author` | `agent` | `agent`, or `prompt` for the baseline |
| `author_harness` | `pi` | the agent: `pi`, or `claude-code` ([Claude Code as the author](#claude-code-as-the-author)); `claude-code` needs `author: agent` |
| `claude_model` | `opus` | with `claude-code`, the model Claude Code asks for (`--model`) |
| `claude_token_env` | `CLAUDE_CODE_OAUTH_TOKEN` | with `claude-code`, the environment variable that holds the Claude Code OAuth token |
| `count` | 8 | sessions in each generation |
| `band_low`, `band_high` | 0.25, 0.75 | the pass-rate band; null is no bound |
| `agent_model` | required | the Designer's model |
| `agent_reef_url` | required | the Reef that serves `agent_model`, records its calls and holds the sessions |
| `agent_token_env` | `REEF_TOKEN` | the environment variable that holds that Reef's token |
| `agent_api` | `openai` | `openai`, `responses` or `anthropic`; the prompt Designer takes `openai` or `responses` |
| `max_checks` | 6 | the agent's checks that ran, in one run |
| `max_model_calls` | 250 | the agent's model calls in one run; Claude Code's turns (`--max-turns`) |
| `agent_timeout_seconds` | 2700 | the longest one agent run lasts |
| `agent_sandbox` | `e2b` | the cloud sandbox the agent runs in: `e2b` or `modal` |
| `agent_sandbox_options` | none | that provider's options, such as `e2b_template`, or `modal_app` and `modal_image` |
| `adaptive` | false | vary tasks outside the band in later generations ([adaptive generation](#adaptive-generation)); needs `author: agent` |
| `variant_share` | 0.5 | with `adaptive`, the largest share of a generation's slots that go to variants, rounded down; a share that gives no slot is refused |

Set `eval_fraction` and `split_seed` in `pipeline.record2sample.config`, outside the generator configuration. Their defaults are 0.0 and 0. The task store keeps each published task's split, not these settings. After a restart, published tasks keep their split and new tasks are placed with the settings in force. Reef does not refuse a changed `eval_fraction` or `split_seed`, so keep both fixed for the life of a scenario.

## The author

The author is pi, Reefine's pinned coding agent, in an E2B sandbox. It reads the recorded session, the repository at the base commit (a fresh one-commit copy with no history, cached under `state_dir/<scenario>/repositories`) and the checkout path. It writes the task under `./task/`: `instruction.md`, `environment/` (a Dockerfile that clones the repository at the base commit into the checkout path), `tests/test.sh`, `solution/solve.sh`, and an optional `hint.txt`.

Its one tool, `task_check`, runs the workspace's task through admission and Harbor's oracle and nop agents on the generator service, and answers with both rewards and, for a failing task, the reason. A check that could not run uses up none of `max_checks`.

The agent holds no credential. Its model calls go through a loopback gateway that sends each one to `agent_reef_url` under the generator's scenario, with the token from `agent_token_env` and the tags `role: designer`, `generation` and `index`, so each call is an inference record. A run lasts at most `agent_timeout_seconds` and makes at most `max_model_calls` model calls. Each run keeps the task and pi's session log in `state_dir/<scenario>/runs/<generation>-<index>/`.

### Claude Code as the author

With `author_harness: claude-code`, the agent is Claude Code, at the version Reef pins, in the same sandbox, with the same inputs, rules and checks. It is signed in with a Claude Code OAuth token that its owner made with `claude setup-token`:

```yaml
pipeline:
  record2sample:
    config:
      generator:
        module: recipes.beta.autoenvscaling:AutoEnvScalingGenerator
        config:
          # The rest as in examples/tinker/serve.yaml.
          author_harness: claude-code
          claude_model: opus
```

```bash
export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...   # the variable claude_token_env names
```

- Claude Code sends every model call itself, straight to Anthropic, and Reef records none of them. The generator still reads the sessions from `agent_reef_url` with the token from `agent_token_env`, and the test review still calls `agent_model` there through `agent_api`.
- The author passes the token through the existing sandbox environment. The sandbox launcher puts environment values in bubblewrap's command arguments, so the token can appear in process arguments and launcher diagnostics. Claude Code removes it from the environment of its shell commands.
- The author redacts its log and error messages and replaces the token in retained run files and file names. It refuses a check of a task that holds the token, including through a symbolic link. Such a task is not admitted, and generation ends with stage `credential`: the token may have left the sandbox. Make a new token before the next run.
- The agent checks its task with the command `python3 "$REEF_AUTHOR_CHECK"`, the same check as pi's tool. The command waits up to 9 minutes; for a longer check it prints `state: running` and the `--wait <check_id>` command that waits again.
- A run ends at `agent_timeout_seconds` or after `max_model_calls` turns, Claude Code's own limit; either way its task stands. Any other error of Claude Code's (a spent plan, a refused token) is an outage.
- The task names one source record, the session's, so a variant keeps its parent's split. Its metadata names the transcript, `designer_transcript`, relative to `state_dir/<scenario>/`, and `designer_harness: claude-code`. Each run keeps Claude Code's transcript under `claude/projects/` in its run directory. Reef deletes Claude Code's other state there, which can hold the account's email.

Reef refuses at startup a token that is missing or is no Claude Code OAuth token (`sk-ant-oat...`). It also refuses an `agent_sandbox_options.env_from` variable that would win over the run's own setup: an `ANTHROPIC_*` or `CLAUDE_*` variable, a `*BASE_URL*` or `*PROXY*` variable, a variable the run sets (such as `HOME`, `DISABLE_ERROR_REPORTING` or `BASH_MAX_TIMEOUT_MS`), or a variable whose value holds the token.

## The band

The pass rate of a task that ran is the mean reward of its training rollouts. The task is inside the band when the pass rate lies in `[band_low, band_high]`, 0.25 to 0.75 by default. A task outside the band stays under the tasks root, so a later analysis can run it again. It stays in the task store and is recorded with stage `band` (`outside the solver band: pass rate 1.00`) and its path in the generation's report. A null bound is no bound; with both null, every task that ran is inside the band.

The band does not decide what trains. The recipe trains on the training rollouts of every valid task, before the band is known. Its `task_group` objective gives an advantage of 0 to every rollout of a group whose rewards are all equal. With 0 or 1 rewards and four rollouts per task, a task outside the default band is such a group, so it trains nothing, but it still takes a place in a batch.

## Solver calibration

The band marks a task after the recipe trained on it. The calibration measures the task in `validate`, before the recipe saves it: the current solver plays the task, and the task is kept only when the solver sometimes solves it. This is the Solver-Guided Calibration of the AutoEnvScaling preprint, rebuilt on the generator service's rollouts. The rule, with its defaults:

1. 4 rollouts run. The rollouts that could not run (the sandbox, the agent or the verifier failed before a score) are left out and asked for once more. A rollout that ran without a reward counts as 0.
2. When the mean reward is within 0.15 of a band edge, on either side, and fewer than 8 rollouts scored, the rest up to 8 run, with the same one retry. A null bound is no edge.
3. The task is kept when the mean reward is inside `[band_low, band_high]` and the population standard deviation of the rewards is at least 0.1. With fewer than 2 scored rollouts, the calibration could not run (below).

Upstream also gives a task below the band one rollout with hints from the reference solution. Only its proposer revision feedback reads that reward. This generator has no such feedback, so the calibration runs no hint rollout.

The calibration is the last task check in `validate`, so it runs only when the oracle check, a refusing contamination check and a refusing test review all pass. It asks the generator service (`POST /rollouts`) for rollouts of the task's copy on the arm `calibration`, with `calibration_model`. Without it, the rollouts use the generator service's default model: its deployment's `inference.upstream-model`, else its `model-path`. In the example's Tinker deployment, that is the policy in training, `Qwen/Qwen3-8B`. When you save the tasks first, as below, on [`recipes/basic/external-provider.yaml`](../../basic/external-provider.yaml), it is that provider's `gpt-4o`, so the calibration measures `gpt-4o` and not the policy; name the solver in `calibration_model` when that deployment's Reef can serve it. The agent is the generator service's own (`generator.agent`). Reef keeps no record of these rollouts, and nothing trains on them.

The calibration has a mode:

- `off`, the default: no rollout runs, and the generation report is the same as without the calibration.
- `measure`: the calibration runs and its result is recorded; it never refuses a task and never ends the round. Use it to compare the calibration's means with the training pass rates before the calibration refuses anything.
- `refuse`: a task below or above the band, or with rewards that vary too little, is invalid with stage `calibration`. That stage is apart from `check` (a defect) and `band` (the training pass rate).

| Key | Default | Meaning |
| --- | --- | --- |
| `calibration` | `off` | `off`, `measure` or `refuse`; a bare YAML `off` reads as false and means `off` |
| `calibration_rollouts` | 4 | the rollouts asked for first; an integer of at least 2 |
| `calibration_max_rollouts` | 8 | the most rollouts that score; an integer of at least `calibration_rollouts` |
| `calibration_margin` | 0.15 | the distance from a band edge that asks for more rollouts; in [0, 1] |
| `calibration_min_std` | 0.1 | the least population standard deviation of the rewards; in [0, 0.5] |
| `calibration_model` | null | the model the calibration rollouts ask for; null is the generator service's default model |

The calibration uses the band of `band_low` and `band_high`. Each key is checked also when the mode is `off`.

The generation report keeps the calibration's settings under `calibration`, beside `band`, when the mode is not `off`. Each task's check results get a `calibration` entry: the mode it ran under, the rewards, their mean and standard deviation, the rollouts asked for, the rollouts that could not run, whether more rollouts ran, the place against the band, the result (`kept`, `below`, `above`, `no_variance` or `outage`) and the errors of the first 3 rollouts that failed.

When fewer than 2 rollouts scored, the task's difficulty is unknown, and the calibration could not run, with the result `outage`. The cause can be the task itself, for example a task image in which the solver's agent cannot start, so the round goes on: in `refuse` mode the task is invalid with stage `unchecked`, and in `measure` mode it stays valid. A rollout request that the generator service fails is an outage of the service. In `refuse` mode it ends the round, as an oracle check that cannot run does; the task stays pending, and the next attempt calibrates it again from the start. In `measure` mode the calibration could not run, and the task stays valid.

The recipe gives a task its train/eval split after `validate`, so eval tasks are calibrated too. Eval tasks run no training rollouts.

Cost: each candidate runs up to `calibration_max_rollouts` full rollouts that score, and one more for each rollout that could not run, `generator.concurrency` (2 by default) at a time. A repository task gives the solver up to an hour per rollout. `validate` checks one candidate at a time, so a generation of 8 candidates can take hours more. All attempts of one generation must finish within `pipeline.record2sample.config.generation_deadline_seconds` (1800 by default), so raise it when the calibration is on. A restart calibrates a pending candidate again from the start.

No results with the calibration are published yet.

## Adaptive generation

With `adaptive: true`, the solver's results steer later generations, as in RFC #811. When a generation closes, each training task whose pass rate fell outside the band becomes a parent: the generator keeps its files under `state_dir/<scenario>/parents/` and its pass rate in `adaptive.json`. A later generation gives up to `variant_share` of its slots to the newest parents. The slot count is rounded down after adding a tolerance of 1e-9 slots, so floating-point error does not turn 29 slots into 28. The agent gets the parent and its pass rate in its inputs and writes a harder task for a parent above the band, or a simpler one for a parent below it, from the same session. So a variant keeps its parent's source and its split. The other slots rebuild the next sessions of the walk, and a variant takes no session from it.

Eval tasks, outages and tasks whose mean reward is outside [0, 1] never become parents, a parent gets one variant slot whatever becomes of the variant, and the generator keeps at most 256 parents, dropping the oldest. Parents with missing files or sessions are skipped but still count toward the limit until they are dropped. A replayed result adds nothing: a closed generation is not closed again, and one closed again after a crash adds each parent once. The proposals of a variant name it in the report: `position` -1, `parent_task` and `direction`. With `adaptive` off, the generator keeps no parents and writes the same reports as before. With `calibration: refuse`, a task outside the band is refused before it trains, so it never becomes a parent: the parents are then the admitted tasks that the solver outgrows, or falls behind on, during training. With `calibration: measure`, every task is admitted and trained, so a task the calibration placed outside the band can still become a parent. Proposer training and retiring tasks from the pool are not part of this mode.

A variant keeps its parent's limits: `[agent]` and `[verifier]` `timeout_sec` and `[environment]` `cpus`, `memory_mb`, `storage_mb` and `gpus` are the parent's, whatever the author wrote. A variant is harder or simpler by what it asks, not by more time or a larger machine, so its pass rate stays comparable with its parent's place in the band. Without this, a variant of a TMax seed (1200 s) took the authored task's 3600 s.

### Seed tasks

With `seed_tasks_path`, an adaptive generator also starts from a directory of Harbor tasks, the seeds, such as tasks converted from TMax-15K ([below](#convert-tmax-tasks)). Seeds need no sessions, so `sources_path` is then optional:

```yaml
pipeline:
  record2sample:
    config:
      generator:
        module: recipes.beta.autoenvscaling:AutoEnvScalingGenerator
        config:
          # The rest as in examples/tinker/serve.yaml, without sources_path.
          adaptive: true
          seed_tasks_path: /data/tmax-seeds
```

| Key | Default | Meaning |
| --- | --- | --- |
| `seed_tasks_path` | null | a directory of seed tasks that reef wrote; needs `adaptive: true` |

Each directory in `seed_tasks_path`, except one whose name starts with a dot, is a seed and must be a task that reef wrote, as the converter does. The generator reads every seed when it starts and checks its digest. A directory that is not such a task fails the config, and the error names it.

- **The slots.** A generation fills its slots in this order: variants of the newest parents, then the seeds that no generation took yet, then the next sessions. A seed takes its slot as it is: the task is the seed under the generation's task name, and no author runs. A seed that was a variant in an earlier run loses that run's `parent_task` and `variant`, so it is a seed here. Each seed is used once, in directory-name order. `cursor.json` keeps the next seed (`next_seed`) once a generation returns its tasks, so a restart continues with it, and an attempt that fails or is cancelled takes no seed; keep the directory fixed for the life of a scenario. Without a sources file, a generation ends when its variants and seeds run out, so it can be empty. The recipe counts an empty generation as one of its `rounds` and starts the next round at once, so the rounds can run out before the seeds' pass rates make parents. Give enough seeds that each generation finds unused seeds until the first parents exist, or give a sources file too.
- **The seed check.** `validate` runs the seed check, then the contamination check when it is on. A seed gets no test review and no calibration: a seed outside the band is the parent a variant starts from. The seed check runs Harbor's oracle trial. It refuses a seed whose environment does not build, or whose verifier writes no reward. A converted seed's solution does nothing, so its oracle trial scores the untouched environment, and the seed passes when that score is below 1. A seed whose solution scores 1 passes when doing nothing scores below 1. A refused seed gets stage `check`.
- **Training and parents.** A seed in the train split rolls out like any training task. Inside the band it stays a training task, as in upstream's static TMax pool. Outside the band it becomes a parent. Its variant has no session: the agent reads only the parent and `variant.md` in its inputs, its workspace starts with a copy of the parent's `environment/`, and its rules ask for a real reference solution, since the seed's does nothing. A harder variant needs a longer horizon or one more skill; a simpler one has fewer compounding steps, a clearer starting state or more scaffolding. The variant's Dockerfile gets no history step, since it has no repository. Upstream varies its first TMax tasks with no rollout and a pass rate of 0.5; here the first rollout of each seed measures it first.
- **The split.** A seed's source records are its own, then `seed-` and the first 16 hex digits of its digest. A converted seed has no records of its own, so each such seed is a split group of its own. A seed from an earlier run's session task keeps that session, so it shares a split with the tasks of the same session. A variant keeps the seed's `seed-` record, so it joins the seed's split. An eval seed is held out: it never rolls out, so it is never varied.
- **The report.** A seed's proposal has `position` -1, `seed_task` (the seed's directory name) and the seed's source record as `source_record_id`. The proposal of a variant of a seed, and the variant's metadata, also name the seed in `seed_task`. The report keeps `next_seed`, and the `seed` check result keeps `oracle_reward` and `nop_reward`. `adaptive.json` keeps each parent's `seed_task`, empty for a parent from a session. Without `seed_tasks_path`, the reports and `cursor.json` have no seed fields.

#### Convert TMax tasks

TMax-15K (`allenai/TMax-15K`) is a `tasks.zip` with one directory per task. Download and unzip it yourself, then convert the tasks into a seed directory:

```bash
unzip tasks.zip -d tmax-tasks
python -m recipes.beta.autoenvscaling.tmax --tasks tmax-tasks --out /data/tmax-seeds --select ids.json
```

`--select` names a JSON list of task ids, or upstream's selection file (a list of objects with a `task_id`; the other keys come from each task's `task.json`); without it, every task directory with a `task.json` is converted. The command makes no network call. It exits with 1 when it refuses a task, and `conversion.json` in the output directory lists the converted tasks, with the `container.def` sections each leaves out, and each refused task with the reason. A second run keeps the tasks it wrote before. The conversion is reimplemented from upstream's `baselines/tmax.py`:

- `container.def` becomes the Dockerfile: `From:` becomes `FROM`, each `%environment` export an `ENV` line, each `%files` entry a `COPY` of its file in the task directory, and `%post` the build script `tmax_build.sh`. Of the files with the entry's name, its file is the one whose path ends with the most parts of the entry's source path. The file goes under `fixtures/` by the last three parts of that path, with each run of characters that a task's file name cannot hold made an underscore, so a variant's task can keep it. The Dockerfile then installs tmux, which the solver's agent needs, and pytest, which the verifier needs, since neither has network.
- `task.json`'s `description` becomes `instruction.md`, and `test_final_state.py` becomes `tests/test_outputs.py`. The verifier, `tmax_verifier.sh`, scores the fraction of those tests that pass with pytest's own JUnit report. It scores 0 when the report is missing or holds no test, never counts a skipped test as passed, removes reports and rewards left before it runs, and reads no pytest config file or `conftest.py` above the tests.
- TMax ships no reference solution, so `solution/solve.sh` does nothing. Upstream's initial-state check is not used.

Limits:

- **Text fixtures only.** A reef task holds text files, so a task with a binary fixture is refused. So is a task with a fixture that no file or several files match equally, two fixtures with one name under `fixtures/`, no `From:` image, no description or no final-state tests.
- **No failure summary yet.** `variant.md` gives the parent's pass rate (its mean reward, since the verifier gives part marks) and the direction, not the solver's most common failure as upstream's assignment does. The generator receives only the rewards of a task's rollouts, so this needs a change to the shared generated-task results first.
- **One rollout per seed.** Each seed in the train split costs one first rollout (`rollout.rollouts_per_task` rollouts) before its pass rate is known.
- **No network.** A converted task has `network_mode = "no-network"`, so terminus-2 cannot install asciinema to record its session. On `e2b` that install hangs until the sandbox's command timeout fails the rollout; set `record_terminal_session: false` in the agent's `kwargs`, in `rollout.agent`, and in the harness of an artifact evaluation that runs such tasks. On `docker` the install fails at once and the rollout runs without a recording.

## The task checks

A verifier written from a session tends to grade the session's own implementation: the wording of a report, or a function or a button the instruction never names. A different correct solution then fails. Two measures keep such tasks out:

- **The behaviour rule.** Both authors get it with the rebuild rules: the tests check behaviour that the instruction names, never free-text wording, never a name the instruction does not give, and never a requirement the instruction does not state. `behaviour_rule: false` turns it off.
- **The test review.** After Harbor's oracle and nop check, and the contamination check when it is on, `validate` makes one model call that reads `instruction.md` and every file under `tests/` and `solution/`. It lists each requirement that the tests check and the instruction does not state, of kind `wording`, `unnamed_interface` or `unstated_requirement`. The call goes where the author's calls go, through `agent_api` (`openai` or `responses`).

The review has a mode: `off`, `measure` (run it and record the result, never refuse) or `refuse`. A refusing review that lists a requirement gives the task stage `check`. One that could not run, because the call failed or the reply is not the JSON asked for, gives it stage `unchecked`. The generation's report keeps the review's result for each task.

With `review: refuse`, the agent's own `task_check` also runs the review once the oracle passes. The check passes only when the review finds nothing, and the agent sees the findings, so it can revise before it finishes. `validate` runs the review again. A review that only measures runs in `validate` alone.

| Key | Default | Meaning |
| --- | --- | --- |
| `behaviour_rule` | true | give both authors the behaviour rule |
| `review` | `refuse` | the test review's mode; a bare YAML `off` reads as false and means `off` |
| `review_model` | `agent_model` | the review's model |

The review needs `agent_api: openai` or `responses`; with `anthropic`, set `review: off`.

### Contamination

A generated task can copy the instruction of a benchmark task that is kept for evaluation. An evaluation on that benchmark then measures what the policy saw in training. The contamination check compares the `instruction.md` of each task with the instructions of a set of reference tasks. It runs in `validate` right after the oracle check, before the test review and the calibration, so a contaminated task costs no review call and no calibration rollout. It makes no model call and runs no sandbox.

The rule is upstream's, which takes the 13-gram size from TMax (Ivison et al., 2026, `allenai/TMax-15K`):

1. Each instruction loses every line that contains `canary`, in any case. Benchmark tasks can carry the same canary string, and it says nothing about the task.
2. The rest is lowercased and split into word tokens (`\w+`), so case, punctuation, Markdown and whitespace do not change a match.
3. The task is contaminated when its instruction shares a run of 13 consecutive tokens (a 13-gram) with the instruction of one reference task. Each reference task is compared on its own, so a 13-gram never spans two reference tasks. An instruction of fewer than 13 tokens has no 13-gram.

The check compares text only. It does not find a paraphrased copy.

Each `instruction.md` under `contamination_tasks_path` is one reference task, named by its directory under that path. A downloaded Harbor dataset, such as Terminal-Bench, is such a directory. The generator reads the directory when it starts, so a missing directory, or one with no `instruction.md`, fails the config before any session is rebuilt.

The check has a mode: `off`, `measure` (run it and record the result, never refuse) or `refuse`. A refusing check that finds a shared 13-gram gives the task stage `check`, with the error `instruction.md shares a 13-gram with reference task <name>: "<the 13-gram>"`. The check cannot fail to run, so it never gives stage `unchecked`.

| Key | Default | Meaning |
| --- | --- | --- |
| `contamination` | `off` | the contamination check's mode; a bare YAML `off` reads as false and means `off` |
| `contamination_tasks_path` | null | the directory of the reference tasks; required when the mode is not `off` |
| `contamination_ngram` | 13 | the number of consecutive tokens that a match shares; an integer of at least 1 |

Each key is checked also when the mode is `off`. The directory is read only when the mode is not `off`.

Each task's check results get a `contamination` entry. Its details hold `n`, `tasks_path`, `reference_tasks` (the number of reference tasks), `shared_ngrams` (the number of distinct shared n-grams) and `hits`. `hits` keeps the first 5 shared n-grams in the order of the instruction, each cut to 200 characters, with the first reference task by name that holds it.

The agent's own `task_check` does not run this check, and the agent's rules do not name it. The check also does not read the scenario's task store yet: a generator gets only its config, and a generated-task scenario holds no imported benchmark tasks. When a recipe evaluates its checkpoints on imported eval tasks, a later change can make those tasks the reference set.

## The prompt Designer baseline

To compare Designers on identical inputs, set `author: prompt`. The author is then the generator service's one-reply prompt Designer, the one SPADE uses, on the same sources. It makes one model call with the repository, the session cut to 240,000 characters (tool results first, never the user's words), the agent's rebuild rules and the Designer's reply format. The reply's JSON block is the task. The call goes where the agent's calls go, through `agent_api`, which must be `openai` or `responses`. The agent's own settings go unused, and no E2B key is needed.

Everything else is the same for both authors: the cursor, admission, the checks, rollouts, generation reports and the task store. SPADE keeps every task that passes its checks, so the baseline runs with the band off:

```yaml
pipeline:
  record2sample:
    config:
      generator:
        module: recipes.beta.autoenvscaling:AutoEnvScalingGenerator
        config:
          # The rest as in examples/tinker/serve.yaml.
          author: prompt
          band_low: null
          band_high: null
```

Run each author as a deployment of its own, with its own state directory: both name their tasks `harbor-<generation>-<index>-autoenvscaling`.

## Run it

List the sessions in a sources file:

```json
[
  {"record_id": "rec-1", "scenario": "sessions",
   "repository": {"url": "https://github.com/org/repo", "commit": "3f2c1ab", "path": "/app"}}
]
```

`record_id` and `scenario` name an inference record on the Designer's Reef (import sessions with `POST /reef/records/batch`). `commit` is the commit the session started from, and `path` is where the task's environment checks the repository out. The Designer's Reef serves `agent_model`, for example a record-only deployment such as [`recipes/basic/external-provider.yaml`](../../basic/external-provider.yaml) on another port. Its `inference.timeout-s` must allow one agent turn. Then:

```bash
export REEF_TOKEN=reef-local REEF_AUTOENVSCALING_STATE_DIR="$PWD/work/autoenvscaling"   # and TINKER_API_KEY, E2B_API_KEY
export REEF_AUTOENVSCALING_SOURCES_PATH="$PWD/sources.json"
export REEF_AUTOENVSCALING_DESIGNER_URL=http://127.0.0.1:8901 REEF_AUTOENVSCALING_DESIGNER_TOKEN=<its token>
reef serve -c recipes/beta/autoenvscaling/examples/tinker/serve.yaml
curl -s -X POST -H "Authorization: Bearer $REEF_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "autoenvscaling"}' http://127.0.0.1:8900/reef/scenarios
curl -s -X POST -H "Authorization: Bearer $REEF_AUTOENVSCALING_DESIGNER_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "autoenvscaling"}' "$REEF_AUTOENVSCALING_DESIGNER_URL/reef/scenarios"
```

`reef serve` creates no scenario on its own. The first `POST /reef/scenarios` above creates the training scenario, and round 0 starts when the recipe's pipeline first looks for job data after that. The second one creates the same scenario on the Designer's Reef, which records the Designer's model calls under it and answers only a scenario that exists. Each generation writes `generations/autoenvscaling/generation-<n>.json` under the state directory. The recipe keeps each task, its split and its rollout result in the scenario's task store.

Optional, for an upstream that serves only the Responses API (a ChatGPT plan): set `agent_api: responses` for the agent, and `model_name: "openai/responses/{model}"` in `generator.agent` for the Reasoning Agent's Harbor agent.

terminus-2 summarizes its history before the model's context window only when it knows the window, and it counts tokens with LiteLLM's tokenizer, which runs about 8 to 10% under a Qwen model's own count. On Tinker it knows neither, so a long episode keeps overflowing the window. Set `model_info` in the `kwargs` of `generator.agent`, of `rollout.agent` and of an artifact evaluation's harness, with `max_input_tokens` well under the window: for a 65536-token window, `{max_input_tokens: 48000, max_output_tokens: 8192, input_cost_per_token: 0.0, output_cost_per_token: 0.0}`.

## Time per session, and training from saved tasks

With `gpt-6-sol`, agent mode takes about 10 to 25 minutes per session, so a generation of 5 sessions takes about 2 to 3 hours before its tasks run. Prompt mode takes about 1 to 2 minutes per session.

The built-in generated-task recipe trains a generation's tasks only after the whole generation is saved, so the first job waits for the first generation. To start training at once, or to train several runs on the same tasks, save the tasks first, then train from them:

1. **Save the tasks.** Run the generator's generations with no training. The recipe starts a round only when its training pipeline asks for job data, so run the generations from a script. A `GenerationJob` run by `GeneratedTaskSamples.run` generates, validates and saves one generation with no trainer, and runs no training rollouts. With `calibration` on, `validate` runs the calibration rollouts on the generator service, with `calibration_model` or that deployment's model (see [solver calibration](#solver-calibration)). Set the generator's `generator_url` to the generator service of a deployment that trains nothing, for example [`recipes/basic/external-provider.yaml`](../../basic/external-provider.yaml) with the example's `generator` section.

   ```python
   import os
   from pathlib import Path

   from recipes.beta.autoenvscaling import AutoEnvScalingGenerator
   from reef.pipeline_context import PipelineContext
   from reef.recipe.generated_tasks import GeneratedTaskRecord2Record, GeneratedTaskSamples
   from reef.recipe.generated_tasks.processor import GenerationJob, RoundSettings, TaskGroupSettings, next_generation
   from reef.storage.samples import TrainerSamples
   from reef.storage.sqlite import SQLiteScenarioStorage
   from reef.storage.tasks import LocalTaskFileStore


   async def save_tasks(generator_config: dict[str, object], rounds: int, state_dir: str) -> None:
       """Run the generator's generations with no trainer: generate, validate, and save the tasks."""
       task_files = LocalTaskFileStore(Path(state_dir) / "task-files")
       storage = SQLiteScenarioStorage(Path(state_dir) / "agent-record", task_files=task_files)
       session = storage.open("autoenvscaling")
       context = PipelineContext("autoenvscaling", task_storage=session.tasks, samples=TrainerSamples(session.samples))
       generator = AutoEnvScalingGenerator.from_config(generator_config, scenario=context.scenario, environ=os.environ)
       settings = RoundSettings(
           rounds=rounds,
           description="coding tasks rebuilt from recorded coding-agent sessions",
           generator_url=str(generator_config["generator_url"]),
       )
       record_stage = GeneratedTaskRecord2Record(context, TaskGroupSettings())
       tasks = GeneratedTaskSamples(context, settings, record_stage, task_generator=generator)
       try:
           for number in range(next_generation(tasks.task_store), rounds):
               await tasks.run(GenerationJob(receipt=f"generation-{number:05d}", generation=number))
       finally:
           tasks.close()
           session.close()
           storage.close()
   ```

   `generator_config` is the generator's `config` from the example, with its environment references filled in. `state_dir` is the deployment's state directory, so the generations save each valid task in the task store that the example's `storage` uses. The band is only recorded in the generation report. A restart continues with the next generation.

2. **Train from the saved tasks.** Start the example in the environment of [Run it](#run-it) with `rounds` at the number of generations that step 1 built, and create the scenario as there. Those generations are complete, so the recipe generates no more: it stores their train tasks in the scenario's sample store and trains them, up to `buffer_size` tasks in one job.

   ```bash
   reef serve -c recipes/beta/autoenvscaling/examples/tinker/serve.yaml --pipeline.record2sample.config.rounds 4
   ```

## Requirements

- For Claude Code as the agent, a Claude Code OAuth token in `claude_token_env`.
- For the agent, a cloud sandbox: an E2B API key in `E2B_API_KEY` and the `e2b` extra (`pip install 'reef-infra[e2b]'`), or with `agent_sandbox: modal`, Modal's credentials (`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, or `~/.modal.toml`) and the `modal` extra. The sandbox runs pi, or Claude Code, from an image Reef builds on first use.
- For the agent, `git` on the Reef host, for the repository snapshots.
- Docker and the `harbor` command line on the generator service's host, as for SPADE.

Known limits: with `agent_reef_url` on the training deployment itself, the agent's calls share the training scenario, and the recipe's record stage keeps the ones no report references in memory. A long run should record them on a separate Designer's Reef, as the example does. No run has used the Claude Code image on E2B or Modal yet, so start `author_harness: claude-code` with one session.
