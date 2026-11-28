# continual_envs

**Status: beta.** Continual environments remain under `recipes/beta/continual_envs/` until complete, reproducible learning results are published. The tests check the generator, its sessions and its generation service against stand-ins. They do not run a generation service, a model or a training run, and they do not establish learning performance.

Continual environments train a policy on Harbor tasks that are built from the failures of its own live traffic:

1. The policy serves traffic through Reef. Each session of the traffic gets a report with a score from a judge, with no verifier.
2. Each round of the generated-task stage takes one window of sessions. Its failed sessions go to a generation service, which mines environments from them, builds and reviews tasks on those environments, and publishes the tasks that pass.
3. The built-in generated-task recipe trains the policy on the tasks. The next window is the traffic of the trained policy, so the tasks follow what the policy still fails.

The generation service owns the method of building tasks: a generation spec says how to mine environments from a failure, how to build a task, and how a review decides. Its build stage keeps what its reviews learn about building tasks, such as a shortcut that a task left open, and gives it to later build sessions. This recipe moves sessions to the service and tasks back. A deployment changes the method by changing the spec.

## Layout

```text
beta/continual_envs/
  sessions.py            the traffic sessions of a round's records, and the record folders that a generation run reads
  generation_service.py  the generation service: Reef Platform's generation API
  generator.py           TrafficTaskGenerator: the sessions of each round to one generation run, and its tasks back
  examples/terminalworld/serve.yaml   Qwen3.5-9B on Slime, one node of 8 GPUs, on a replay of TerminalWorld traffic
```

The deployment selects the generator in `pipeline.record2sample.config.generator`. The built-in generated-task recipe (`reef.recipe.generated_tasks:GeneratedTaskRecipe`) runs the tasks and trains on them. Its stages and their settings are in [generated tasks](../../../docs/reference/configuration-generated-tasks.rst).

## The traffic

A source session is a report whose model calls all carry the tag of `source_tag`: with `source=traffic`, the client sends `x-reef-tag-source: traffic` with each call. The report references the session's last call and holds the score in `score`, from 0 to 1, and an optional `feedback` text. Its metadata does not reach the generation service, so a client can keep its own data there.

A round starts when `min_source_records_per_round` sessions wait, and takes at most `source_records_per_round` of them. With both at the window size, as in the example, each round is one whole window. A round never runs on part of a window, and never on no session.

## The generator

`TrafficTaskGenerator` implements `reef.recipe.generated_tasks.TaskGenerator`. One generation of it:

1. `generate` makes one session of each report of the round and its calls. The sessions with a score of at most `max_score` are failures. A session with no score is not a failure. The failures go to one generation run of `spec`, as one run source of kind `records` named `traffic`: one record folder for each session, `<report ID>/session.json`. The file holds the session's last request, which carries the whole conversation and each tool result, the last reply, the score, the feedback and the number of calls. The generator waits until the run ends, then downloads each task that the run published from its build stage `stage`. Each task gets the record IDs of the session that it names as its source, so the tasks of one session share a train/eval split.
2. `validate` runs Harbor's oracle and nop agents on each task in this host's Docker: the reference solution must score 1, and doing nothing must score below 1. The training rollouts run on this host, so a task whose environment does not build here does not train. The generation service ran the same check in its own sandboxes.
3. When each train task of the generation has a first rollout, `record_results` keeps the rewards of each task in the state document `results/<generation>`.

A run that ends `failed` or `cancelled` stops the round, and the stage tries the generation again up to `generation_attempts` times. A run that has not ended after `run_timeout_seconds` is cancelled. Set `generation_deadline_seconds` of the stage above `run_timeout_seconds`, so that the generator cancels the run before the stage stops the generation.

The generator keeps the run of each generation in its state, so a restart waits for that run and starts no second run. A run that ended `failed` or `cancelled` ends one attempt, and the next attempt of the generation starts a new run:

| Document | Content |
| --- | --- |
| `starting/<generation>` | `{requested_at}`: written before the run starts |
| `run/<generation>` | `{run_id}`: written when the service queued the run |
| `results/<generation>` | the result of each task of the generation |

When the answer of a start is lost, only `starting` is there. After a restart, the newest run of `generator` that was queued after `requested_at` and that no other generation names is the run of the generation. When there is none, the generation starts a run again.

The generator's `config` keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `service_url` | required | the Reef Platform's URL; the routes are under `/api/v1/generation` |
| `spec` | required | the generation spec, `<name>` or `<name>@<version>` |
| `generator` | required | the generation service's generator: every generation of the scenario shares its tables and its stage memory |
| `token_env` | `REEF_ACCESS_TOKEN` | the variable that holds a workspace access token with the scope `generators:write` |
| `stage` | `tasks` | the spec's build stage whose published tasks come back |
| `params` | `{}` | the run's parameters, for example `per_failure` |
| `targets` | `{}` | the run's targets of each stage |
| `max_score` | 0.5 | the highest score of a failure; 1 sends every scored session |
| `max_source_bytes` | 15 MiB | the most bytes of the session files of one run; a session that does not fit is left out and logged |
| `run_timeout_seconds` | 21600 | the longest time of one run |
| `poll_seconds` | 30 | how often the generator reads the state of a run |
| `oracle_check` | true | run Harbor's oracle and nop agents on each task on this host |
| `oracle_environment` | `docker` | the Harbor environment of that check |

## The generation service

The generator calls the Reef Platform's generation API with the workspace access token. The Platform relays each call to its generation Cell:

| Call | Use |
| --- | --- |
| `POST /api/v1/generation/runs` | start a run with the sessions as its `records` source |
| `GET /api/v1/generation/runs?generator=` | find the run of a start whose answer was lost |
| `GET /api/v1/generation/runs/{id}` | read the state of a run |
| `GET /api/v1/generation/runs/{id}/items?stage=&state=verified` | the published tasks of a run: the items of the run's generator that the run wrote last |
| `POST /api/v1/generation/runs/{id}/cancel` | cancel a run past its timeout |
| `GET /api/v1/generation/tasks/{name}/archive` | the files of a published task, by a signed address that takes no token |

A read that fails for a reason that can pass (a 5xx answer, a lost connection, a timeout) is tried again, at most 4 times with a growing, jittered wait. A refusal (4xx) is not tried again. A start is never tried again: the restart path above finds a run that a lost answer queued.

The spec's mine stage reads the source `traffic`: one folder for each session, `sources/traffic/<report ID>/session.json`. Each environment that it mines from a session names the session's folder in its `source_record` field, so each task that it builds names the session as its source.

## Run it

The example trains Qwen3.5-9B on Slime on one node of 8 GPUs of 80 GB: 4 for the actor, 3 for the rollout engines, and 1 for Reef's serving engine, which serves the traffic. Run it in the Reef image ([docker/README.md](../../../docker/README.md)), with this host's Docker for the training rollouts and the oracle checks:

```bash
export REEF_TOKEN=reef-local REEF_CONTINUAL_ENVS_STATE_DIR="$PWD/work/continual-envs"
export REEF_CONTINUAL_ENVS_MODEL_PATH=/models/Qwen3.5-9B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REEF_PLATFORM_URL=https://console.example REEF_ACCESS_TOKEN=rft_...
export REEF_GENERATION_SPEC=terminal-failures REEF_GENERATION_GENERATOR=terminalworld-continual
reef serve -c recipes/beta/continual_envs/examples/terminalworld/serve.yaml
curl -s -X POST -H "Authorization: Bearer $REEF_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "traffic"}' http://127.0.0.1:8900/reef/scenarios
```

Then send the traffic to the scenario `traffic`. Each session runs an agent with the served model, with `x-reef-tag-source: traffic` on each model call, and ends with a report of its score against its last call. `reef.harness.harbor.rollouts.TaskRolloutRunner` with `labels={"source": "traffic"}` and `is_reporting=False` runs such sessions on Harbor tasks; the client then reports its own score with `ReefClient.report`.

Settings the example depends on:

- **Whole windows.** `source_records_per_round` and `min_source_records_per_round` are both 64, so each round is one window of 64 sessions.
- **Generation runs ahead.** `jobs_per_round: 0` starts the next round when the last one ends, so the next window's run builds tasks while the earlier tasks train.
- **The training pool.** `pool_capacity: 256` trains each task again until a review finds its pass rate over its last 32 rewards below 2/32 or above 30/32.
- **The deadline.** `generation_deadline_seconds` (28800) is above `run_timeout_seconds` (27000).
- **No recording.** `record_terminal_session: false`: the tasks have no network, so the recorder cannot install.

## Requirements

- A Reef Platform whose generation Cell runs a spec with a mine stage that reads the run source `traffic` and a build stage named by `stage`, and a workspace access token with the scope `generators:write`.
- Docker on the training host, for the training rollouts and the oracle checks.
- The `harbor` command line, which the oracle check runs.
