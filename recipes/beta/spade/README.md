# spade

**Status: beta.** SPADE remains under `recipes/beta/spade/` until complete, reproducible learning results are published. The tests validate the experience section and the generator's generations against stand ins; they do not establish learning performance. See [issue #482](https://github.com/Human-Agent-Society/reef/issues/482) and [issue #498](https://github.com/Human-Agent-Society/reef/issues/498) for the pieces, [issue #447](https://github.com/Human-Agent-Society/reef/issues/447) for the pipeline they belong to, and [issue #422](https://github.com/Human-Agent-Society/reef/issues/422) for this classification.

[SPADE](https://arxiv.org/abs/2608.19197), self play in adaptive synthetic executable environments, as a Reef task generator: one policy plays an Environment Designer that writes executable environments and a Reasoning Agent that learns in them. Reef knows one task format, Harbor, and writes and checks Harbor tasks and runs their rollouts in `reef.record2sample.generator`: the task contract as a prompt, the served model asked through Reef (every proposal a record with a receipt), the reply held to the authoring rules and written with `reef.core.tasks`, Harbor's oracle and nop agents on the result, and the rollout runner. That runs as the generator service `reef serve` starts beside the HTTP service. What SPADE adds is the method, and it lives here.

- Paper: [arXiv:2608.19197](https://arxiv.org/abs/2608.19197)
- Reference code: [spade-rl/spade](https://github.com/spade-rl/spade)

## Layout

```text
beta/spade/
  generation.py   the experience section (results sorted by regret into the frontier, the mastered and the out of reach), the Designer's score and advantages, a generation's records
  generator.py    SpadeTaskGenerator: the Designer's generations through the generator service
  slime/          SpadeObjective: trains a task job's rollouts and the Designer replies that it carries, on Slime
  examples/tinker/serve.yaml   a Tinker deployment with the generator service and the generated-task recipe
  examples/slime/serve.yaml    a Slime deployment that also trains the Designer, with the paper's Qwen3-4B settings
```

SPADE is a task generator. The deployment selects it in `pipeline.record2sample.config.generator`. The built-in generated-task recipe (`reef.recipe.generated_tasks:GeneratedTaskRecipe`) runs the tasks and trains on them. Its stages and their settings are described in [generated tasks](../../../docs/reference/configuration.rst#generated-tasks).

## The generator

`SpadeTaskGenerator` implements `reef.recipe.generated_tasks.TaskGenerator`. One round of the recipe is one SPADE generation:

1. `generate` makes `count` Designer proposals across the configured `skills`. Each prompt includes the previous generation's results as SPADE's experience section.
2. `validate` writes the task under the service's tasks root, runs Harbor's oracle and nop agents, and removes the task again. The reference solution must score 1. Doing nothing must score below 1. A duplicate task is invalid. After a reload cancels a generation, a rerun replaces a task with the same name from the earlier attempt.
3. The recipe saves each valid task with its train/eval split. Later it hands the training tasks to the training component as task jobs. The component runs each task `rollout.rollouts_per_task` times on the plain arm and trains on the rollouts. The recipe runs no eval tasks.
4. When every training task of the generation has a result, `record_results` runs each train task that ran and that has `solution/hint.txt` `hint_plays` more times through the generator service, on the `hint` arm, with that file appended to the instruction. Hint rollouts measure only. It then reports the score of each train proposal against its Designer receipt (see [Designer training](#designer-training)). A refused proposal, or one whose task ran no rollouts, scores 0. Eval proposals and tasks that could not run get no Designer report and do not go into the experience. This can happen after later generations started.

Regret is the mean hint reward minus the mean plain reward. Without hint rollouts, regret is 0. The mean plain reward determines the task's band: mastered above 0.9, out of reach below 0.1, and frontier otherwise. The next prompt lists frontier tasks by regret, highest first.

The recipe saves each task and its train/eval split in the scenario's task store. Related source sessions keep the same split across generations.

The generator keeps its own state in the same task store, as state documents, and keeps no file of its own. So the state is in the deployment's database, and a restart on another machine reads it. Each number in a key has five digits:

| Document | Content |
| --- | --- |
| `open/<generation>` | the proposals of a generation that `record_results` has not closed, named by the recipe's generation number |
| `report/<number>` | each proposal, refusal, and measurement of a closed generation, named by SPADE's own number of the generation |

After a restart, the generator reads the last report to recover the experience, closes open generations when their results arrive, and numbers new generations after the reports and open generations.

The documents are rows of the table `task_state_document` in the scenario's database, under the store name `generated-tasks`. With SQLite storage, the database of a scenario is `<storage.agent-record-dir>/<SHA-256 of the scenario name>.sqlite3`. To read the report of SPADE's generation 0:

```bash
sqlite3 "$REEF_SPADE_STATE_DIR/agent-record/$(printf spade | shasum -a 256 | cut -d' ' -f1).sqlite3" \
  "select document from task_state_document where store_name = 'generated-tasks' and document_key = 'report/00000'"
```

The Designer's reports and the hint rollouts share the scenario with the training rollouts. The recipe trains only on the plain arm.

The generator's `config` keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `generator_url` | required | the generator service, `${endpoints.generator}` |
| `count` | 8 | Designer proposals in each generation |
| `skills` | none | skills the proposals cycle through |
| `difficulty` | `medium` | the difficulty the Designer aims for |
| `turn_limit` | 12 | the turn limit the Designer writes for |
| `grounding_path` | none | a text file added to the Designer prompt |
| `hint_plays` | 2 | plays of the hint arm; 0 turns it off |
| `designer_report` | true | report each proposal's score against the Designer's receipt |
| `served_model` | the deployment's | the model the Designer and the hint plays ask for |
| `plateau_low`, `plateau_high` | 0.4, 0.6 | the band of mean plain rewards where the plateau reward is 1 |
| `plateau_ramp` | 0.25 | the width over which the plateau reward falls from 1 to 0 on each side of the band |
| `plateau_weight` | 0.0 | the weight of the plateau reward in the score |
| `regret_weight` | 1.0 | the weight of the floored regret in the score |
| `regret_scale` | 1.0 | the regret that counts as 1 in the score |
| `max_designer_scale` | 50 | the largest scale of the Designer's advantages |
| `train_designer` | false | keep each Designer reply's sample, and give the recipe the advantages that train the replies |

The generator takes no `state_dir`: a config that still has the key is refused. The generator does not read the files of an earlier `state_dir`, so a scenario that upgrades starts with no experience, and an open generation of the earlier version gets no feedback. The task store keeps every generation and task.

Set `eval_fraction` and `split_seed` in `pipeline.record2sample.config`, outside the generator configuration. With the defaults, 0.0 and 0, all tasks train. The task store keeps each published task's split, not these settings. After a restart, published tasks keep their split and new tasks are placed with the settings in force. Reef does not refuse a changed `eval_fraction` or `split_seed`, so keep both fixed for the life of a scenario.

The generated-task recipe names no objective. The deployment provides one, as a dotted reference in `recipe.config.job_parameters.objective`; the example `serve.yaml` holds a placeholder that you must replace. The objective's loss can be Tinker's `importance_sampling`. The example uses the `tinker` backend, which runs the rollouts on its own sampler.

## Run it

```bash
export REEF_TOKEN=reef-local REEF_SPADE_STATE_DIR="$PWD/work/spade"   # and TINKER_API_KEY
reef serve -c recipes/beta/spade/examples/tinker/serve.yaml
curl -s -X POST -H "Authorization: Bearer $REEF_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "spade"}' http://127.0.0.1:8900/reef/scenarios
```

`reef serve` creates no scenario on its own. The `POST /reef/scenarios` above (or the first model call that names the scenario) creates it. Round 0 starts when the recipe's processor first looks for job data after that.

The deployment's `generator` section makes `reef serve` start the generator service before the HTTP service and hand its address to the recipe as `${endpoints.generator}`. The generator runs under Reef's interpreter; its host needs Docker and the `harbor` command line, which the Reef service itself does not. `execution: {generator: ray}` places it elsewhere. `generator.designer-url` and `generator.designer-model` point the Designer at another service (a strong model on OpenRouter while the Reasoning Agent is the deployment under training); by default both roles are the served model. `generator.designer-options` adds fields to the Designer's request: a model that thinks for thousands of tokens before writing an environment runs past the service's inference deadline, and `{"reasoning_effort": "none"}` keeps it to the reply. With `generator.designer-api: responses` (the default when `inference.upstream-api` is `responses`), the Designer asks through the Responses API, and its options take Responses fields such as `{"reasoning": {"effort": "low"}}`. See [the generator section](../../../docs/reference/configuration.rst) for every key.

With `rounds: 0` the recipe generates nothing; it trains only on the train tasks that earlier generations saved in the scenario's task store. A rollout is not a record: Reef keeps no record of its model calls. The training component assembles the rollout's sample from the calls that its sampler answered.

Thinking stays off because a thinking model's rollout never assembles into one sample: the agent's history carries earlier turns without their thinking, so the second turn's prompt no longer extends the first turn's tokens. With thinking off, Qwen3's generation prompt still ends with an empty think block that the history drops; `scaffold_tolerance` (in `recipe.config.job_parameters.rollout`) lets the assembly realign those masked tokens. A tasks root under a path Docker shares with the host (on macOS, under the home directory) is required, or the verifier's reward file never reaches the host.

Each task job runs its rollouts at the weights the job starts from, so a task group holds rollouts of one weight version. The Designer can train in the same jobs; see [Designer training](#designer-training).

## Designer training

The Designer trains on its own replies in the task jobs of a later generation. Two settings turn it on:

- `train_designer: true` in the generator's `config`. Each Designer call then asks the generator service to keep the reply's sampled tokens, their log-probs and the sampling distribution in its inference record.
- `designer_delay` of 1 or more in `pipeline.record2sample.config`. The replies of generation g then train in the first task job of round g + `designer_delay`. The tasks of that round wait until generation g has its advantages. When the generator gives them, the samples service saves each reply's sample in the task store, and control reads them from there, so the two can run in separate processes. The replies of the last `designer_delay` generations of a run have no such round: they stay in the task store, and a run that continues with more rounds trains them. See [generated tasks](../../../docs/reference/configuration.rst#generated-tasks) for the rules of the stage.

The Designer's score of a train task that ran follows the reference code:

```text
score = plateau_weight * plateau(win rate) + regret_weight * min(1, max(0, regret / regret_scale))
```

The win rate is the task's mean plain reward. `plateau` is 1 on [`plateau_low`, `plateau_high`] and falls linearly to 0 over `plateau_ramp` on each side. The defaults score the regret alone, clipped to [0, 1]. The generator reports this score against the Designer's receipt.

The advantage of a reply is `scale * (score - mean)`. `mean` is the mean score of the generation's replies of the same skill. `scale` is the number of training rollouts of the generation's measured tasks divided by the number of those tasks, at most `max_designer_scale`. Only the replies whose train task ran rollouts get an advantage: refused proposals, eval proposals and tasks that could not run get none. The generation report keeps the advantages, so the recipe reads the same values after a restart.

Each reply trains from the sample that its served call kept, so the Designer must be the served model under training (no `generator.designer-url`). The Designer's calls must name `temperature`, `top_p` and `top_k`, in the request or in the serving engine's `sampling_defaults`, and set no `min_p`, penalty or logit bias, so that the record names the distribution of the sample. The Slime example has not run end to end yet.

### On Slime

Task jobs of a SPADE run name the objective `spade` (`recipes.beta.spade.slime:SpadeObjective`). Slime rolls each task out. The plays of each task are centered by SPADE's rule: the reward minus the task's mean, over the population deviation (`np.std`) plus 1e-8. The job's delayed samples are the Designer replies of an earlier generation, with the advantages that the generator gave them. They train in the job's first optimizer step, before the plays, one reply to a row. Slime trims the last plays when the step is full, as SPADE does.

Slime options for this objective:

- `--advantage-estimator` grpo, gspo or cispo, and no `--normalize-advantages` or `--custom-advantage-function-path`.
- `--n-samples-per-prompt` of at least 2.
- `--use-tis`, with no `--custom-tis-function-path` and no `--use-rollout-logprobs`.
- `--use-dynamic-batch-size`, and no `--use-score-centering`.
- `--global-batch-size` of at most `(rollout-batch-size - 1) * n-samples-per-prompt + 1`, the fewest rollouts that a first rollout holds.
- `--rollout-batch-size` of at least 2, and at least the number of tasks plus `max(1, ceil(replies / n-samples-per-prompt))`, so that one rollout is one step. A job that carries no reply still leads with one group of padding rows.
- `--rollout-temperature`, `--rollout-top-p` and `--rollout-top-k` equal to the distribution of the Designer's calls.

The deployment refuses these options when it loads, and the training component refuses a job that breaks them. A job carries at most `min(global-batch-size, (rollout-batch-size - 1) * n-samples-per-prompt)` replies, each sampled from the rollout distribution. Control leaves the other replies out when it builds the job's data, with a warning in the log, so the job still trains its tasks.

For example, 24 tasks, 24 replies and n = 8 need a rollout batch size of 27. With a global batch size of 192, the step trains the 24 replies and the first 168 plays. A job that carries fewer replies, such as the first job of a run, which carries none, fills the rollout with its first tasks again, as every task job fills its last rollout. Those plays come last, so the step does not train them, but they count toward their tasks' win rates and the Designer's scale.

Checkpoint retention ranks a job's checkpoint by the mean reward of its plays. Slime's own raw reward metrics count the Designer rows as reward 0.

```bash
export REEF_TOKEN=reef-local REEF_SPADE_STATE_DIR="$PWD/work/spade"
reef serve -c recipes/beta/spade/examples/slime/serve.yaml
```

### Declared divergences

The Slime example follows the reference code's Qwen3-4B paper run in its score, its band, its weights and its optimizer settings. These parts differ:

- The win rate uses the 8 plays of one task job, not 32 plays over 4 steps. So the scale is 8, not 32.
- The Designer's replies train one task job after their generation, not 4 steps after it.
- The hint plays run after the task job, on the artifact that the job published.
- The hint is part of the Designer's reply, so it trains with the reply.
- Both roles sample at temperature 0.6 with top_p 1 and top_k -1, the distribution that the trainer reads. The reference code samples both roles at temperature 0.6, top_p 0.95 and top_k 20, and its trainer reads temperature 1.0.
- A play that fails in the infrastructure leaves its task's group. The reference code counts a failed play as -1.
- The score has no learning progress or frontier term. The paper runs set both weights to 0.
- The generator's defaults score the regret alone. The example sets the paper's weights and band.

## Generate from selected records

To reconstruct a past task, pass its `AgentRecord` objects and selected local materials to `generate`. The records must belong to the generator's scenario. The recipe's rounds send no records, so call the generator yourself. Run the generator service first. Then call this function from an async caller:

```python
from pathlib import Path

from recipes.beta.spade import SpadeTaskGenerator
from reef.core import AgentRecord
from reef.core.tasks import HarborTask
from reef.recipe.generated_tasks import GeneratorState
from reef.record2sample.generator import HttpGenerator, TaskGenerationRequest


async def rebuild_task(
    records: tuple[AgentRecord, ...], assets: tuple[Path, ...], generator_url: str, state: GeneratorState
) -> HarborTask:
    generator = SpadeTaskGenerator(
        scenario="spade", service=HttpGenerator(generator_url), state=state, count=1
    )
    (generated,) = await generator.generate(
        TaskGenerationRequest(records, "Reconstruct the original task before it was solved.", assets)
    )
    result = await generator.validate(generated.task)
    if not result.is_valid:
        raise ValueError("; ".join(result.errors))
    return generated.task
```

The example generates and validates one task. It does not write it for rollouts or train on it. `state` is the generator's state. In a deployment, the recipe builds it from the scenario's task store. A direct caller builds it the same way, from an open scenario session: `StoredGeneratorState(session.tasks.open("generated-tasks"))`, with `StoredGeneratorState` from `reef.recipe.generated_tasks`. With source records, the task keeps their original IDs in `source_agent_record_ids`. It stores the new Designer receipt in `metadata["designer_record_id"]`. If the Designer refuses, `generate` returns no task. To report the Designer's score, pass the task results to `record_results`.

History tasks use `harbor-history-<hash>` names derived from the scenario, source record IDs, and Designer receipt. Each new Designer call gets a separate name, even with the same source records. Name conflicts are refused without deleting the existing task. Identical task content is still refused as a duplicate.

Materials can be UTF-8 files or directories on the caller's machine. The client sends their contents over HTTP. The service does not need a shared filesystem. A selected directory keeps its relative paths under `asset-0/`, `asset-1/`, and so on. A selected file uses `asset-N/<filename>`.

Use a prepared snapshot containing only the materials the Designer should see. The client rejects symlinks, special files, non-text files, and unreadable files. Limits are 128 files, 256 KiB of file contents, and 512 KiB of JSON for records and files combined. Oversized inputs fail before a Designer call. The client and service do not truncate these inputs.

The input contract is `TaskGenerationRequest` in `reef.record2sample.generator`. Both SPADE and the generator service use this type. For materials already in memory, pass `asset_files={"state.txt": "file contents"}` instead of `assets`.

To call the generator directly, wrap the shared request in `DesignerRequest(inputs=request, difficulty="hard")`. Then call `HttpGenerator.propose(...)`. `DesignerRequest` holds the shared input and adds Designer settings. The HTTP client reads local `assets` into `asset_files`. The service reconstructs the same input type. Each source record retains its payload, type, timestamp, references, and artifact version. Existing `DesignerRequest(target="...")` calls still work for description-only generation. Use matching client and generator versions for historical inputs.

The recipe's rounds use the `description` of its record-to-sample stage and the previous generation's scores. With `source_records_per_round` above 0, each round also sends recorded sessions that carry the record stage's `source_tag`; see [generated tasks](../../../docs/reference/configuration.rst#generated-tasks). Otherwise, the caller selects historical sessions. The Designer makes one model call. It does not yet inspect files with tools or revise a task after a failed check. Passing the oracle check does not establish that the reconstruction is faithful to the original task.
