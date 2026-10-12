# spade

**Status: beta.** SPADE remains under `recipes/beta/spade/` until complete, reproducible learning results are published. The tests validate the experience section and the generator's generations against stand ins; they do not establish learning performance. See [issue #482](https://github.com/Human-Agent-Society/reef/issues/482) and [issue #498](https://github.com/Human-Agent-Society/reef/issues/498) for the pieces, [issue #447](https://github.com/Human-Agent-Society/reef/issues/447) for the pipeline they belong to, and [issue #422](https://github.com/Human-Agent-Society/reef/issues/422) for this classification.

[SPADE](https://arxiv.org/abs/2608.19197), self play in adaptive synthetic executable environments, as a Reef task generator: one policy plays an Environment Designer that writes executable environments and a Reasoning Agent that learns in them. Reef knows one task format, Harbor, and writes, checks and plays Harbor tasks in `reef.record2dataset`: the task contract as a prompt, the served model asked through Reef (every proposal a record with a receipt), the reply held to the authoring rules and written with `reef.core.tasks`, Harbor's oracle and nop agents on the result, and the task player for the episodes. That runs as the generator service `reef serve` starts beside the HTTP service. What SPADE adds is the method, and it lives here.

- Paper: [arXiv:2608.19197](https://arxiv.org/abs/2608.19197)
- Reference code: [spade-rl/spade](https://github.com/spade-rl/spade)

## Layout

```text
beta/spade/
  generation.py   the experience section (results sorted by regret into the frontier, the mastered and the out of reach), a generation's records
  generator.py    SpadeTaskGenerator: the Designer's generations through the generator service
  examples/tinker/serve.yaml   a Tinker deployment with the generator service and the generated-task recipe
```

SPADE is a task generator. The deployment selects it in the `task-generation` section. The built-in generated-task recipe (`reef.record2dataset.generated_tasks:GeneratedTaskRecipe`) plays the tasks and trains on them. The recipe and its settings are described in [the task-generation section](../../../docs/reference/configuration.rst).

## The generator

`SpadeTaskGenerator` implements `reef.core.tasks.TaskGenerator`. One round of the recipe is one SPADE generation:

1. `generate` makes `count` Designer proposals across the configured `skills`. Each prompt includes the previous generation's results as SPADE's experience section. A task that has `solution/hint.txt` asks for a `hint` arm of `hint_plays` plays with that file appended to the instruction.
2. `validate` writes the task under the service's tasks root, runs Harbor's oracle and nop agents, and removes the task again. The reference solution must score 1. Doing nothing must score below 1. A duplicate task is invalid. After a reload cancels a generation, a rerun replaces a task with the same name from the earlier attempt.
3. The recipe writes each valid task again and plays it: `plays-per-task` plain episodes for training, then the hint arm. Hint episodes measure only.
4. `record_results` reports each proposal's regret against the Designer's receipt. A refused or unplayed proposal scores 0.

Regret is the mean hint reward minus the mean plain reward. Without hint episodes, regret is 0. The mean plain reward determines the task's band: mastered above 0.9, out of reach below 0.1, and frontier otherwise. The next prompt lists frontier tasks by regret, highest first.

When the generation finished without a service error, the service splits the played tasks by source record ID into `manifest-<generation>.json` under the tasks root. The generator saves each proposal, refusal, and measurement in `state_dir/<scenario>/generation-<generation>.json`. After a restart, it reads the last report to recover the experience and continue with the next generation.

The Designer's reports and the hint episodes share the scenario with the training episodes. The recipe trains only on the plain arm.

The generator's `config` keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `generator_url` | required | the generator service, `${endpoints.generator}` |
| `state_dir` | required | where each scenario keeps its generation reports |
| `count` | 8 | Designer proposals in each generation |
| `skills` | none | skills the proposals cycle through |
| `difficulty` | `medium` | the difficulty the Designer aims for |
| `turn_limit` | 12 | the turn limit the Designer writes for |
| `grounding_path` | none | a text file added to the Designer prompt |
| `hint_plays` | 2 | plays of the hint arm; 0 turns it off |
| `eval_fraction`, `seed` | 0.25, 0 | the manifest's evaluation split |
| `designer_report` | true | report each proposal's regret against the Designer's receipt |
| `served_model` | the deployment's | the model the Designer asks for |

The generated-task recipe uses the `task_group` objective. It centers and scales each episode's reward within its task group. If all rewards in a group are equal, each advantage is 0. Tinker's `importance_sampling` loss applies that advantage to every response token. The recipe requires the `tinker` backend. Selecting Slime fails because it has no loss family with that name.

## Run it

```bash
export REEF_TOKEN=reef-local REEF_SPADE_STATE_DIR="$PWD/work/spade"   # and TINKER_API_KEY
reef serve -c recipes/beta/spade/examples/tinker/serve.yaml
curl -s -X POST -H "Authorization: Bearer $REEF_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "spade"}' http://127.0.0.1:8900/reef/scenarios
```

`reef serve` creates no scenario on its own. The `POST /reef/scenarios` above (or the first model call that names the scenario) creates it. Round 0 starts when the recipe's processor first looks for a batch after that.

The deployment's `generator` section makes `reef serve` start the generator service before the HTTP service and hand its address to the recipe as `${endpoints.generator}`. The generator runs under Reef's interpreter; its host needs Docker and the `harbor` command line, which the Reef service itself does not. `execution: {generator: ray}` places it elsewhere. `generator.designer-url` and `generator.designer-model` point the Designer at another service (a strong model on OpenRouter while the Reasoning Agent is the deployment under training); by default both roles are the served model. `generator.designer-options` adds fields to the Designer's chat request: a model that thinks for thousands of tokens before writing an environment runs past the service's inference deadline, and `{"reasoning_effort": "none"}` keeps it to the reply. See [the generator section](../../../docs/reference/configuration.rst) for every key.

With `rounds: 0` the recipe generates nothing and trains on the plain episodes that the task player reports. Use this to train on tasks written elsewhere:

```bash
python -m reef.harness.client.tasks --reef-url http://127.0.0.1:8900 --scenario spade --model Qwen/Qwen3-8B \
  --manifest tasks/manifest-00000.json --tasks-root tasks --side train --work-dir work/play --label arm=plain
```

Thinking stays off because a thinking model's episode never assembles into one sample: the agent's history carries earlier turns without their thinking, so the second turn's prompt no longer extends the first turn's tokens. With thinking off, Qwen3's generation prompt still ends with an empty think block that the history drops; `scaffold-tolerance` lets the assembly realign those masked tokens. A tasks root under a path Docker shares with the host (on macOS, under the home directory) is required, or the verifier's reward file never reaches the host.

Known limits: a round runs for hours while the weights reload every step, so one task group can hold episodes of two weight versions; `max-staleness` bounds that. The Designer's own training, its regret as the reward of its proposals, follows.

## Generate from selected records

To reconstruct a past task, pass its `AgentRecord` objects and selected local materials to `generate`. The records must belong to the generator's scenario. The recipe's rounds send no records, so call the generator yourself. Run the generator service first. Then call this function from an async caller:

```python
from pathlib import Path

from recipes.beta.spade import SpadeTaskGenerator
from reef.core import AgentRecord
from reef.core.tasks import HarborTask, TaskGenerationRequest
from reef.record2dataset import HttpGenerator


async def rebuild_task(
    records: tuple[AgentRecord, ...], assets: tuple[Path, ...], generator_url: str, state_dir: Path
) -> HarborTask:
    generator = SpadeTaskGenerator(
        scenario="spade", service=HttpGenerator(generator_url), state_dir=state_dir, count=1
    )
    (generated,) = await generator.generate(
        TaskGenerationRequest(records, "Reconstruct the original task before it was solved.", assets)
    )
    result = await generator.validate(generated.task)
    if not result.is_valid:
        raise ValueError("; ".join(result.errors))
    return generated.task
```

The example generates and validates one task. It does not write it for play or train on it. With source records, the task keeps their original IDs in `source_agent_record_ids`. It stores the new Designer receipt in `metadata["designer_record_id"]`. If the Designer refuses, `generate` returns no task. To report the Designer's score, pass the play results to `record_results`.

History tasks use `harbor-history-<hash>` names derived from the scenario, source record IDs, and Designer receipt. Each new Designer call gets a separate name, even with the same source records. Name conflicts are refused without deleting the existing task. Identical task content is still refused as a duplicate.

Materials can be UTF-8 files or directories on the caller's machine. The client sends their contents over HTTP. The service does not need a shared filesystem. A selected directory keeps its relative paths under `asset-0/`, `asset-1/`, and so on. A selected file uses `asset-N/<filename>`.

Use a prepared snapshot containing only the materials the Designer should see. The client rejects symlinks, special files, non-text files, and unreadable files. Limits are 128 files, 256 KiB of file contents, and 512 KiB of JSON for records and files combined. Oversized inputs fail before a Designer call. The client and service do not truncate these inputs.

The input contract is `TaskGenerationRequest` in `reef.core.tasks`. Both SPADE and record2dataset use this type. For materials already in memory, pass `asset_files={"state.txt": "file contents"}` instead of `assets`.

To call the generator directly, wrap the shared request in `DesignerRequest(inputs=request, difficulty="hard")`. Then call `HttpGenerator.propose(...)`. `DesignerRequest` holds the shared input and adds Designer settings. The HTTP client reads local `assets` into `asset_files`. The service reconstructs the same input type. Each source record retains its payload, type, timestamp, references, and artifact version. Existing `DesignerRequest(target="...")` calls still work for description-only generation. Use matching client and generator versions for historical inputs.

The recipe's rounds use the recipe's `description` and the previous generation's scores. The caller selects historical sessions. The Designer makes one model call. It does not yet inspect files with tools or revise a task after a failed check. Passing the oracle check does not establish that the reconstruction is faithful to the original task.
