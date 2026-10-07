# spade

**Status: beta.** SPADE remains under `recipes/beta/spade/` until complete, reproducible learning results are published. The tests validate the experience section, the processor's grouping and its generations against stand ins; they do not establish learning performance. See [issue #482](https://github.com/Human-Agent-Society/reef/issues/482) and [issue #498](https://github.com/Human-Agent-Society/reef/issues/498) for the pieces, [issue #447](https://github.com/Human-Agent-Society/reef/issues/447) for the pipeline they belong to, and [issue #422](https://github.com/Human-Agent-Society/reef/issues/422) for this classification.

[SPADE](https://arxiv.org/abs/2608.19197), self play in adaptive synthetic executable environments, as a Reef recipe: one policy plays an Environment Designer that writes executable environments and a Reasoning Agent that learns in them. Reef knows one task format, Harbor, and writes, checks and plays Harbor tasks in `reef.record2dataset`: the task contract as a prompt, the served model asked through Reef (every proposal a record with a receipt), the reply held to the authoring rules and written with `reef.core.tasks`, Harbor's oracle and nop agents on the result, and the task player for the episodes. That runs as the generator service `reef serve` starts beside the HTTP service. What SPADE adds is the method, and it lives here.

- Paper: [arXiv:2608.19197](https://arxiv.org/abs/2608.19197)
- Reference code: [spade-rl/spade](https://github.com/spade-rl/spade)

## Layout

```text
beta/spade/
  generation.py   the experience section (results sorted by regret into the frontier, the mastered and the out of reach), a generation's records
  processor.py    the reported half: episodes grouped by task; the task generation half: the Designer's generations, run on a worker
  objective.py    group relative advantages per task group, on Tinker's importance sampling loss
  recipe.py       the configuration that binds them
  examples/tinker/serve.yaml   a Tinker deployment with the generator service
```

## The processor

`SpadeProcessor` implements two of Reef's processor contracts at once.

As a reported feedback processor, it groups the task player's reports by task. Each plain episode identifies its task in `metadata.task`. A group is complete when it contains `rollouts-per-task` episodes. A batch contains `tasks-per-step` complete groups.

`SpadeObjective` centers and scales each episode's reward within its task group. If all rewards in a group are equal, each advantage is 0. Tinker's `importance_sampling` loss applies that advantage to every response token. The recipe requires the `tinker` backend. Selecting Slime fails because it has no loss family with that name.

Designer reports (`metadata.role: designer`) and hint episodes (`arm: hint`) share the scenario. The processor releases these records without assembling them into training samples. It trains only on plain episodes.

As a task generation processor (`reef.train.processors.TaskGenerationProcessor`), it runs the Designer through the generator service. Each generation runs on a private worker, outside the trainer's thread. It makes `count` proposals across the configured `skills`:

1. The Designer proposes a task. Its prompt includes the previous generation's results as SPADE's experience section.
2. The service writes the task under its tasks root. Duplicate tasks are refused. After a reload cancels a generation, a retry replaces a task with the same name from the earlier attempt.
3. `validate` runs Harbor's oracle and nop agents. The reference solution must score 1. Doing nothing must score below 1.
4. The task player runs `rollouts-per-task` plain episodes for training. If the task has `solution/hint.txt`, it also runs `hint-plays` episodes with that hint appended. Hint episodes measure performance only.
5. The processor reports the task's regret against the Designer's receipt. Refused proposals receive a score of 0.

Regret is the mean hint reward minus the mean plain reward. Without hint episodes, regret is 0. The mean plain reward determines the task's band: mastered above 0.9, out of reach below 0.1, and frontier otherwise. The next prompt lists frontier tasks by regret, highest first.

The service splits tasks by source record ID into `manifest-<generation>.json` under the tasks root. The processor saves each proposal, refusal, and measurement in `state-dir/generation-<generation>.json`. After a restart, it reads these reports to recover the last experience and continue with the next generation.

The first generation starts when the processor first checks for a batch. Training can consume its episodes while generation continues. The next generation starts after `batches-per-generation` batches have been acknowledged since the previous generation started. If a generation measures no tasks, the next starts immediately. The `generations` setting limits the total number of generations. `GET /reef/status` shows the active generation, the number completed, and the last error.

## Run it

```bash
export REEF_TOKEN=reef-local REEF_SPADE_STATE_DIR="$PWD/work/spade"   # and TINKER_API_KEY
reef serve -c recipes/beta/spade/examples/tinker/serve.yaml
curl -s -X POST -H "Authorization: Bearer $REEF_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "spade"}' http://127.0.0.1:8900/reef/scenarios
```

The scenario is what the processor belongs to, and `reef serve` creates none on its own: the `POST /reef/scenarios` above (or the first model call that names the scenario) brings it into being, and generation 0 starts on the processor's first look for a batch after that.

The deployment's `generator` section makes `reef serve` start the generator service before the HTTP service and hand its address to the recipe as `${endpoints.generator}`. The generator runs under Reef's interpreter; its host needs Docker and the `harbor` command line, which the Reef service itself does not. `execution: {generator: ray}` places it elsewhere. `generator.designer-url` and `generator.designer-model` point the Designer at another service (a strong model on OpenRouter while the Reasoning Agent is the deployment under training); by default both roles are the served model. `generator.designer-options` adds fields to the Designer's chat request: a model that thinks for thousands of tokens before writing an environment runs past the service's inference deadline, and `{"reasoning_effort": "none"}` keeps it to the reply. See [the generator section](../../../docs/reference/configuration.rst) for every key.

With `generations: 0` the processor generates nothing and trains on whatever the task player reports, which is how to train on tasks written elsewhere:

```bash
python -m reef.harness.client.tasks --reef-url http://127.0.0.1:8900 --scenario spade --model Qwen/Qwen3-8B \
  --manifest tasks/manifest-00000.json --tasks-root tasks --side train --work-dir work/play --label arm=plain
```

Thinking stays off because a thinking model's episode never assembles into one sample: the agent's history carries earlier turns without their thinking, so the second turn's prompt no longer extends the first turn's tokens. With thinking off, Qwen3's generation prompt still ends with an empty think block that the history drops; `scaffold-tolerance` lets the assembly realign those masked tokens. A tasks root under a path Docker shares with the host (on macOS, under the home directory) is required, or the verifier's reward file never reaches the host.

Known limits: a generation runs for hours while the weights reload every step, so one task group can hold episodes of two weight versions; `max-staleness` bounds that. The Designer's own training, its regret as the reward of its proposals, follows.

## Generate from selected records

To reconstruct a past task, pass its `AgentRecord` objects and selected local materials to `generate`. The records must belong to the processor's scenario. Run the generator service first. Then call this function from an async caller outside the trainer lock:

```python
from pathlib import Path

from recipes.beta.spade import SpadeProcessor
from reef.core import AgentRecord
from reef.core.tasks import HarborTask, TaskGenerationRequest
from reef.record2dataset import HttpGenerator
from reef.train.types import ProcessorContext


async def rebuild_task(
    records: tuple[AgentRecord, ...], assets: tuple[Path, ...], generator_url: str
) -> HarborTask:
    generator = HttpGenerator(generator_url)
    processor = SpadeProcessor(ProcessorContext("spade", {}), generator=generator)
    task = await processor.generate(
        TaskGenerationRequest(records, "Reconstruct the original task before it was solved.", assets)
    )
    written = await generator.write_task(task)
    result = await processor.validate(written.path)
    if not result.is_valid:
        raise ValueError("; ".join(result.errors))
    return task
```

The example generates a task, writes it, and validates it. With source records, the returned task keeps their original IDs in `source_agent_record_ids`. It stores the new Designer receipt in `metadata["designer_record_id"]`. Calling `generate` alone does not write, check, play, or train the task.

To run SPADE's write, check, play, and report sequence, call `proposed(generation, index, skill, request=...)`. This method sends feedback against the new Designer receipt.

Materials can be UTF-8 files or directories on the caller's machine. The client sends their contents over HTTP. The service does not need a shared filesystem. A selected directory keeps its relative paths under `asset-0/`, `asset-1/`, and so on. A selected file uses `asset-N/<filename>`.

Use a prepared snapshot containing only the materials the Designer should see. The client rejects symlinks, special files, non-text files, and unreadable files. Limits are 128 files, 256 KiB of file contents, and 512 KiB of JSON for records and files combined. Oversized inputs fail before a Designer call. The client and service do not truncate these inputs.

The input contract is `TaskGenerationRequest` in `reef.core.tasks`. Both SPADE and record2dataset use this type. For materials already in memory, pass `asset_files={"state.txt": "file contents"}` instead of `assets`. The old import from `reef.train.processors.task_generation` remains supported.

To call the generator directly, wrap the shared request in `DesignerRequest(inputs=request, difficulty="hard")`. Then call `HttpGenerator.propose(...)`. `DesignerRequest` holds the shared input and adds Designer settings. The HTTP client reads local `assets` into `asset_files`. The service reconstructs the same input type. Each source record retains its payload, type, timestamp, references, and artifact version. Existing `DesignerRequest(target="...")` calls still work for description-only generation. Use matching client and generator versions for historical inputs.

The automatic generation loop still uses the configured description and previous task scores. The caller selects historical sessions. The Designer makes one model call. It does not yet inspect files with tools or revise a task after a failed check. Passing the oracle check does not establish that the reconstruction is faithful to the original task.
