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

As a reported feedback processor it takes the task player's reports: every plain episode names its task under `metadata.task`, the episodes of one task form a group, a group is complete at `rollouts-per-task` episodes, and a batch holds `tasks-per-step` complete groups. `SpadeObjective` centers and scales each episode's reward within its task group (a group with one reward everywhere gives 0), and Tinker's built in `importance_sampling` loss puts that advantage on every response token, so the recipe runs on the `tinker` backend today and fails at selection on Slime, which has no loss family of that name. The Designer's own reports (`metadata.role: designer`) and the hint arm's episodes (`arm: hint`) share the scenario; the processor releases them unassembled and trains on the plain arm alone.

As a task generation processor (`reef.train.processors.TaskGenerationProcessor`) it runs the Designer. One generation is one job on a private worker, off the trainer's thread: `count` proposals over the `skills`, each a call to `generate` (the Designer asked through the generator service, with the last generation's results in the prompt as SPADE's experience section), written under the generator's tasks root (a duplicate refused; a name that an earlier attempt of the same generation took before a reload cancelled it is replaced), checked by `validate` (Harbor's oracle and nop agents: the reference solution scores 1, doing nothing below 1), played `rollouts-per-task` times as it is (the training data) and `hint-plays` times with `solution/hint.txt` appended (measured only), and reported against the Designer's receipt with its regret as the score, 0 for a refused one. Regret is the mean hint reward minus the mean plain reward; the plain mean puts the task in its band (mastered above 0.9, out of reach below 0.1, else frontier), and the frontier, highest regret first, is what the next prompt shows. The tasks are split by the Designer's record ids into `manifest-<generation>.json` under the tasks root, and `state-dir/generation-<generation>.json` keeps every proposal, refusal and measure; a restart reads it to carry on with the next generation and the last experience.

The first generation starts when the processor first looks for a batch; the next once `batches-per-generation` batches were acknowledged since the previous one started (its episodes train while it runs), so the Designer writes for the policy that trains now; a generation that measured no task is followed at once; `generations` caps them. `GET /reef/status` shows the generation in flight, the count completed and the last error.

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
