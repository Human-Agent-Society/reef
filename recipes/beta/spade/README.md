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
  designer_processor.py   the Designer's reports grouped by generation; skills compare through the group id
  objective.py    group relative advantages per group, on Tinker's importance sampling loss; both roles use it
  recipe.py       the two recipes: the Reasoning Agent's on its episodes, the Designer's on its regret
  examples/tinker/serve.yaml            a Tinker deployment with the generator service
  examples/designer-tinker/serve.yaml   a Tinker deployment of the Designer recipe, on its own port
  harness.py      the Designer's prompt as a harness tree, rewritten from the regret reports
  examples/designer-harness/serve.yaml   the Designer's harness evolution service, on its own port
```

## The processor

`SpadeProcessor` implements two of Reef's processor contracts at once.

As a reported feedback processor it takes the task player's reports: every plain episode names its task under `metadata.task`, the episodes of one task form a group, a group is complete at `rollouts-per-task` episodes, and a batch holds `tasks-per-step` complete groups. `SpadeObjective` centers and scales each episode's reward within its task group (a group with one reward everywhere gives 0), and Tinker's built in `importance_sampling` loss puts that advantage on every response token, so the recipe runs on the `tinker` backend today and fails at selection on Slime, which has no loss family of that name. The Designer's own reports (`metadata.role: designer`) and the hint arm's episodes (`arm: hint`) share the scenario; the processor releases them unassembled and trains on the plain arm alone.

As a task generation processor (`reef.train.processors.TaskGenerationProcessor`) it runs the Designer. One generation is one job on a private worker, off the trainer's thread: `count` proposals over the `skills`, each a call to `generate` (the Designer asked through the generator service, with the last generation's results in the prompt as SPADE's experience section), written under the generator's tasks root (a duplicate refused; a name that an earlier attempt of the same generation took before a reload cancelled it is replaced), checked by `validate` (Harbor's oracle and nop agents: the reference solution scores 1, doing nothing below 1), played `rollouts-per-task` times as it is (the training data) and `hint-plays` times with `solution/hint.txt` appended (measured only), and reported against the Designer's receipt with its regret as the score, as measured and negative when the hint hurt, or `REFUSAL_SCORE` (-1.0) for a refused one, below any measured task; the report's metadata names the generation and its size (`proposals`), and its feedback carries the task's measure and, under `round.previous`, the last generation's summary (its mean regret, tasks measured and refused). Regret is the mean hint reward minus the mean plain reward; the plain mean puts the task in its band (mastered above 0.9, out of reach below 0.1, else frontier), and the frontier, highest regret first, is what the next prompt shows. The tasks are split by the Designer's record ids into `manifest-<generation>.json` under the tasks root, and `state-dir/generation-<generation>.json` keeps every proposal, refusal and measure; a restart reads it to carry on with the next generation and the last experience.

The first generation starts when the processor first looks for a batch; the next once `batches-per-generation` batches were acknowledged since the previous one started (its episodes train while it runs), so the Designer writes for the policy that trains now; a generation that measured no task is followed at once; `generations` caps them. `GET /reef/status` shows the generation in flight, the count completed and the last error.

With `report-plays-after-generation` every play is held until the generation lands, and the generation then reports them all at once: the plain plays stamped with the generation as their round and with the round's size, the hint plays beside them. Nothing trains while the generation runs, so the Designer is measured against one Reasoning Agent version. The processor keeps a round as one unit and trains the generation as one batch, ordered by task, whatever `tasks-per-step` says; the batch counter counts one batch per generation, so `batches-per-generation: 1` puts one training step between generations. Off by default: each play reports as it ends and trains in the next batch of complete groups. The generation's report says how many held plays went out.

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

Known limits: a generation runs for hours while the weights reload every step, so one task group can hold episodes of two weight versions; `max-staleness` bounds that.

## Train the Designer on its regret

```bash
export REEF_TOKEN=reef-local REEF_SPADE_DESIGNER_STATE_DIR="$PWD/work/spade-designer"   # and TINKER_API_KEY
reef serve -c recipes/beta/spade/examples/designer-tinker/serve.yaml
```

Two Reef services run. The agent stack (`examples/tinker/serve.yaml`, port 8900) runs the generator and trains the Reasoning Agent; the Designer's service (`examples/designer-tinker/serve.yaml`, port 8901) serves the Designer and trains it. The agent stack's `generator` section points the Designer's calls at the second service:

```yaml
generator:
  designer-url: http://127.0.0.1:8901
  designer-token: ${REEF_TOKEN}
  designer-scenario: designer
  designer-model: Qwen/Qwen3-8B
```

`designer-scenario` is the scenario the Designer's records and reports go to on that service, whatever scenario the agent stack trains under; the Designer's deployment sets `allow-implicit-scenario-creation: true`, so the generator's first call creates it. `designer-model` names the model that service serves. Every proposal is then an inference record on the Designer's service, and `SpadeProcessor` reports each one there with its regret as the score, the generation and its size (`proposals`) in the metadata.

`SpadeDesignerRecipe` is the Designer's weight training recipe: each proposal is one chat call, so one sample, with the regret its task earned as the score. `SpadeDesignerProcessor` batches one generation as one unit (`metadata.proposals` reports arrived, refusals included at `REFUSAL_SCORE`), and a sample's group id names the generation and, when the report names one, the skill, so `SpadeObjective` centers the scores within a generation and a skill; a generation whose proposals all scored alike is skipped rather than trained on zero advantages. `generations-per-step` sets how many complete generations one step trains on. `GET /reef/status` on the Designer's service shows the groups still buffered and how many reports each holds. This trains a separate Designer LoRA: two Tinker deployments never share a parameter, so the Designer and the Reasoning Agent are two policies, not the paper's shared weight self play; one scenario training both roles is a later issue.
## Evolve the Designer's prompt

The Designer's prompt is two texts, the system turn and the rules block (`reef.record2dataset.designer.DesignerPrompt`), and `SpadeDesignerHarnessRecipe` evolves them as a harness tree of two skill entries, `designer-system` and `designer-rules`. Run it as a second deployment beside the training one:

```bash
export REEF_TOKEN=reef-local REEF_SPADE_DESIGNER_STATE_DIR="$PWD/work/designer"
export REEF_UPSTREAM_URL=https://openrouter.ai/api REEF_UPSTREAM_MODEL=openai/gpt-5 REEF_UPSTREAM_API_KEY=...
reef serve -c recipes/beta/spade/examples/designer-harness/serve.yaml
```

The training deployment's `generator` section points the Designer at it: `designer-url: http://127.0.0.1:8901`, `designer-token`, `designer-scenario` naming the scenario the Designer's calls create there, and `designer-prompt: harness`. Every Designer call is then an inference record on that service and every proposal's report lands there with its regret as the score. `batch-size` on the harness service equals the training recipe's `count`, so one evolve step reads one whole generation: `propose_prompt` shows the Designer model each proposal's regret, feedback and instruction excerpt (fenced as data) beside the current texts and asks for a rewrite of one or both as a JSON array of skill entries; only the texts that changed become `update` mutations. `ReportedRegretSelection` publishes every rewrite without a gate episode, and the generator pulls the release (`GET /reef/harness`, `native/tree.json`) once per generation before it asks the Designer, so the next generation is written with the new texts; a scenario that serves no tree yet leaves the fixed prompt in place. A generation's first proposal waits, polling every `designer-poll-s` seconds for up to `designer-wait-s`, until the Designer's service serves a release (or, for a weight training Designer, a runtime load id and step) other than the one it served when the previous generation's reports went out, so the rewrite those reports produced is the one the next generation pulls and a release that appeared before them, such as the scenario's creation release, does not count; on timeout it proceeds with a warning.

No gate holds a worse rewrite back: the regret the next generation earns is the rewrite's measure, and the trend of the mean regret across generations (`state-dir/generation-<generation>.json` on the training side, the step records under `evolution.step-record-dir` on this one) is what to watch. A rewrite that drops the `{turn_limit}` placeholder or a rule of the task contract shows up as refusals in the next generation, which the following rewrite sees.

Known limits: a generation runs for hours while the weights reload every step, so one task group can hold episodes of two weight versions; `max-staleness` bounds that. The Designer's own weight training, its regret as the reward of its proposals, follows.
