# spade

**Status: beta.** SPADE remains under `recipes/beta/spade/` until complete, reproducible learning results are published. The tests validate the Designer call, the checks on what it writes and one generation against stand ins; they do not establish learning performance. See [issue #482](https://github.com/Human-Agent-Society/reef/issues/482) and [issue #498](https://github.com/Human-Agent-Society/reef/issues/498) for the pieces, [issue #447](https://github.com/Human-Agent-Society/reef/issues/447) for the pipeline they belong to, and [issue #422](https://github.com/Human-Agent-Society/reef/issues/422) for this classification.

[SPADE](https://arxiv.org/abs/2608.19197), self play in adaptive synthetic executable environments, as a Reef recipe: one policy plays an Environment Designer that writes executable environments and a Reasoning Agent that learns in them. Reef knows one task format, Harbor, and the Designer writes it directly: an instruction, a container, a verifier and a reference solution, so every environment is a Harbor task any Harbor agent can play (in reef's gate today that is the `terminus` adapter). Everything SPADE needs beyond the format lives in this package.

- Paper: [arXiv:2608.19197](https://arxiv.org/abs/2608.19197)
- Reference code: [spade-rl/spade](https://github.com/spade-rl/spade)

## Layout

```text
beta/spade/
  designer.py     the Designer's adversarial prompt, its reply parsed, the experience the prompt carries
  harbor.py       the written task under the team's structural gate, its hash, a split per generation, Harbor's oracle check
  generation.py   one generation end to end: propose, write, check, play both arms, split, report
  recipe.py       the Reasoning Agent's training recipe: the task player's reports, grouped by task, on Tinker
  processor.py    the reports grouped by task; a batch is tasks_per_step complete groups
  preparer.py     group relative advantages per task group
  examples/tinker/serve.yaml   a Tinker deployment of the Reasoning Agent recipe
```

## The Designer

`designer_messages` builds one Designer call: the skill and difficulty, the rules of the task contract, and what the agent did on the last generation's environments as `PlayRecord` rows sorted by the hint based regret into the frontier (the hint turned losses into wins), the mastered and the out of reach, so the next environment lands where the agent fails today. The rules follow the team's own terminal task designer: two network phases (the build has network, the agent and the verifier do not), a Dockerfile the classic Docker parser accepts, an image that carries tmux for the agent, an instruction of at least 80 characters, a verifier that writes the reward file, a reference solution, and hidden state the agent must inspect before it can act; an environment that answers the agent step by step is a program in the image whose state the agent cannot read. `parse_harbor_reply` reads the reply's `json` block into the instruction, the three file mappings and the hint.

## The task

`harbor_task` writes the task with `reef.core.tasks`: the instruction, the container files, the verifier under `tests/`, the reference solution and the hint under `solution/` (Harbor never mounts `solution/` for the agent), and `task.toml` with generation, step, index and, when set, skill, difficulty, document, a category from the team's vocabulary and the tags the category and the skill make; the task is named `harbor-<generation>-<index>` with the skill appended when there is one; the verifier runs as root, so a Dockerfile may keep state from the agent's user. `reply_errors` holds a reply to the authoring contract first (an untouched scaffold, a short instruction, a Dockerfile the parser rejects or without tmux, a verifier that never writes the reward file, a reference solution without a command are refused); `content_hash` deduplicates tasks across generations; `oracle_check` runs the task through the `harbor` command line with the oracle and the nop agents and accepts it only when the reference solution scores 1 and doing nothing scores below 1. `split_generation` puts every task of one Designer call in one split and refuses two tasks with one name.

## One generation

```bash
python -m recipes.beta.spade.generation \
  --reef-url http://127.0.0.1:8900 --scenario spade --model qwen/qwen3-coder \
  --tasks-root tasks --description "shell tasks with hidden state under /var and /etc" \
  --count 12 --generation 1 \
  --experience tasks/.spade/generation-00000.json
```

`Generation.run(GenerationRequest)` is the same call from Python. The Designer and the Reasoning Agent are two objects with their own service and model: `--designer-reef-url`, `--designer-scenario`, `--designer-model` and `--designer-token` point the Designer elsewhere (a strong model on OpenRouter while the Reasoning Agent is the deployment under training); without them both share `--reef-url`, `--scenario` and `--model`. Their harnesses differ too: the Designer's is its prompt, the Reasoning Agent's is the Harbor agent `--agent-json` names. The description is the target; `--skills` adds an optional axis, comma separated names cycled over the proposals, named in the prompt and in the task names and used to sort the experience. The Designer is the served model through Reef, so every proposal is a record with a receipt; the reply is parsed, the task is written under the tasks root, a duplicate of a task already there is refused, and the oracle check runs. The Reasoning Agent plays each written task through `reef.harness.client.tasks`: `--plays` times as it is, reported to Reef as training data, and `--hint-plays` times with `solution/hint.txt` appended to the instruction, measured only. Regret is the mean hint reward minus the mean plain reward; the plain mean puts the task in its band. The tasks are split by the Designer's record ids into `manifest-<generation>.json` under the root, which `evolution.task_manifest` and the task player read; `.spade/generation-<generation>.json` keeps every proposal, refusal and measure, and `--experience` hands its records to the next generation's prompts. Each proposal is reported against the Designer's receipt with its regret as the score, the Designer's own training signal; `--no-designer-report` turns that off. `--agent-json` and `--agent-host` choose the Reasoning Agent's Harbor agent as the task player does; `--harbor` names the harbor command line for the oracle check. `--designer-timeout-s` bounds one Designer call (30 minutes by default; a large local model writes an environment in minutes, and the service's `inference.timeout_s` must allow it too). `--designer-json` adds fields to the Designer's chat request: a model that thinks for thousands of tokens before writing an environment runs past the service's inference deadline, and `{"reasoning_effort": "none"}` keeps it to the reply.

## Train the Reasoning Agent on the generated tasks

```bash
export REEF_TOKEN=reef-local REEF_SPADE_STATE_DIR="$PWD/work/spade-tinker"   # and TINKER_API_KEY
reef serve -c recipes/beta/spade/examples/tinker/serve.yaml
```

`SpadeRecipe` is a weight training recipe: the served model is the Reasoning Agent, its episodes on the generated tasks are its training data. Every plain episode the task player reports names its task under `metadata.task`; `SpadeProcessor` groups the reports by task, a group is complete at `rollouts-per-task` episodes, and a batch holds `tasks-per-step` complete groups. `SpadePreparer` centers and scales each episode's reward within its task group (a group with one reward everywhere gives 0), and Tinker's built in `importance_sampling` loss puts that advantage on every response token, so the recipe runs on the `tinker` backend today and fails at selection on Slime, which has no loss family of that name. With the deployment up, play the train split of a manifest `rollouts-per-task` times per task, with thinking off, and watch the scenario's step count:

```bash
AGENT='{"name": "terminus-2", "model_name": "openai/{model}", "kwargs": {"api_base": "{base_url}/v1", "llm_kwargs": {"api_key": "{api_key}",
  "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}}}'
for rollout in 1 2 3 4; do
  python -m reef.harness.client.tasks --reef-url http://127.0.0.1:8900 --scenario spade --model Qwen/Qwen3-8B \
    --manifest tasks/manifest-00000.json --tasks-root tasks --side train --work-dir work/play-$rollout --agent-json "$AGENT"
done
curl -s -H "Authorization: Bearer $REEF_TOKEN" http://127.0.0.1:8900/reef/status | python -m json.tool | grep -A 3 '"spade"'
```

Thinking stays off because a thinking model's episode never assembles into one sample: the agent's history carries earlier turns without their thinking, so the second turn's prompt no longer extends the first turn's tokens. With thinking off, Qwen3's generation prompt still ends with an empty think block that the history drops; `scaffold-tolerance` lets the assembly realign those masked tokens. A work directory under a path Docker shares with the host (on macOS, under the home directory) is required, or the verifier's reward file never reaches the host.

A generation whose Reasoning Agent is this deployment (`--reef-url` the same service, `--plays` at least `rollouts-per-task`) trains as it plays: the plain arm's reports are the batch. The Designer's own training, its regret as the reward of its proposals, follows.
