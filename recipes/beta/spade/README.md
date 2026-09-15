# spade

**Status: beta.** SPADE remains under `recipes/beta/spade/` until complete, reproducible learning results are published. The tests validate the task form, the Designer call and the checks on what it writes; they do not establish learning performance. See [issue #479](https://github.com/Human-Agent-Society/reef/issues/479) and [issue #482](https://github.com/Human-Agent-Society/reef/issues/482) for the pieces, [issue #447](https://github.com/Human-Agent-Society/reef/issues/447) for the pipeline they belong to, and [issue #422](https://github.com/Human-Agent-Society/reef/issues/422) for this classification.

[SPADE](https://arxiv.org/abs/2608.19197), self play in adaptive synthetic executable environments, as a Reef recipe: one policy plays an Environment Designer that writes executable environments and a Reasoning Agent that learns in them. Reef knows one task format, Harbor, so every environment SPADE writes is a Harbor task any Harbor agent can play (in reef's gate today that is the `terminus` adapter); everything SPADE needs beyond the format lives in this package. The package is being built in pieces.

- Paper: [arXiv:2608.19197](https://arxiv.org/abs/2608.19197)
- Reference code: [spade-rl/spade](https://github.com/spade-rl/spade)

## Layout

```text
beta/spade/
  environment_loader.py   load and play a Gym style class as the reference does; ships into every task
  tasks.py                a generated environment as a Harbor task any agent can play; a split per generation
  process.py              the class in a child interpreter on the host, for the smoke test
  designer.py             the Designer's adversarial prompt for two kinds, its reply parsed, the smoke test
  harbor.py               the harbor kind: a Harbor task written directly, and Harbor's oracle check
  openenv.py              the openenv kind: an OpenEnv package served inside the container, and its check
  generation.py           one generation end to end: propose, check, write, play both arms, split, report
```

## The task form

`environment_task` turns one generated environment (its class, skill, generation, index, hint and the Designer's generation record id) into a Harbor task from `reef.core.tasks`. The agent never holds the class: the container runs the agent as a non root user, the class and its command live under `/opt/env` readable by root only, and a sudo rule lets the agent reach that one command through `observe` and `act`, which replay the root held action log from the seed, take one action when asked and print the next observation. `tests/replay.py` replays the same log through the same loader and writes the episode return (the last step's reward, clipped to [-1, 1], when the episode terminated, else 0). `solution/hint.txt` holds the privileged hint, and `task.toml` carries kind, skill, generation, step, index, difficulty, document, class name, turn limit and seed. One task is one seeded instance of the class: every play starts from the same hidden state, so a generation that wants the reference's spread over instances writes one task per seed. `split_generation` puts every environment of one Designer call in one split and refuses two tasks with one name.

## The Designer

`designer_messages` builds one Designer call for one kind, the three interfaces the community writes environments in: `harbor` asks for a Harbor task written directly (the instruction, the container files, the verifier, a reference solution); `gym` for a Python class with the Gym interface that the task wraps behind `observe` and `act`; `openenv` for an OpenEnv environment package (`models.py` with one Action and one Observation, an Environment subclass with reset, step and state) that the task serves inside the container behind a root only `serve` command, the agent acting with curl against `/reset` and `/step`, the server logging every step under root for the verifier. Both carry the skill and difficulty, the rules of the kind's contract, and what the agent did on the last generation's environments as `PlayRecord` rows sorted by the hint based regret into the frontier (the hint turned losses into wins), the mastered and the out of reach, so the next environment lands where the agent fails today. `parse_gym_reply` and `parse_harbor_reply` read the replies; `smoke_test` runs a `gym` class in three child interpreters on the host and refuses one that is not deterministic or breaks the contract; the children run with the caller's user, files and network, so the reply is first refused for any import the rules forbid (files, processes, the network, the interpreter itself), and that check is the trust boundary on the host; a `harbor` reply is first held to the team's own authoring contract (an instruction of at least 80 characters, a Dockerfile the classic Docker parser accepts, no untouched scaffold, a verifier that writes the reward file, a reference solution with a command), `content_hash` deduplicates tasks across generations, and `oracle_check` runs the task through the `harbor` command line with the oracle and the nop agents and accepts it only when the reference solution scores 1 and doing nothing scores below 1; `openenv_check` builds the openenv task's image, starts the server in a container without network, plays a reset and one step with the Designer's example action and reads the state.

## One generation

```bash
python -m recipes.beta.spade.generation \
  --reef-url http://127.0.0.1:8900 --scenario spade --model qwen3.8:27b \
  --tasks-root tasks --description "multi turn deduction puzzles with hidden state" \
  --skills deduction,planning --kinds gym,openenv,harbor --count 12 --generation 1 \
  --experience tasks/.spade/generation-00000.json
```

`Generation.run(GenerationRequest)` is the same call from Python. The Designer is the served model through Reef, so every proposal is a record with a receipt; the reply is parsed for its kind, the kind's check runs (`gym`: the smoke test on the host; `harbor`: the oracle and nop agents through the `harbor` command line; `openenv`: the docker serve check), the task is written under the tasks root, and a duplicate of a task already there is refused. The solver plays each written task through `reef.harness.client.tasks`: `--plays` times as it is, reported to Reef as training data, and `--hint-plays` times with `solution/hint.txt` appended to the instruction, measured only. Regret is the mean hint reward minus the mean plain reward; the plain mean puts the task in its band. The tasks are split by the Designer's record ids into `manifest-<generation>.json` under the root, which `evolution.task_manifest` and the task player read; `.spade/generation-<generation>.json` keeps every proposal, refusal and measure, and `--experience` hands its records to the next generation's prompts. Each proposal is reported against the Designer's receipt with its regret as the score, the Designer's own training signal; `--no-designer-report` turns that off. `--agent-json` and `--agent-host` choose the solver's Harbor agent as the task player does. `--designer-json` adds fields to the Designer's chat request: a model that thinks for thousands of tokens before writing an environment runs past the service's inference deadline, and `{"reasoning_effort": "none"}` keeps it to the reply.
