# spade

**Status: beta.** SPADE remains under `recipes/beta/spade/` until complete, reproducible learning results are published.

[SPADE](https://arxiv.org/abs/2608.19197), self play in adaptive synthetic executable environments, as a Reef recipe: one policy plays an Environment Designer that writes complete Gym style environments and a Reasoning Agent that learns in them. The package is being built in pieces.

- Paper: [arXiv:2608.19197](https://arxiv.org/abs/2608.19197)
- Reference code: [spade-rl/spade](https://github.com/spade-rl/spade)

## Layout

```text
beta/spade/
  environment_loader.py   load and play a generated environment as the reference does; ships into every task
  tasks.py                a generated environment as a Harbor task directory, a replay verifier, a split per generation
  designer.py             the Designer's adversarial prompt, its reply parsed, the smoke test of what it wrote
  play.py                 the agent plays one environment turn by turn, the game in a child process
```

## The task form

`environment_task` turns one generated environment (its code, skill, generation, index, hint, the Designer's generation record id) into a `HarborTask` from `reef.core.tasks`. The agent never holds the game's code: `instruction.md` says the game is served one turn at a time, `tests/env.py` holds the class unchanged beside `tests/env_loader.py` and `tests/replay.py`, which replay the agent's action log through it and write the episode return (the last step's reward, clipped to [-1, 1], when the episode terminated, else 0), `solution/hint.txt` holds the privileged hint, and `task.toml` carries skill, generation, step, index, difficulty, document, class name, turn limit and seed. `split_generation` puts every environment of one Designer call on one side of a train/eval split.

## The Designer

`designer_messages` builds one Designer call: the skill and difficulty, the rules of the environment contract, and what the agent did on the last generation's environments as `PlayRecord` rows, sorted into the frontier (lost without the hint, won with it), the ones it wins anyway and the ones out of reach, so the next environment lands where the agent fails today. `parse_designer_reply` reads the `python` and `hint` blocks of the reply, and `smoke_test` runs the code in a child interpreter before anything else trusts it.

## The play

`play_episode` plays one environment with the agent behind a chat callable, the reference's actor loop: the first observation wrapped in the gameplay prompt (the hint appended for the with hint arm), every later observation a plain user turn, the reply re-boxed before it reaches `step`. The game runs in a child interpreter through `GameProcess` with a timeout per `reset` and `step`, so a game that hangs, exits or floods stdout costs one episode. `Episode.actions` is the action log the task's verifier replays, so the driver's return and the verifier's agree.
