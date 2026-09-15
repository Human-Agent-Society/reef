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
```

## The task form

`environment_task` turns one generated environment (its code, skill, generation, index, hint, the Designer's generation record id) into a `HarborTask` from `reef.core.tasks`. The agent never holds the game's code: `instruction.md` says the game is served one turn at a time, `tests/env.py` holds the class unchanged beside `tests/env_loader.py` and `tests/replay.py`, which replay the agent's action log through it and write the episode return (the last step's reward, clipped to [-1, 1], when the episode terminated, else 0), `solution/hint.txt` holds the privileged hint, and `task.toml` carries skill, generation, step, index, difficulty, document, class name, turn limit and seed. `split_generation` puts every environment of one Designer call on one side of a train/eval split.
