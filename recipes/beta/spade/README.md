# spade

**Status: beta.** SPADE remains under `recipes/beta/spade/` until complete, reproducible learning results are published. The tests validate the task form; they do not establish learning performance. See [issue #479](https://github.com/Human-Agent-Society/reef/issues/479) for this piece, [issue #447](https://github.com/Human-Agent-Society/reef/issues/447) for the pipeline they belong to, and [issue #422](https://github.com/Human-Agent-Society/reef/issues/422) for this classification.

[SPADE](https://arxiv.org/abs/2608.19197), self play in adaptive synthetic executable environments, as a Reef recipe: one policy plays an Environment Designer that writes executable environments and a Reasoning Agent that learns in them. Reef knows one task format, Harbor, so every environment SPADE writes is a Harbor task any Harbor agent can play (in reef's gate today that is the `terminus` adapter); everything SPADE needs beyond the format lives in this package. The package is being built in pieces.

- Paper: [arXiv:2608.19197](https://arxiv.org/abs/2608.19197)
- Reference code: [spade-rl/spade](https://github.com/spade-rl/spade)

## Layout

```text
beta/spade/
  environment_loader.py   load and play a Gym style class as the reference does; ships into every task
  tasks.py                a generated environment as a Harbor task any agent can play; a split per generation
  process.py              the class in a child interpreter on the host, for the smoke test
```

## The task form

`environment_task` turns one generated environment (its class, skill, generation, index, hint and the Designer's generation record id) into a Harbor task from `reef.core.tasks`. The agent never holds the class: the container runs the agent as a non root user, the class and its command live under `/opt/env` readable by root only, and a sudo rule lets the agent reach that one command through `observe` and `act`, which replay the root held action log from the seed, take one action when asked and print the next observation. `tests/replay.py` replays the same log through the same loader and writes the episode return (the last step's reward, clipped to [-1, 1], when the episode terminated, else 0). `solution/hint.txt` holds the privileged hint, and `task.toml` carries kind, skill, generation, step, index, difficulty, document, class name, turn limit and seed. One task is one seeded instance of the class: every play starts from the same hidden state, so a generation that wants the reference's spread over instances writes one task per seed. `split_generation` puts every environment of one Designer call in one split and refuses two tasks with one name.
