# spade

**Status: beta.** SPADE remains under `recipes/beta/spade/` until complete, reproducible learning results are published. The tests validate the task form; they do not establish learning performance. See #479 for this piece and #447 for the pipeline it belongs to.

[SPADE](https://arxiv.org/abs/2608.19197), self play in adaptive synthetic executable environments, as a Reef recipe: one policy plays an Environment Designer that writes executable environments with the Gym interface and a Reasoning Agent that learns in them. Reef owns the environment formats (a gym task under `reef.core.tasks.gym`, played by the `gym` harness adapter); the recipe owns what SPADE adds on top. The package is being built in pieces.

- Paper: [arXiv:2608.19197](https://arxiv.org/abs/2608.19197)
- Reference code: [spade-rl/spade](https://github.com/spade-rl/spade)

## Layout

```text
beta/spade/
  tasks.py    a generated environment as a gym task with SPADE's naming, metadata and hint; a split per generation
```

## The task form

`environment_task` turns one generated environment (its code, skill, generation, index, hint and the Designer's generation record id) into a gym task from `reef.core.tasks`: the class under `tests/` with the shared loader and the replay verifier, the privileged hint under `solution/hint.txt`, and skill, generation, step, index, difficulty and document in `task.toml` beside the turn limit and the seed. `split_generation` puts every environment of one Designer call on one side of a train/eval split and refuses two tasks with one name.
