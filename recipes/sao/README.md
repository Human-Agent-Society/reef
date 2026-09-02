# sao

Reproduction of [Single-Rollout Asynchronous Optimization](https://arxiv.org/abs/2607.07508) as a Reef weight-training recipe. SAO trains on one graded rollout at a time: there is no comparison group and no barrier, a rollout enters training the moment its score arrives, and the next request is served by the updated weights. The package holds the method; the loop that drives it lives in [examples/sao](examples/sao/README.md).

- Paper: [arXiv:2607.07508](https://arxiv.org/abs/2607.07508)
- Pins: `slime` pinned to `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219` (via `pyproject.toml` `dependency-groups.runtime`); the completed comparison ran `Qwen3-30B-A3B-Thinking-2507` on three IMOAnswerBench problems with 48 scored rollouts per run
- Claim scope: the mean rewards order as the paper predicts at the paper's model scale, SAO at 0.479 above the untrained base at 0.458 above a GRPO(+DIS) control at 0.417. Three problems, one run each; this is an ordering result, not a benchmark-wide reproduction.

## Layout

```text
sao/
  recipe.py       SAORecipe: training spec, loss family "sao", runtime binding
  processor.py    reported feedback, singleton: one scored rollout is one unit
  preparer.py     builds the per-rollout policy batch for the driver
  slime/          the training-plane objective: DIS ratio, colocated critic
  examples/sao/   the runnable loop: Harbor tasks, agent, serve.yaml, results
```

## Where the rest is documented

[The sao recipe page](../../docs/user-guide/recipes/sao.rst) covers configuration and runtime metrics, [Evolve your model](../../docs/user-guide/evolve-your-model.rst) walks the training stack, and the [example README](examples/sao/README.md) records implementation details, distance from the paper's protocol, and the completed comparison.
