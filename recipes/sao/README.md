# sao

Reproduction of [Single-Rollout Asynchronous Optimization](https://arxiv.org/abs/2607.07508) as a Reef weight-training recipe. SAO samples one rollout per prompt and grades each on its own: there is no comparison group and no barrier, a scored rollout joins the next optimizer step without waiting for siblings, and the next request is served by the updated weights. The package holds the method; the loop that drives it lives in [examples/imo_answerbench](examples/imo_answerbench/README.md).

- Paper: [arXiv:2607.07508](https://arxiv.org/abs/2607.07508)
- Pins: `slime` pinned to `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219` (via `pyproject.toml` `dependency-groups.runtime`); the completed comparison trained `Qwen3-30B-A3B-Thinking-2507` on DeepMath pools at the paper's batch of 128 rollouts per step, without tools, and evaluated held-out AIME 2025, HMMT February 2025 and IMO-AnswerBench
- Claim scope: in this setting SAO trained stably for 99 steps and gained 2 to 4 points on AIME and IMO-AnswerBench, within per-checkpoint intervals, while a GRPO(+DIS) control (reported, not shipped) matched it through step 40 and then shortened its responses until held-out accuracy collapsed. Two SAO seeds and two control runs, about 130 steps each; the paper's tool-integrated numbers are out of reach here, as the example README's Limitations explain.

## Layout

```text
sao/
  recipe.py       SAORecipe: training spec, loss family "sao", batching, runtime binding
  processor.py    reported feedback, singleton: one scored rollout is one unit
  objective.py    selects the SAO loss; the recipe binds the per-sample step schedule
  slime/          the training-plane objective: DIS ratio, colocated critic
  examples/
    imo_answerbench/  the runnable loop: three IMOAnswerBench problems as Harbor tasks
    ceobench/         CEO-Bench through Reef, trained by the episode
```

## Where the rest is documented

[The sao recipe page](../../docs/user-guide/recipes/sao.rst) covers configuration and runtime metrics, [Evolve your model](../../docs/user-guide/evolve-your-model.rst) walks the training stack, and the [example README](examples/imo_answerbench/README.md) records implementation details, distance from the paper's protocol, and the completed comparison.
