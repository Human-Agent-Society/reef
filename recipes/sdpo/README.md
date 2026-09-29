# sdpo

Reproduction of [Self-Distillation Policy Optimization](https://arxiv.org/abs/2601.20802) as a Reef weight-training recipe. SDPO samples a question several times and turns the attempts that succeed into teachers for the others. The teacher is the model itself reading the question with a correct attempt appended. The student reads the question alone. The loss is the per-token divergence between their next-token distributions along the student's own attempt. The package holds the method. Its report contract is `reef.core.reports.TeacherContextReport` plus the attempt's place in the sampling grid.

- Paper: [arXiv:2601.20802](https://arxiv.org/abs/2601.20802)
- Reference implementation: [lasgroup/SDPO](https://github.com/lasgroup/SDPO) at `7c457fc1b1f6`. The recipe's processor maps onto its reprompt templates and the Slime backend's distillation base (`reef/train/slime_backend/distill/`, which the `sdpo` family configures) onto its loss.
- Pins: `slime` pinned to `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219` (via `pyproject.toml` `dependency-groups.runtime`)
- Claim scope: [SDPO on SciKnowEval Chemistry](examples/sciknoweval/README.md), the paper's Chemistry run on Qwen3-8B.

## Layout

```text
sdpo/
  recipe.py          SDPORecipe: training spec, loss family "sdpo"; its report contract is report.SDPOReport
  report.py          SDPOReport: TeacherContextReport plus the attempt's step, group and rollout
  processor.py       the shared DistillProcessor with a whole sampling grid as its batch unit
  objective.py       selects the sdpo loss; one optimizer step per grid
  slime/             the loss family: SDPO's defaults and hook names on the Slime backend's distillation base
  examples/
    sciknoweval/     the Chemistry split of SciKnowEval on Qwen3-8B through reef-eval
```

## Where the rest is documented

[The sdpo recipe page](../../docs/user-guide/recipes/sdpo.rst) covers the report contract, configuration and the driver flags, and [Loss families](../../docs/developer-guide/loss-families.rst) describes how the family plugs into the Slime backend.
