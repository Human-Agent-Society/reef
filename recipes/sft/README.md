# sft

Supervised fine-tuning on demonstrations as a Reef weight-training recipe: the control arm of the self-distillation comparisons ([#502](https://github.com/Human-Agent-Society/reef/issues/502)). Each report carries a rollout's receipt and a demonstration as `context` (`reef.core.reports.TeacherContextReport`, the report the self-distillation recipes take), the demonstration becomes the assistant turn of the recorded request in the served model's chat template, and Slime's stock `sft_loss` trains its tokens. The student's own sample is recorded and ignored, so a harness drives SFT and SDFT unchanged.

- Pins: `slime` pinned to `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219` (via `pyproject.toml` `dependency-groups.runtime`)
- Claim scope: [SFT on a skill stream](examples/skill_stream/README.md), the control arm of Figure 3 of arXiv:2601.19897.
## Layout

```text
sft/
  recipe.py          SFTRecipe: training spec, loss family "sft"; its report contract is reef.core.reports.TeacherContextReport
  processor.py       reported feedback, singleton: one report is one supervised sample, the demonstration as the assistant turn
  objective.py       selects the sft loss; the recipe binds the per-sample step schedule
  slime/             the loss family: Slime's stock sft_loss, advantages refused
  examples/
    skill_stream/    Tool Use, then Science Q&A, through reef-eval: what sequential SFT learns and forgets
```

## Where the rest is documented

[The sft recipe page](../../docs/user-guide/recipes/sft.rst) covers the report contract and configuration, and [Loss families](../../docs/developer-guide/loss-families.rst) describes how the family plugs into the Slime backend.
