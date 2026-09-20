# sdft

Reproduction of [Self-Distillation Fine-Tuning](https://arxiv.org/abs/2601.19897) as a Reef weight-training recipe. SDFT learns from demonstrations on policy: the served model reads a demonstration in its prompt and is the teacher, the same model without it is the student, and the loss is the per-token KL between their next-token distributions along the student's own sample. The package holds the method; its report contract, a rollout's receipt plus the teacher's `context` (`reef.core.reports.TeacherContextReport`), is shared with SDPO.

- Paper: [arXiv:2601.19897](https://arxiv.org/abs/2601.19897)
- Reference implementation: [idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation) at `d77573212fa0`; the recipe's processor maps onto its `main.py` (the demonstration prompt) and the Slime backend's distillation base (`reef/train/slime_backend/distill/`, which the `sdft` family configures) onto `distil_trainer.py` (the loss). Forward KL is the default, as the authors' 2026-04-07 note says the paper's results used it; reverse KL is a switch.
- Pins: `slime` pinned to `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219` (via `pyproject.toml` `dependency-groups.runtime`)
- Claim scope: [SDFT on a skill stream](examples/skill_stream/README.md), the paper's Figure 3 protocol (Tool Use, then Science Q&A) against an SFT control on the same demonstrations. The CEO-Bench comparison is the roadmap's next item ([#502](https://github.com/Human-Agent-Society/reef/issues/502)).

## Layout

```text
sdft/
  recipe.py          SDFTRecipe: training spec, loss family "sdft"; its report contract is reef.core.reports.TeacherContextReport
  processor.py       the shared DistillProcessor with the demonstration appended to the recorded request
  objective.py       selects the sdft loss; the recipe binds the per-sample step schedule
  slime/             the loss family: SDFT's defaults and hook names on the Slime backend's distillation base
  examples/
    skill_stream/    Tool Use, then Science Q&A, through reef-eval: the paper's Figure 3 against an SFT control
```

## Where the rest is documented

[The sdft recipe page](../../docs/user-guide/recipes/sdft.rst) covers the report contract, configuration and the driver flags, and [Loss families](../../docs/developer-guide/loss-families.rst) describes how the family plugs into the Slime backend.
