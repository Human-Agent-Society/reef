# opd

On-policy distillation from a separate frozen teacher as a Reef weight-training recipe on Slime. The student samples responses through Reef, the teacher scores the exact prompt and response token ids the engine recorded, and the sampled-token reverse KL between the two updates the student once per batch. The package holds the method. Its report contract is `reef.core.reports.TeacherContextReport` with an empty `teacher_context`; a reported score is metadata, not training input.

- Method: [Thinking Machines Lab, On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
- Reference implementation: the [tinker-cookbook distillation recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/dfe4d77e8e8c/tinker_cookbook/recipes/distillation) at `dfe4d77e8e8c`. The Reef implementation configures the Slime backend's distillation base (`reef/train/slime_backend/distill/`) and makes no Tinker API calls.
- Pins: `slime` pinned to `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219` (via `pyproject.toml` `dependency-groups.runtime`).
- Claim scope: [OPD on mathematical reasoning](examples/math/README.md), Qwen3.5-9B-Base after OpenThoughts3 SFT distilled from Qwen3.5-9B and evaluated on AIME'24. Tracked in [#682](https://github.com/Human-Agent-Society/reef/issues/682) under roadmap [#502](https://github.com/Human-Agent-Society/reef/issues/502).

## Layout

```text
opd/
  recipe.py          OPDRecipe: training spec, loss family "opd"; its report contract is TeacherContextReport
  processor.py       the shared DistillProcessor reading the recorded prompt and response ids as the teacher sequence
  objective.py       selects the opd loss; one optimizer step per batch
  slime/             the loss family: OPD's defaults on the Slime backend's distillation base
  examples/
    math/            DeepMath prompts and AIME'24 evaluation through a direct reef_client campaign driver
```

## The teacher

`--opd-teacher=separate` names a checkpoint (`--opd-teacher-checkpoint`) that the teacher pass swaps into the actor's model layout, so teacher and student must share the architecture, parallel layout and token ids; the Qwen3.5-9B base/post-trained pair does. `--opd-top-k=1` selects the base's sampled-token reverse-KL estimator: the loss uses the teacher's log-probability of the student's sampled token, normalized over the full vocabulary, with the score-function gradient. The other divergences and representations of the base are reachable through the inherited `--opd-*` flags; changing them changes the method.

## Where the rest is documented

[The opd recipe page](../../docs/user-guide/recipes/opd.rst) covers the report contract, configuration and the driver flags, and [Loss families](../../docs/developer-guide/loss-families.rst) describes how the family plugs into the Slime backend.
