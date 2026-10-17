# opd

On-policy distillation from a separate frozen teacher as a Reef weight-training recipe on Slime. The student samples responses through Reef, the teacher scores the exact prompt and response token ids the engine recorded, and the sampled-token reverse KL between the two updates the student once per batch. The package holds the method. Its report contract is `reef.core.reports.TeacherContextReport` with an empty `teacher_context`; a reported score is metadata, not training input.

- Method: [Thinking Machines Lab, On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
- Reference implementation: the [tinker-cookbook distillation recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/dfe4d77e8e8c/tinker_cookbook/recipes/distillation) at `dfe4d77e8e8c`. The Reef implementation configures the Slime backend's distillation base (`reef/train/slime_backend/distill/`) and makes no Tinker API calls.
- Pins: `slime` pinned to `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219` (via `pyproject.toml` `dependency-groups.runtime`).
- Claim scope: the [math example](examples/math/README.md) distils Qwen3-32B into a Qwen3-8B OpenThoughts3-SFT student through the teacher engine and raises AIME'24 accuracy from 41.0% to about 50%, in a run recorded before advantage normalization.

## Layout

```text
opd/
  recipe.py          OPDRecipe: training spec, loss family "opd", and its stages; its report contract is TeacherContextReport
  processor.py       OPDTasks and OPDBatches: the recorded prompt and response ids as the teacher sequence
  objective.py       selects the opd loss; one optimizer step per batch
  slime/             the loss family: OPD's defaults on the Slime backend's distillation base
  examples/
    math/            DeepMath prompts and AIME'24 evaluation through a direct reef_client campaign driver
```

## The teacher

`--opd-teacher=separate` names a checkpoint (`--opd-teacher-checkpoint`) that the teacher pass swaps into the actor's model layout, so teacher and student must share the architecture, parallel layout and token ids. `--opd-top-k=1` selects the base's sampled-token reverse-KL estimator: the loss uses the teacher's log-probability of the student's sampled token, normalized over the full vocabulary, with the score-function gradient. The other divergences and representations of the base are reachable through the inherited `--opd-*` flags; changing them changes the method.

The gradient's raw per-token advantage is the teacher's log-probability of
the sampled token minus the rollout engine's. The family centers these values
and divides them by their standard deviation over every trained token of the
training job. Normalization has no flag. The reported divergence stays
unnormalized.

For optional advantage clipping, set `opd-advantage-clip: 5.0` under
`training.options`. It bounds the normalized advantage to `[-5, 5]` before
importance-sampling and sample weights. Score centering uses the same
advantage. The default is `0`, which applies no bound. Clipping requires
sampled reverse KL with positive `opd-top-k` and
`opd-top-k-distribution: renormalized`. The reported divergence stays
unclipped.

## Teacher engine

The deployment's `teacher` section serves the teacher as a separate frozen
SGLang model that scores the student's recorded token ids, so the teacher can
be larger than the student. Student and teacher must share a tokenizer. That
is the only OPD the recipe supports today.

```yaml
teacher:
  model-path: Qwen/Qwen3-32B
  num-gpus: 2
training:
  options:
    opd-teacher: separate
```

`num-gpus` is the teacher's tensor parallel size on its own GPUs. Drop
`opd-teacher-checkpoint`.

## Where the rest is documented

[The opd recipe page](../../docs/user-guide/recipes/opd.rst) covers the report contract, configuration and the driver flags, and [Loss families](../../docs/developer-guide/loss-families.rst) describes how the family plugs into the Slime backend.
