# opd

On-policy distillation from a separate frozen teacher as a Reef weight-training recipe on Slime. The student samples responses through Reef, the teacher scores the exact prompt and response token ids the engine recorded, and the sampled-token reverse KL between the two updates the student once per batch. The package holds the method. Its report contract is `reef.core.reports.TeacherContextReport` with an empty `teacher_context`; a reported score is metadata, not training input.

- Method: [Thinking Machines Lab, On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
- Reference implementation: the [tinker-cookbook distillation recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/dfe4d77e8e8c/tinker_cookbook/recipes/distillation) at `dfe4d77e8e8c`. The Reef implementation configures the Slime backend's distillation base (`reef/train/slime_backend/distill/`) and makes no Tinker API calls.
- Pins: `slime` pinned to `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219` (via `pyproject.toml` `dependency-groups.runtime`).
- Claim scope: none yet. Experiments are tracked in [#682](https://github.com/Human-Agent-Society/reef/issues/682) under roadmap [#502](https://github.com/Human-Agent-Society/reef/issues/502).

## Layout

```text
opd/
  recipe.py          OPDRecipe: training spec, loss family "opd"; its report contract is TeacherContextReport
  processor.py       the shared DistillProcessor reading the recorded prompt and response ids as the teacher sequence
  objective.py       selects the opd loss; one optimizer step per batch
  slime/             the loss family: OPD's defaults on the Slime backend's distillation base
```

## The teacher

`--opd-teacher=separate` names a checkpoint (`--opd-teacher-checkpoint`) that the teacher pass swaps into the actor's model layout, so teacher and student must share the architecture, parallel layout and token ids. `--opd-top-k=1` selects the base's sampled-token reverse-KL estimator: the loss uses the teacher's log-probability of the student's sampled token, normalized over the full vocabulary, with the score-function gradient. The other divergences and representations of the base are reachable through the inherited `--opd-*` flags; changing them changes the method.

## Teacher engine

For a larger teacher with the same token ids, the deployment's `teacher`
section serves the teacher on its own frozen SGLang process with its own Ray
GPUs, and the bridge scores each recorded sequence there instead of swapping a
checkpoint into the actor:

```yaml
teacher:
  model-path: Qwen/Qwen3-32B
  num-gpus: 2
  port: 30001
  options:
    dtype: bfloat16
    context-length: 32768
    chunked-prefill-size: 2048
    mem-fraction-static: 0.80
training:
  options:
    opd-teacher: separate
    opd-top-k: 1
    rollout-temperature: 1.0
```

`num-gpus` is both the teacher's GPU budget and its tensor parallel size, on
top of the student's training and inference GPUs; its context limit must
cover the teacher prefix plus the recorded response. Drop
`opd-teacher-checkpoint`. Set the student's Megatron `vocab-size` to the
checkpoint's `vocab_size`, the embedding rows including the padding beyond
the tokenizer (151936 for Qwen3): the student samples from the padded rows
too, and the bridge scores them like any other token.

The bridge sends the exact recorded ids to the engine's `/generate` with zero
output tokens, sixteen requests at a time, checks every returned response id,
and fills the teacher probability columns before the optimizer step;
`perf/distill_teacher_time` is the scoring wall time in seconds. A vocabulary
or decoder mismatch, an endpoint serving another model, a missing row or a
timeout fails the step. The engine path supports the sampled reverse KL and
teacher-selected top-K objectives at rollout temperature 1; the exact
full-vocabulary and student-selected top-K representations stay on the
checkpoint teacher. Engine log-probs come from bf16 kernels rather than the
trainer's, a small constant bias for a frozen teacher.

## Where the rest is documented

[The opd recipe page](../../docs/user-guide/recipes/opd.rst) covers the report contract, configuration and the driver flags, and [Loss families](../../docs/developer-guide/loss-families.rst) describes how the family plugs into the Slime backend.
