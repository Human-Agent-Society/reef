# opd

On-policy distillation from a separate frozen teacher, implemented as a Reef
weight-training recipe on Slime. The student samples responses; the teacher
scores those exact prompt and response token IDs; the sampled-token reverse KL
updates the student. Reports link each response to its inference receipt and
leave `teacher_context` empty. Scalar correctness rewards are not training input.

- Method: [Thinking Machines Lab, On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
- Reference code: [distillation cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/dfe4d77e8e8c/tinker_cookbook/recipes/distillation)
  at `dfe4d77e8e8c`. This is a method reference; the Reef implementation runs on
  Slime/Megatron and SGLang and makes no Tinker API calls.
- Runtime: `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219`, as pinned by Reef.
- Tracking: [experiment #682](https://github.com/Human-Agent-Society/reef/issues/682),
  under [roadmap #502](https://github.com/Human-Agent-Society/reef/issues/502).

## Scope of reproduction

The original October 2025 math experiment starts from Qwen3-8B-Base trained on
400,000 OpenThoughts3 examples. The blog's OPD footnote identifies Qwen3-8B as
the actual teacher, although the introduction and compute comparison name
Qwen3-32B. It reports about 150 OPD updates, 77,000 prompts, four responses per
prompt, and AIME'24 improving from 60% to 70%.

The June 2026 cookbook update changes the pair to Qwen3.5-9B-Base and
Qwen3.5-9B. The example being developed here targets that newer pair with full
parameter updates on B200. Changing the architecture and SFT initialization
means this is a reproduction of the method on the updated model pair, not an
identical replay of the original experiment. A smoke test or a passing unit
suite does not establish the reported AIME gain.

The example uses 512 prompts times four responses: 2,048 responses per
optimizer step. Smaller batches and shorter windows are suitable for startup
checks but must not be presented as the full experiment. An experiment report
must state its actual batch, initialization, training budget, evaluation settings
and deviations.
No benchmark result is claimed until the baseline and trained model have been
evaluated using the same held-out protocol.

## Implementation

`recipe.py` binds the `opd` objective and `OPDProcessor`. `processor.py` preserves
the recorded token IDs rather than re-rendering the prompt. `slime/` configures
the shared distillation backend with a separate frozen teacher, sampled-token
reverse KL, and one update per on-policy batch. The next batch must wait for
publication of the updated weights.

The default `teacher-checkpoint` mode swaps checkpoints into the actor's model
layout. Teacher and student must have compatible architecture, vocabulary and
token IDs. The 9B base/post-trained pair has a compatible layout.

For a different-size teacher with the same token mapping, use the opt-in
[independent teacher example](examples/math/serve.teacher-engine.yaml). Its
`teacher` section launches a frozen SGLang process with a separate Ray GPU
allocation. For example, a Qwen3-4B-Base student can learn from Qwen3-8B:

```yaml
teacher:
  model-path: Qwen/Qwen3-8B
  num-gpus: 1
  port: 30001
  options:
    dtype: bfloat16
    context-length: 32768
    mem-fraction-static: 0.6
training:
  options:
    opd-teacher: separate
    opd-top-k: 1
    rollout-temperature: 1.0
```

The example sets `vocab-size: 151936` to match the Qwen3 student embedding
rows, including padding beyond the tokenizer length. For another student, set
this option to the model configuration's `vocab_size`.

Remove `opd-teacher-checkpoint` when using this section. `teacher.num-gpus`
is both the dedicated GPU budget and teacher tensor parallel size; it is
additional to student training/inference GPUs and never joins their weight
updates. Reef owns startup, readiness and shutdown. The teacher needs a context
limit covering the full teacher prompt plus recorded response.

The bridge sends exact recorded token IDs to `/generate` with zero output
tokens, checks every scored response ID, and fills the existing teacher
probability columns before starting the optimizer step. `perf/distill_teacher_time`
reports scoring wall time in seconds. Vocabulary/decoder mismatches, an endpoint
serving a different model, missing probability rows and timeouts fail the step;
there is no teacher-generated replacement response or correctness reward.

This mode supports sampled reverse KL and teacher-selected top-K objectives.
SGLang input logprobs are untempered, so `rollout-temperature` must be 1.
Exact full-vocabulary (`top-k: 0`) and student-selected top-K retain checkpoint
mode. The pinned SGLang API accepts one arbitrary-ID list per request rather
than per-position student top-K lists; that optimization is not implemented.
Engine and Megatron BF16 kernels need not produce identical probabilities;
measure their bias when comparing experiments. This integration alone makes no
AIME improvement claim.

`--opd-top-k=1` selects the shared backend's sampled-token reverse KL path.
The loss uses the teacher's probability of the student's sampled token,
normalized over the full vocabulary; it does not train against a renormalized
one-token distribution. Full-distribution KL and other divergence choices are
available through the inherited `--opd-*` flags, but change the experiment.
