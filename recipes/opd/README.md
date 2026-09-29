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

The current separate-teacher loader swaps checkpoints into the actor's model
layout. Teacher and student must have compatible architecture, vocabulary and
token IDs. It does not support an arbitrary larger teacher solely because the
tokenizer matches. The 9B base/post-trained pair has a compatible layout.

`--opd-top-k=1` selects the shared backend's sampled-token reverse KL path.
The loss uses the teacher's probability of the student's sampled token,
normalized over the full vocabulary; it does not train against a renormalized
one-token distribution. Full-distribution KL and other divergence choices are
available through the inherited `--opd-*` flags, but change the experiment.
