# OPD

On-policy distillation trains a student against a separate frozen teacher on
responses sampled by the student. The teacher scores the exact recorded token
IDs. Sampled-token reverse KL updates the student once per batch. Verifier
scores remain evaluation metadata.

This package uses the checkpoint-backed implementation from
[`codex/opd-slime-reproduction` at `a06415d7`](https://github.com/Human-Agent-Society/reef/pull/683)
on the shared native distillation backend introduced by
[Han's PR #533](https://github.com/Human-Agent-Society/reef/pull/533).
The method reference is [Thinking Machines Lab's on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
The implementation runs on Slime/Megatron and makes no Tinker API calls.

## Components

- `recipe.py` binds `OPDProcessor`, the `opd` objective and one update per batch.
- `processor.py` copies the recorded student sequence into `teacher_tokens`.
- `slime/` delegates teacher scoring and loss computation to the shared backend.
- [AgentCL coding](examples/agentcl/README.md) supplies isolated task episodes,
  exact receipt linkage and frozen evaluations.

Reports use `TeacherContextReport` with empty `teacher_context`. Whole-episode
training is opt-in through `accept_multi_turn_policy_samples`. It preserves all
assistant tokens, gives tool observations zero loss, and rejects history drift
or teacher-window overflow. The processor loads no tokenizer and does no prompt
re-rendering.

The regular teacher loader swaps a frozen checkpoint into the actor's model
layout. Student and teacher must share architecture, parallel layout and token
IDs. The separately allocated teacher-engine path in PR #707 remains outside
this example's supported configuration.

See the [OPD guide](../../docs/user-guide/recipes/opd.rst) for configuration.
