Self-Distillation Policy Optimization (SDPO)
============================================

SDPO (arXiv:2601.20802) improves a policy by letting its teacher read privileged
feedback while the student continues to read the original question. The
``recipes.sdpo.recipe:SDPORecipe`` cookbook recipe uses Slime's shared
distillation backend with Section 3 defaults: an EMA teacher (rate 0.05),
Jensen-Shannon divergence, the student's top 100 vocabulary entries plus their
remaining probability mass, and detached per-token importance weights capped at 2.

Sampling and feedback
---------------------

Sample ``groups-per-step`` questions, each with ``rollouts-per-group`` attempts,
from one policy release. Defaults are 32 questions and 8 attempts. Report each
recorded inference receipt with a score and coordinates:

.. code-block:: python

   client.report("sdpo", {
       "references": [receipt],
       "score": score,
       "metadata": {
           "step": 0,
           "group": question_index,
           "rollout": attempt_index,
           "teacher_context": environment_feedback,
       },
   })

Only the complete grid can train. Reports at an occupied coordinate are retries;
the first accepted report wins. A step with missing or mixed policy releases,
or different requests within one question group, is discarded and exposed in
the processor's ``failed_steps`` status. Wait for the training release before
sampling the next step; set ``max-staleness: 0``.

The teacher sees the first successful sibling in rollout order, excluding the
sample itself unless ``allow-own-success-as-demonstration`` is set. Success
means score >= 0.5, matching the author's resolved actor configuration.
Thinking blocks in demonstrations are removed.
Section 3 disables environment feedback, so a score alone does not invent a
textual answer or make a failed group trainable. Enable
``include-environment-feedback`` to consume ``teacher_context``; by default a
successful solution takes precedence over environment feedback.

A rollout whose teacher read nothing privileged stays in the batch with the
plain request and a sample weight of 0 in the shared distillation row, so
even an entirely inactive step performs its zero-gradient optimizer step. The
reference token-means each one-sample microbatch and then averages all
samples, inactive rows included; the weight reproduces that sequence mean
whatever the trainer's packing, and the step's ``distill_sample_weight``
metric is the active fraction. Teacher prompts are right-truncated at
``max-teacher-prompt-tokens`` (10240), then the student's response token IDs
are appended verbatim. An overlong total teacher sequence fails the batch
rather than silently dropping part of the grid.

Configuration
-------------

.. code-block:: yaml

   recipe:
     implementation: recipes.sdpo.recipe:SDPORecipe
     config:
       tokenizer-path: /models/Qwen3-8B
       groups-per-step: 32
       rollouts-per-group: 8
       max-staleness: 0
       max-teacher-prompt-tokens: 10240
       max-teacher-tokens: 18432
       include-environment-feedback: false
       enable-thinking: false
   training:
     backend: slime
     options:
       loss-type: custom_loss
       use-rollout-logprobs: true
       disable-compute-advantages-and-returns: true
       num-steps-per-rollout: "1"
       attention-dropout: "0.0"
       hidden-dropout: "0.0"
       seq-length: "18944"
       sdpo-teacher: self
       sdpo-divergence: jsd
       sdpo-top-k: "100"
       sdpo-top-k-source: student
       sdpo-top-k-distribution: tail
       sdpo-teacher-update-rate: "0.05"
       sdpo-importance-sampling-level: token
       sdpo-importance-sampling-cap: "2.0"

This is a method configuration fragment, not a complete GPU deployment. The
backend uses an extra student forward to select vocabulary support, followed by
the teacher forward. Context parallelism must be 1. Only one optimizer update
per sampling step is supported; the sequential minibatch updates in the paper's
Section 4 are not implemented by this recipe.

The existing teacher implementation keeps its EMA accumulator on the host and
does not checkpoint it. Restarting from actor weights reseeds that accumulator,
so interrupted runs must not be presented as exact continuations of a paper run.
SDFT retains its previous teacher-selected, renormalized top-K and sequence
importance-weight defaults.

See `the reproduction guide <../../../recipes/sdpo/examples/paper/README.md>`_
for the pinned reference check and experiment protocol.
