OPD: train on student samples with a frozen teacher
==================================================

On-policy distillation (OPD) updates model weights using a separate frozen
teacher. The student samples a response. The teacher scores the exact recorded
student token IDs, and sampled-token reverse KL updates the student.

Use ``recipes.opd.recipe:OPDRecipe`` for regular recorded responses. The
`AgentCL coding example <../../../recipes/opd/examples/agentcl/README.md>`__
adds isolated Harbor episodes, complete receipt linkage, ordered training and
frozen evaluations. Existing AgentCL examples also support
`SDFT <../../../recipes/sdft/examples/agentcl/README.md>`__ and
`SDPO <../../../recipes/sdpo/examples/agentcl/README.md>`__.

Implementation and teacher requirements
--------------------------------------

This integration uses PR #683's checkpoint-backed implementation from
``codex/opd-slime-reproduction`` at ``a06415d7``. It uses Han's shared native
distillation backend from PR #533. The Slime loader swaps the teacher checkpoint
into the actor's layout for a forward-only pass. Teacher and student must have
matching architecture, parallel layout and token IDs. The independent
teacher-engine implementation in PR #707 needs separate support and GPU
qualification.

The processor copies the captured student sequence into ``teacher_tokens``.
It loads no tokenizer and renders no replacement prompt. A verifier score can
be retained in the report, but the OPD objective uses the teacher distribution
as its training signal.

Report and configuration
------------------------

A terminal report includes every ordered inference receipt and empty teacher
context:

.. code:: json

   {
     "references": ["<turn-1-receipt>", "<turn-2-receipt>"],
     "score": 1.0,
     "metadata": {"teacher_context": ""}
   }

Recipe settings:

.. config::

   batch_size | 1 | complete samples in an optimizer batch.
   max_teacher_tokens | 0 | maximum teacher sequence length; 0 disables this check.
   accept_multi_turn_policy_samples | false | retain one complete episode per report with exact history and zero tool/context loss.
   max_staleness | 0 | accepted lag between the sampled and current policy release.

Single-turn overflow releases the report and counts it. Whole-episode overflow
raises an error and keeps its records protected. Missing turns, changed token
history or mixed policy releases also fail explicitly.

Native training flags:

.. config::

   --opd-teacher | separate | frozen checkpoint teacher.
   --opd-teacher-checkpoint | required | checkpoint with a compatible actor layout and token mapping.
   --opd-divergence | reverse | sampled reverse KL for this method.
   --opd-top-k | 1 | selects sampled-token reverse KL; sampled probabilities use the full vocabulary.
   --opd-teacher-update-rate | 0 | the separate teacher remains frozen.
   --opd-importance-sampling-cap | 0 | no sampler/trainer importance correction in the regular OPD configuration.
   --opd-skip-response-tokens | 0 | retain every selected assistant token.

The caller waits for publication of each updated release before sampling the
next training batch. Frozen evaluations produce no reports or teacher lookup.

AgentCL usage
-------------

From ``recipes/opd/examples/agentcl``:

.. code:: bash

   export AGENTCL_TEACHER_CHECKPOINT=/models/compatible-frozen-teacher
   ./run.sh train --profile smoke --data-root /path/to/data --run-root /path/to/opd-smoke --dry-run

Dry-run validates the configuration and prints the proposal without loading
models or submitting training. Follow the example README for export,
reference-verifier qualification, external service supervision and all phases.
Use distinct run roots and scenarios for OPD, SDFT and SDPO.

Validation limits
-----------------

CPU tests check exact episode assembly, masked loss gradients, report and
commit handoff, configuration and resume behavior. A native AgentCL smoke
verified two OPD updates, a compatible frozen-teacher load, changed weights,
exact token masks and a fresh checkpoint reload. Other configurations and full
benchmark runs require independent verification. Final old-node cleanup and
full recovery-tree byte identity remained unverified after node eviction.
Numerical parity and benchmark improvement remain unestablished.
