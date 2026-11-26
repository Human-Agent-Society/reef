Loss families
=============

A loss family implements a weight method's model-dependent computation.
The objective prepares the job's data; the family computes the loss for
each optimizer step.

This page covers Slime's ``SlimeAlgorithm`` contract. Tinker uses
``TinkerLoss`` from ``reef.train.tinker_backend.losses`` and has no worker hooks.
Read `Training service <recipe-training-service.rst>`__ for objective registration.

Layout
------

Keep the family declaration separate from GPU worker code:

.. code:: text

   recipes/<name>/slime/
     __init__.py     objective, SlimeAlgorithm and wire-row helpers; no torch
     objective.py    worker hooks; torch is allowed

Import backend interfaces from ``reef.train.slime_backend``, never
``reef_adapters``. Register family classes with ``@register_loss_family``.
Reef sends their import references to Megatron workers.
Instance registration is local to one process and is suitable for tests.

Family to Slime flags
---------------------

The objective's family and ``training.options.loss-type`` must agree.
Reef rejects a mismatch at startup.

.. list-table::
   :header-rows: 1
   :widths: 65 35

   * - Family
     - ``loss-type``
   * - ``sft``
     - ``sft_loss``
   * - ``slime``, ``sao``
     - ``policy_loss``
   * - ``importance_sampling``, ``tttd``, ``openclawrl``, ``sdft``, ``sdpo``, ``opd``
     - ``custom_loss``

Only ``slime`` accepts ``use-rollout-logprobs``.
Rendered traffic samples normally use actor-computed old-policy probabilities.
``importance_sampling`` instead requires recorded ``rollout_log_probs``;
its input provider must supply them.

The spec
--------

Declare the family, Slime loss type and argument checks:

.. code:: python

   from reef.train.slime_backend.algorithm import SlimeAlgorithm
   from reef.train.slime_backend.loss_families import register_loss_family


   @register_loss_family
   class MyAlgorithm(SlimeAlgorithm):
       loss_family = "my"
       loss_type = "custom_loss"
       advantages = "required"
       required_objective_hooks = ("custom_loss_function_path",)

       def validate_specific_args(self, args, source):
           if args.kl_coef <= 0:
               raise RuntimeError(f"{source} requires --kl-coef > 0")

Use ``custom_loss`` to replace the complete loss.
Set ``uses_pg_loss_primitive=True`` with ``policy_loss`` to replace only
its per-token policy-gradient primitive.

Optional methods extend specific stages:

- ``parse_specific_options`` and ``apply_driver_options`` handle
  ``--<family>-*`` flags before Slime parses its arguments.
- ``configure_backend_args`` derives backend settings.
- ``shape_sample_row`` and ``build_rollout_data`` define custom wire data.
- ``prepare_rollout`` runs before workers train the job.
- ``bind`` creates per-run state; ``train`` controls actor and critic steps.
  The default runs one actor step.
- ``rollout_metrics`` reports step measurements.
- ``policy_gradient_weight`` declares support for `score centering`_.

Objective hooks
---------------

In ``objective.py``, register every declared channel with
``@worker_hook("<channel>")``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Channel
     - Hook arguments
   * - ``custom_loss_function_path``
     - ``args, batch, logits, sum_of_sample_mean``
   * - ``custom_pg_loss_function_path``
     - ``args, ppo_kl, log_probs, advantages``
   * - ``custom_advantage_function_path``
     - ``args, rollout_data``
   * - ``reef_actor_init_hook_path``
     - ``actor``; once after loading weights
   * - ``reef_actor_pre_train_hook_path``
     - ``actor, rollout_data``; before each step

Prefix hook names and flags with the family name.
Missing modules or hooks stop the worker; Reef does not fall back to Slime's
loss. Keep torch imports out of the spec module.

Wire declarations
-----------------

The default row is
``[source_id, tokens, loss_mask, rollout_log_probs, reward]``.
Rendering supplies tokens and masks. Recorded rollout probabilities are
empty for rendered traffic samples; training task rollouts can supply them.

For extra columns, declare:

- ``rollout_data_keys``: per-sample keys partitioned across data-parallel ranks.
- ``rollout_tensor_dtypes``: tensor keys and dtypes: ``int``, ``long`` or
  ``float32``. Undeclared ragged values remain lists.
- ``response_aligned_keys``: tensors sliced by response tokens for context parallelism.
- ``external_batch_keys``: values forwarded to microbatches.
- ``rollout_log_skip_keys``: nonscalar values excluded from numeric logging.

Critic families can set ``critic_value_head_zero_init`` and
``critic_value_mask_key``. Use ``recipes/tttd/slime/`` for default rows,
``recipes/openclawrl/slime/`` for custom rows, or ``recipes/sao/slime/`` for
critic orchestration.

Per-job families
----------------

Jobs can switch objectives and actor loss families. Workers retain startup
flags; the new family supplies its loss type, hooks, routing, wire keys and
flag defaults. The critic retains its startup family.

Reef rejects a switched family that needs custom family flags, configures
the critic, adds a second distillation family or conflicts with an existing
wire dtype. Its validation must also pass against startup flags.
A self-distillation teacher copies actor weights at its first job; that
copy is not checkpointed.

Learning-rate schedules advance actor optimizer steps and survive checkpoints.
Reef runs an existing ``custom-megatron-before-train-step-hook-path`` first.
Schedules are incompatible with ``decoupled-lr``.

The distillation base
---------------------

Extend ``DistillAlgorithm`` for SDFT, SDPO or on-policy distillation.
Declare defaults through ``DistillAlgorithmSettings`` and forward loss and
pre-train hooks to ``reef.train.slime_backend.distill.objective``.

The wire row adds ``teacher_tokens`` and ``sample_weight``.
Teacher tokens contain the teacher prompt followed by student response IDs.
Sample weight defaults to 1; stages can set ``distill_sample_weight``.

Family-prefixed flags select the teacher and divergence:
``teacher``, ``teacher-update-rate``, ``teacher-checkpoint``, ``divergence``,
``top-k``, ``top-k-source``, ``top-k-distribution``,
``importance-sampling-cap``, ``importance-sampling-level``,
``skip-response-tokens`` and ``jsd-beta``.

- ``self`` uses student weights: update rate 1 follows current weights,
  0 freezes a copy, and intermediate values update a slower copy.
- ``separate`` loads a compatible teacher checkpoint beside the actor.

The teacher runs a forward-only pass before training.
Losses support forward KL, reverse KL and generalized JSD over the full
vocabulary or selected top-K IDs. Student-selected top-K requires an extra
forward pass and zero dropout. Top-K distributions can be renormalized or
include a bucket for the remaining vocabulary.

Teacher engine
--------------

A separate teacher can run on a frozen SGLang engine.
It may differ in size and parallel layout, but must use compatible token IDs.
See `OPD teacher engine <../user-guide/recipes/opd.rst#teacher-engine>`__.

The ``teacher`` section starts ``distill-teacher`` and supplies family-prefixed
``teacher-url``, ``teacher-model-path`` and ``teacher-timeout`` flags.
Set those flags directly for an externally managed engine.

Reef checks tokenizer compatibility at startup and renders teacher inputs
with the student's chat template. Scoring happens before worker training;
a failure rejects the whole job's scored batch.

The engine supports sampled reverse KL and teacher-selected top-K forward KL,
reverse KL and JSD at temperature 1. Full-vocabulary and student-selected
top-K modes require a checkpoint teacher. Invalid combinations fail validation.
``perf/distill_teacher_time`` reports scoring seconds.

Score centering
---------------

Score centering corrects policy-gradient drift when sampler probabilities
``q`` differ from trainer probabilities ``p``.
It follows `Score Centering Stabilizes Off-policy Reinforcement Learning
<https://arxiv.org/abs/2609.20807>`__ and is disabled by default:

.. code:: yaml

   training:
     options:
       score-centering: true
       score-centering-top-k: 128
       score-centering-min-tail-mass: 1e-6

Declare support
~~~~~~~~~~~~~~~

The loss must have the form
``-A_t * sg[f(p_t / q_t)] * log p_t``.
Implement ``policy_gradient_weight(args)`` with a ``PolicyGradientWeight``:

- ``none``: ``f(r) = 1``.
- ``truncated``: ``f(r) = min(r, upper)``.
- ``masked``: ``f(r) = r`` strictly inside ``(lower, upper)``, otherwise 0.

Returning ``None`` rejects centering. SAO, importance sampling and sampled
reverse-KL distillation declare support.

For sampler top-K IDs ``H``, Reef adds:

.. code:: text

   A * sum_{v in H} sg[q_v * f(p_v / q_v) - alpha * p_v] * log p_v
   rho   = max(1 - q(H), eps) / max(1 - p(H), eps)
   alpha = rho * f(1 / rho)

``sg`` stops gradients. The sampler tail is approximated by ``rho * p``.
The correction uses the loss's advantages and per-sample mean, and vanishes
when ``q = p``.

Requirements
~~~~~~~~~~~~

Samples must carry sampler ``topk_indices`` and ``topk_log_probs`` under
``extra.reef.training``, with enough entries at every trained position.
Traffic records contain no such tensors. Spindle task rollouts can record
them with ``score_centering`` enabled.

Reef rejects missing or short rows, duplicate or negative IDs, nonfinite
log probabilities, head mass above one and inconsistent sampled-token probabilities.
Teacher probabilities cannot replace sampler probabilities.
Recorded probabilities are after temperature but before top-k, top-p and
min-p filtering; enabled filters therefore leave the correction centered
against the unfiltered distribution.

Slime requires context parallel size 1. Policy-loss families also reject
``use-tis``, ``get-mismatch-metrics``, ``use-opsm`` and a custom PG reducer.
Native ``slime`` task jobs use Slime's ``use-score-centering`` instead.

Sampled on-policy distillation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For distillation centering, select reverse KL, positive teacher top-K and
a renormalized distribution:

.. code:: yaml

   training:
     options:
       sdft-divergence: reverse
       sdft-top-k: 128
       sdft-top-k-distribution: renormalized
       sdft-importance-sampling-level: token
       sdft-importance-sampling-cap: 2.0

Use the family's prefix, such as ``sdpo-`` for SDPO.
Token-level truncated importance weights and cap 0 are supported.
Sequence weights, full-vocabulary KL, top-K-plus-tail losses, forward KL,
JSD and OpenClaw-RL's separate top-K surrogate are unsupported.
The correction's top-K and teacher top-K can differ.

Metrics
~~~~~~~

Step metrics sum per-sample means:
``score_centering_term`` (included in loss),
``score_centering_correction_l1``, sampler and trainer head masses,
``score_centering_tail_ratio`` and ``score_centering_tail_clipped``.
The last metric is the fraction of positions whose tail fell below the floor.
