Loss families
=============

A method's ``TrainingObjective`` owns signal preparation and declares its
``loss_family``. A loss family implements the model-dependent computation for
one backend. Preparation runs on the full batch; tensor loss hooks run after
the backend's forward passes.

- ``recipes/<name>/objective.py`` holds the backend-neutral objective.
  ``reef/train/algos/`` defines its shared contract and registry.
- ``reef/train/slime_backend/`` holds Slime integration machinery; each
  method's Slime implementation lives in ``recipes/<name>/slime/``. Its spec
  is torch-free driver code and its ``objective.py`` contains worker hooks.
- Methods can also supply a Tinker loss implementation. TTTD shares one
  preparation method between its Slime and Tinker implementations.

A recipe binds ``WeightTrainingSpec(objective=..., processor=..., scheduling=...)``.
``WeightTrainingSpec.loss_family`` derives the family from that objective;
``StepSignal`` carries advantages, metrics and proposed state, and the recipe's
``StepScheduling`` says how the runtime cuts the batch into optimizer steps.
The bridge still rejects a payload whose ``loss`` differs from its boot family.

Layout
------

.. code:: text

   recipes/<name>/slime/
     __init__.py     the spec: a SlimeAlgorithm subclass, no torch
     objective.py    the @objective hooks, torch
     utils/          driver-side helpers, only when the wire row is custom

A package ``__init__`` registers its family by reference
(``register_loss_family_ref("<name>", "my_pkg.slime:MyAlgorithm")``);
``loss_families.py`` imports the reference on first resolve so
``@register_loss_family`` runs at boot. An external family decorates its class
the same way. The recipe names it in ``training_spec().loss_family``; the
driver reads that binding after importing the configured recipe class. Resolving the
reference imports the module, which registers the family; the driver also keeps
the reference on ``args.loss_family_ref`` so each Megatron worker, whose
registry starts empty, can import it too.

Family to driver flags
----------------------

The recipe's ``loss_family`` and the driver's flags must describe the same
objective; the driver checks it at start and refuses a mismatch.

+---------------------+-----------------------------+----------------------------+
| Loss family         | ``--loss-type``             | Rollout log-probs          |
+=====================+=============================+============================+
| ``sao``             | ``policy_loss``             | ``--use-rollout-logprobs`` |
+---------------------+-----------------------------+----------------------------+
| ``tttd``            | ``custom_loss``             | ``--use-rollout-logprobs`` |
+---------------------+-----------------------------+----------------------------+
| ``openclawrl``      | ``custom_loss``             | not required               |
+---------------------+-----------------------------+----------------------------+
| ``sdft``            | ``custom_loss``             | ``--use-rollout-logprobs`` |
+---------------------+-----------------------------+----------------------------+

The spec
--------

.. code:: python

   from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family


   @register_loss_family
   class MyAlgorithm(SlimeAlgorithm):
       loss_family = "my"
       loss_type = "custom_loss"
       advantages = "required"
       required_objective_hooks = ("custom_loss_function_path",)

       def validate_specific_args(self, args, source):
           if getattr(args, "kl_coef", 0) <= 0:
               raise RuntimeError(f"{source} requires --kl-coef > 0")

``loss_family``, ``loss_type``, and ``validate_specific_args`` are required. The
rest has defaults. The overrides, in the order the pipeline reaches them:

- ``parse_specific_options`` and ``apply_driver_options``: family flags on
  the driver's argv (``--<name>-*``), stripped before Slime's parser sees them
  and stamped onto ``args``.
- ``configure_backend_args``: derive backend settings once ``loss_family`` is
  stamped. SAO turns Slime's advantage pass on here.
- ``shape_sample_row`` and ``build_rollout_data``: a custom wire row. The
  default row is ``[source_id, tokens, loss_mask, rollout_log_probs, reward]``;
  a family appends its own columns and reads them back.
- ``prepare_rollout``: driver-side work before a step.
- ``bind``: a per-run instance carrying state such as a critic schedule.
- ``train``: critic and actor orchestration; the default is one actor step.
- ``rollout_metrics``: rollout version and timing metrics after the step.
- ``policy_gradient_weight``: the weight the family's loss puts on the
  sampled token's score, which score centering needs (below).

Two loss lanes
--------------

``loss_type = "custom_loss"`` replaces Slime's loss with the
``custom_loss_function_path`` hook (``tttd``, ``openclawrl``, ``sdft``).
``uses_pg_loss_primitive = True`` keeps Slime's ``policy_loss`` and swaps only
the per-token primitive through ``custom_pg_loss_function_path`` (``sao``); the
adapter layer points Slime's CISPO callsite at it.

Objective hooks
---------------

``objective.py`` registers each channel the spec listed in
``required_objective_hooks``. The worker imports the module at init, checks that
every declared channel is present, and projects the dotted paths onto ``args``.
A missing module or hook stops the worker; nothing falls back to Slime's default
loss.

- ``custom_loss_function_path``: ``<name>_loss(args, batch, logits, sum_of_sample_mean)``
- ``custom_pg_loss_function_path``: ``<name>_loss(args, ppo_kl, log_probs, advantages)``
- ``custom_advantage_function_path``: ``<name>_advantages(args, rollout_data)``
- ``reef_actor_init_hook_path``: ``<name>_actor_init(actor)``, once, after the
  actor has loaded its weights
- ``reef_actor_pre_train_hook_path``: ``<name>_actor_pre_train(actor, rollout_data)``,
  before every training step

``tests/reef_service/test_slime_algorithm_contract.py`` enforces the entry point
names and the layering: family packages never import ``reef_adapters``, the
adapter layer never names a family, ``utils/`` stays torch-free, family flags
carry the ``--<name>-`` prefix.

Wire declarations
-----------------

A family that ships more than the five policy columns declares them on the spec.

- ``rollout_data_keys``: per-sample payload keys the rollout manager
  partitions across data-parallel ranks.
- ``rollout_tensor_dtypes``: which of those become tensors, and as what
  (``"int"``, ``"long"``, ``"float32"``). Ragged fields stay undeclared and
  pass through as lists.
- ``response_aligned_keys``: tensors laid out per response token; the worker
  slices them for context parallelism the way it slices advantages.
- ``external_batch_keys``: keys the worker forwards through ``get_batch``
  into the microbatch.
- ``rollout_log_skip_keys``: non-scalar keys hidden from Slime's numeric
  rollout logger.
- ``critic_value_head_zero_init`` and ``critic_value_mask_key``: critic
  families only.

Bundled families worth reading: ``recipes/tttd/slime/`` (two hooks, the default
row), ``recipes/sao/slime/`` (critic schedule, the pg-primitive lane),
``recipes/openclawrl/slime/`` (a custom row, both actor lifecycle hooks, a
frozen Megatron teacher), ``recipes/sdft/slime/`` and ``recipes/sdpo/slime/``
(thin families on the distillation base below).

The distillation base
---------------------

The recipes that distil a teacher on the student's own samples (SDFT, SDPO,
on-policy distillation) compute the same per-token divergence and differ in
who the teacher is and which divergence is minimized. Both are settings of
one implementation in the backend, ``reef/train/slime_backend/distill/``,
and each such recipe's family is a thin subclass of it:

- ``DistillAlgorithm`` is the driver-side base: the seven-column wire row
  (the policy row plus ``teacher_tokens``, the teacher's prompt ids followed
  by the student's response ids verbatim, and ``sample_weight``, the factor
  on that sample's mean divergence, 1 unless the recipe's processor sets
  ``distill_sample_weight`` on the sample), the ``--<name>-*`` flags under
  the family's own prefix (``teacher``, ``divergence``, ``top-k``,
  ``top-k-source``, ``top-k-distribution``, ``teacher-update-rate``,
  ``teacher-checkpoint``, ``importance-sampling-cap``,
  ``importance-sampling-level``, ``skip-response-tokens``, ``jsd-beta``) and
  the settings they stamp on ``args`` under ``distill_*`` names, which the
  worker hooks read whatever the prefix was. A family names itself, sets
  its defaults in a ``DistillSettings`` subclass, and its ``objective.py``
  forwards ``<name>_loss`` and ``<name>_actor_pre_train`` to
  ``distill.objective``.
- The teacher is ``self`` (the student's own weights reading the privileged
  prefix: the current weights at update rate 1, a slow-moving copy below it,
  a frozen snapshot at 0) or ``separate`` (another checkpoint that fits the
  actor's model, loaded beside the actor's weights). The pre-train hook
  switches the teacher's weights in through the actor's backups, runs one
  forward-only pass over the batch's teacher sequences, and switches the
  actor back.
- The divergence is the forward KL, the reverse KL or the generalized JSD,
  over the teacher's whole distribution (``top-k`` 0: one row of this rank's
  vocab shard per response position, kept in float16 on the host) or over K
  ids per position, the teacher's own top-K or the current student's (one
  more forward before the step, which needs zero dropout), either
  renormalized over them, the reverse KL then estimated at the sampled
  token, or with one bucket for the rest of the vocabulary as SDPO's
  reference does. The kernels reduce across the vocab shards of tensor
  parallel with ``reef/train/slime_backend/vocab_parallel.py`` and write the
  gradients out where autograd over one shard would drop the coupling
  through the global log-sum-exp;
  ``tests/reef_service/test_distill_parity.py`` pins them to a pure-Python
  reference and to the dense gradients across four ranks.

The base registers no family and imports nothing from ``reef_adapters``;
``recipes/openclawrl/slime/`` imports its packing schedule from it. The
operations over vocab shards (log-sum-exp, the log-probs at ids on any
shard, the top-K ids) live in ``reef/train/slime_backend/vocab_parallel.py``,
shared by the distillation base, score centering and OpenClaw-RL's teacher.

Score centering
---------------

Score centering is a correction Reef adds to a family's own policy-gradient
loss; it is not a loss family. It implements `Score Centering Stabilizes
Off-policy Reinforcement Learning <https://arxiv.org/abs/2609.20807>`_
(Appendix A, equations 9-14). When the rollout engine's distribution ``q``
differs from the trainer's ``p`` (a quantized engine, stale weights), a
policy gradient drifts toward ``q``; score centering subtracts that drift.
It is off unless ``--score-centering`` is set, and only a family that
declares the weight its loss puts on the sampled token's score accepts it.
SAO declares one.

A family declares its weight with ``policy_gradient_weight``. Its loss must
have the form ``-A_t * sg[f(p_t / q_t)] * log p_t``, with ``q`` the rollout
engine's probability:

.. code:: python

   def policy_gradient_weight(self, args):
       # SAO: the ratio masked to its trust region.
       return PolicyGradientWeight("masked", lower=1 - args.eps_clip, upper=1 + args.eps_clip_high)

``PolicyGradientWeight`` (``reef.train.slime_backend.algorithm``) is ``none``
(``f = 1``, plain off-policy REINFORCE), ``truncated`` (``min(r, upper)``) or
``masked`` (``r`` strictly inside ``(lower, upper)``, else 0). The default,
``None``, refuses score centering: a clipped surrogate against a recomputed
old policy or a distillation loss has no such weight. For unclipped
importance sampling (``f = r``) the term below is identically zero, since
that estimator has no drift.

Reef then adds this term to the family's loss at every trained response
position:

.. code:: text

   A * sum_{v in H} sg[q_v * f(p_v / q_v) - alpha * p_v] * log p_v

   rho   = max(1 - q(H), eps) / max(1 - p(H), eps)
   alpha = rho * f(1 / rho)

``H`` is the sampler's recorded top-K ids, the sampler's tail is
approximated as ``rho * p``, and ``sg`` stops the gradient. The term uses the
loss's own advantages and is reduced with the same per-sample mean, so the
loss's weighted score ends up centered under the sampler, tail included. It
is zero when ``q = p``. It is added inside Slime's policy loss for a stock or
pg-primitive family, and around the family's custom loss otherwise.

To enable it, record the sampler's top-K and set the flags in
``training.options``:

.. code:: yaml

   inference:
     handler-factory: reef.inference.sglang.chat.SGLangInferenceHandler
     handler-config:
       capture_topk: 128        # at least score-centering-top-k
   training:
     options:
       score-centering: true
       score-centering-top-k: 128

Each step then reports the ``score_centering_*`` metrics listed below.

+-----------------------------------+---------+-------------------------------------+
| Flag                              | Default | Meaning                             |
+===================================+=========+=====================================+
| ``score-centering``               | off     | add the term to the family's loss   |
+-----------------------------------+---------+-------------------------------------+
| ``score-centering-top-k``         | 128     | sampler log-probs read per position |
+-----------------------------------+---------+-------------------------------------+
| ``score-centering-min-tail-mass`` | 1e-6    | floor applied to both tail masses   |
+-----------------------------------+---------+-------------------------------------+

Inputs and requirements:

- Records need the sampler's top-K. Serve through a token-native handler
  (SGLang or vLLM) with ``inference.handler-config.capture_topk`` at least
  ``score-centering-top-k``. The recorded log-probs are the same distribution
  as ``rollout_log_probs`` and the trainer's: after temperature and before the
  top-k, top-p and min-p filters (see the configuration reference). With
  those filters on, the correction centers against the unfiltered
  distribution.
- When the flag is on, the bridge adds each wire row's recorded top-K to the
  payload as ``sampler_topk_indices`` and ``sampler_topk_log_probs``; the
  family's wire row is unchanged. Multi-turn assembly keeps the recorded rows
  aligned with the joined response, gives inserted context an empty row, and
  drops top-K for the whole sample when a turn has none.
- Before each step, the bridge keeps the first ``top-k`` entries of every
  trained position. It refuses a sample with missing or short rows,
  duplicate or negative ids, non-finite log-probs, or a head mass above one,
  and, when the sample carries ``rollout_log_probs``, a head whose entry for
  the sampled token disagrees with them (rows shifted against the
  response). The worker also refuses ids outside the vocabulary. The check
  runs in torch (``score_centering/heads.py``); 64 samples of 1,024 tokens at
  ``top-k`` 128 take under a second on a CPU.
- The driver refuses a family that declares no weight and
  ``--context-parallel-size`` above 1. For a family on Slime's policy loss it
  also refuses ``--use-tis``, ``--get-mismatch-metrics``, ``--use-opsm`` and
  ``--custom-pg-loss-reducer-function-path``: Slime reweights, masks or
  re-reduces the policy-gradient term under them, beyond the declared weight.
  The critic's value loss is left alone.
- The recorded top-K dominates record size: at ``capture_topk`` 128 a
  response token carries about 3.5 KB of JSON instead of about 35 bytes. A
  smaller head shrinks records, but leaves more of the drift uncorrected
  when the sampler's tail differs from the trainer's (the paper found 32
  effective in its settings).

The step reports aggregate metrics only, as sums of per-sample means:
``score_centering_term`` (the reduced term), ``score_centering_correction_l1``
(the L1 norm of the centering coefficients),
``score_centering_sampler_head_mass``, ``score_centering_trainer_head_mass``,
``score_centering_tail_ratio`` and ``score_centering_tail_clipped`` (the
fraction of positions where a tail fell below the floor); ``loss`` includes
the term. ``tests/reef_service/test_score_centering_parity.py`` checks the
term, added to each weight's loss and to SAO's own loss, against a
full-vocabulary reference.
