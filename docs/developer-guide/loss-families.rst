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

+----------------+-----------------------------+----------------------------+
| Loss family    | ``--loss-type``             | Rollout log-probs          |
+================+=============================+============================+
| ``sao``        | ``policy_loss``             | ``--use-rollout-logprobs`` |
+----------------+-----------------------------+----------------------------+
| ``tttd``       | ``custom_loss``             | ``--use-rollout-logprobs`` |
+----------------+-----------------------------+----------------------------+
| ``openclawrl`` | ``custom_loss``             | not required               |
+----------------+-----------------------------+----------------------------+
| ``sdft``       | ``custom_loss``             | ``--use-rollout-logprobs`` |
+----------------+-----------------------------+----------------------------+

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
  parallel themselves and write the gradients out where autograd over one
  shard would drop the coupling through the global log-sum-exp;
  ``tests/reef_service/test_distill_parity.py`` pins them to a pure-Python
  reference and to the dense gradients across four ranks.

The base registers no family and imports nothing from ``reef_adapters``;
``recipes/openclawrl/slime/`` imports its packing schedule and its sharded
gathers from it.
