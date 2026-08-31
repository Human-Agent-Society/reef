Loss families
=============

A loss family is the tensor objective a weight-training recipe runs in the Slime
backend. The step preparer decides what the signal is (advantages, which
family); the loss family decides how the backend turns that into a loss.

They live in two places, and the names are easy to confuse:

- ``reef/train/algos/`` holds step preparers. Backend-neutral, no torch.
- ``reef/train/slime_backend/`` holds the machinery (``algorithm.py``,
  ``loss_families.py``, ``data_builder.py``); every family lives in its
  method package, ``recipes/<name>/slime/``. The spec is torch-free driver
  code; the objective is worker-side torch code.

A recipe names its family through ``WeightTrainingSpec.loss_family``. The
preparer's ``StepSignal.loss_family`` must carry the same string; the bridge
rejects a payload whose ``loss`` differs from the family it booted with.

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
| ``opd``        | ``policy_loss``             | ``--use-rollout-logprobs`` |
+----------------+-----------------------------+----------------------------+
| ``sao``        | ``policy_loss``             | ``--use-rollout-logprobs`` |
+----------------+-----------------------------+----------------------------+
| ``tttd``       | ``custom_loss``             | ``--use-rollout-logprobs`` |
+----------------+-----------------------------+----------------------------+
| ``openclawrl`` | ``custom_loss``             | not required               |
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
- ``provenance_metrics``: telemetry after the step.

Two loss lanes
--------------

``loss_type = "custom_loss"`` replaces Slime's loss with the
``custom_loss_function_path`` hook (``tttd``, ``openclawrl``).
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
frozen Megatron teacher).

See also
--------

- `Python API <../reference/python-api.rst#step-preparer>`__: the signal side of the pair.
- `Write a recipe <write-a-recipe.rst>`__: binding a family to a method.
