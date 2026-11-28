Loss families
=============

A loss family supplies model-dependent computation for each optimizer step.
Slime uses ``SlimeAlgorithm``; Tinker uses ``TinkerLoss`` without worker hooks.
See `objectives <recipe-training-service.rst#the-objective>`__ for job preparation.

Family to Slime flags
---------------------

The family and ``training.options.loss-type`` must agree:

- ``sft``: ``sft_loss``.
- ``slime`` and ``sao``: ``policy_loss``.
- ``importance_sampling``, ``tttd``, ``openclawrl``, ``sdft``, ``sdpo`` and ``opd``: ``custom_loss``.

Mismatches fail startup. Only ``slime`` accepts ``use-rollout-logprobs``.
Importance sampling requires recorded ``rollout_log_probs``;
other rendered traffic normally uses actor-computed old-policy probabilities.

The spec
--------

Keep declarations in ``recipes/<name>/slime/__init__.py`` without GPU imports.
Put worker hooks in ``objective.py``; torch imports are allowed there.
Import public backend interfaces, not ``reef_adapters``.

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

``custom_loss`` replaces the loss. For only the policy-gradient primitive,
use ``policy_loss`` with ``uses_pg_loss_primitive=True``.
Class registration sends import references to workers; instance registration is process-local.

Worker hooks and data
---------------------

Register declared channels with ``@worker_hook(channel)`` in ``objective.py``.
Missing hooks stop the worker without falling back.
A custom loss receives ``args, batch, logits, sum_of_sample_mean``;
a custom policy-gradient primitive receives ``args, ppo_kl, log_probs, advantages``.

Extra wire columns need ``rollout_data_keys`` and optional tensor dtypes.
Declare response-aligned tensors for context parallelism and ``external_batch_keys``
for microbatches. See `SlimeAlgorithm <../../reef/train/slime_backend/algorithm.py>`__
for hook signatures, wire fields and actor/critic orchestration.

Jobs can switch actor families, but startup flags remain fixed.
New family flags, critic changes, a second distillation family or conflicting dtypes are rejected.

Distillation
------------

Extend ``DistillAlgorithm`` and forward loss/pre-train hooks to
``reef.train.slime_backend.distill.objective``.
Rows add teacher prompt tokens followed by student response tokens, and sample weight.
For importance sampling, distillation uses recorded sampler log probabilities if every row supplies them.
Otherwise it uses the actor’s job-start probabilities for the batch. Combining importance sampling
with score centering requires recorded sampler log probabilities for every row.

Teachers can follow student weights, retain a copy, load a checkpoint or use
a compatible frozen engine. Teacher scoring precedes training; failure rejects the batch.
Engine teachers support sampled reverse KL and teacher-selected top-K KL/JSD
at temperature 1. Full-vocabulary or student-selected top-K requires checkpoint teachers.
See the `distillation contract <../../reef/train/slime_backend/distill/algorithm.py>`__
and `OPD teacher setup <../user-guide/recipes/opd.rst#teacher-engine>`__.

Score centering
---------------

Centering is optional and corrects sampler/trainer probability mismatch.
Enable ``training.options.score-centering`` and set ``score-centering-top-k``.
Native ``slime`` task jobs use ``use-score-centering`` instead.

Declare ``policy_gradient_weight(args)`` for losses of the form
``-A_t * sg[f(p_t / q_t)] * log p_t``. Here ``p`` and ``q`` are trainer and sampler probabilities;
``A`` is advantage and ``sg`` stops gradients:

- ``none``: ``f(r)=1``.
- ``truncated``: ``f(r)=min(r, upper)``.
- ``masked``: ``f(r)=r`` strictly inside ``(lower, upper)``, otherwise zero.

Returning ``None`` refuses centering. Samples need sampler ``topk_indices``
and ``topk_log_probs`` under ``extra.reef.training`` at every trained position.
Teacher probabilities cannot substitute for sampler probabilities.
Traffic records lack these tensors. Invalid rows fail validation.

Slime requires context parallel size 1 and compatible policy-loss options.
Distillation requires sampled reverse KL, positive teacher top-K, renormalization
and token-level weights. See the `centering implementation
<../../reef/train/slime_backend/score_centering>`__ for supported combinations and metrics.
