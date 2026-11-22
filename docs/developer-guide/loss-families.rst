Loss families
=============

A loss family implements the model-dependent part of a weight method for one
training backend. The method's objective names it in ``loss_family``. The
objective prepares the job's data once; the family's hooks run in each
optimizer step, after the backend's forward passes. This page defines the
Slime loss family contract and the shared implementations to build on.

:For: authors of a weight method's loss on Slime
:Before you start: the objective and its registration from `Training service <recipe-training-service.rst>`__
:You will have: a loss family spec, its worker hooks, and the Slime flags that it needs

The family runs in the training service. Its spec runs in the training
component's process, and its hooks run in Slime's Megatron workers. The Slime
backend reads the objective of the recipe's parameters when it connects
(trajectory kind first, then task kind), and the workers start with that
objective's family. A Tinker method supplies a ``TinkerLoss``
(``reef.train.tinker_backend.losses``) instead; Tinker trains in the
component's process and has no worker hooks.

Layout
------

.. code:: text

   recipes/<name>/slime/
     __init__.py     the objective and the spec: a SlimeAlgorithm subclass, no torch;
                     a custom wire row's shaper and payload builder live here too
     objective.py    the @worker_hook hooks, torch

A Slime module of a method imports from ``reef.train.slime_backend`` only.
``@register_loss_family`` records the reference of the class
(``my_pkg.slime:MyAlgorithm``). The Slime component sends that reference to
each Megatron worker, which starts without the component's registrations and
imports the family from it. A family that registers as an instance has no
reference, so only that process knows it.

Family to Slime flags
---------------------

The objective's ``loss_family`` and the Slime flags in ``training.options``
must describe the same loss. The Slime component checks them at start and
refuses a mismatch.

+-----------------------------+-------------------+
| Loss family                 | ``--loss-type``   |
+=============================+===================+
| ``sft``                     | ``sft_loss``      |
+-----------------------------+-------------------+
| ``slime``                   | ``policy_loss``   |
+-----------------------------+-------------------+
| ``sao``                     | ``policy_loss``   |
+-----------------------------+-------------------+
| ``importance_sampling``     | ``custom_loss``   |
+-----------------------------+-------------------+
| ``tttd``                    | ``custom_loss``   |
+-----------------------------+-------------------+
| ``openclawrl``              | ``custom_loss``   |
+-----------------------------+-------------------+
| ``sdft``, ``sdpo``, ``opd`` | ``custom_loss``   |
+-----------------------------+-------------------+

Every family except ``slime`` refuses ``--use-rollout-logprobs``. Serving
returns no log probabilities, so the actor computes the old-policy log
probabilities of each row with the weights of the job's start.
``importance_sampling`` reads recorded rollout log probabilities
(``rollout_log_probs``) instead, and refuses a row without them. No built-in
job data records them on Slime, so a method that selects this family must
provide such samples.

The spec
--------

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
           if getattr(args, "kl_coef", 0) <= 0:
               raise RuntimeError(f"{source} requires --kl-coef > 0")

``loss_family``, ``loss_type`` and ``validate_specific_args`` are required.
The other members have defaults. In the order that a job reaches them:

- ``parse_specific_options`` and ``apply_driver_options``: the family's
  flags (``--<name>-*``) in the Slime arguments. Reef strips them before
  Slime's parser sees them and stamps them on ``args``.
- ``configure_backend_args``: derive backend settings once ``loss_family``
  is stamped.
- ``shape_sample_row`` and ``build_rollout_data``: a custom wire row. The
  default row is ``[source_id, tokens, loss_mask, rollout_log_probs,
  reward]``. The component fills ``tokens`` and ``loss_mask`` when it
  renders the sample. ``rollout_log_probs`` is empty for a rendered sample;
  only a task rollout of the component fills it.
- ``prepare_rollout``: work in the component's process before the workers
  train a job.
- ``bind``: a per-run instance that carries state, such as a critic
  schedule.
- ``train``: critic and actor orchestration. The default is one actor step.
- ``rollout_metrics``: rollout version and timing metrics after the step.
- ``policy_gradient_weight``: the weight that the loss puts on the sampled
  token's score (see `Score centering`_).

A family selects one of two loss lanes:

- ``loss_type = "custom_loss"`` replaces Slime's loss with the
  ``custom_loss_function_path`` hook (``tttd``, ``openclawrl``, ``sdft``).
- ``uses_pg_loss_primitive = True`` keeps Slime's ``policy_loss`` and swaps
  only the per-token primitive through ``custom_pg_loss_function_path``
  (``sao``).

Objective hooks
---------------

``objective.py`` registers each channel that the spec lists in
``required_objective_hooks``, with ``@worker_hook("<channel>")``
(``reef.train.slime_backend.algorithm``):

- ``custom_loss_function_path``:
  ``<name>_loss(args, batch, logits, sum_of_sample_mean)``
- ``custom_pg_loss_function_path``:
  ``<name>_loss(args, ppo_kl, log_probs, advantages)``
- ``custom_advantage_function_path``:
  ``<name>_advantages(args, rollout_data)``
- ``reef_actor_init_hook_path``: ``<name>_actor_init(actor)``, once, after
  the actor loaded its weights
- ``reef_actor_pre_train_hook_path``:
  ``<name>_actor_pre_train(actor, rollout_data)``, before every training step

The worker imports the module at init and checks that every declared channel
is present. A missing module or hook stops the worker. Nothing falls back to
Slime's default loss.

Rules that the contract tests enforce: a family package never imports
``reef_adapters``; the spec module imports no torch; a hook carries the
family's name; a family flag carries the ``--<name>-`` prefix.

Wire declarations
-----------------

A family that ships more than the five policy columns declares them on the
spec:

- ``rollout_data_keys``: per-sample payload keys that the rollout manager
  partitions across data-parallel ranks.
- ``rollout_tensor_dtypes``: which of those keys become tensors, and of
  which dtype (``"int"``, ``"long"``, ``"float32"``). Ragged fields stay
  undeclared and pass through as lists.
- ``response_aligned_keys``: tensors with one entry per response token. The
  worker slices them for context parallelism as it slices advantages.
- ``external_batch_keys``: keys that the worker forwards into the
  microbatch.
- ``rollout_log_skip_keys``: non-scalar keys that Slime's numeric rollout
  logger skips.
- ``critic_value_head_zero_init`` and ``critic_value_mask_key``: critic
  families only.

Families to read:

- ``reef/train/slime_backend/sft.py``: the built-in ``sft`` family, with
  Slime's ``sft_loss`` and no hooks.
- ``reef/train/slime_backend/native.py``: the built-in ``slime`` family of
  dataset jobs and task jobs, whose advantages and loss are Slime's (see
  `Dataset jobs on Slime
  <../reference/configuration-training.rst#dataset-jobs-on-slime>`__).
- ``recipes/tttd/slime/``: two hooks and the default row.
- ``recipes/sao/slime/``: a critic schedule and the pg-primitive lane.
- ``recipes/openclawrl/slime/``: a custom row, both actor lifecycle hooks
  and a frozen Megatron teacher.
- ``recipes/sdft/slime/`` and ``recipes/sdpo/slime/``: thin families on
  `the distillation base`_.

Per-job families
----------------

A job can select an objective of another family (see `Job parameters
<recipe-training-service.rst#job-parameters>`__). The workers keep their
startup flags. The family brings its ``--loss-type``, objective hooks,
advantage routing, wire keys and the defaults of its family flags. Before
the family's first job trains, the Slime component validates the family
against the startup flags. Only the actor workers switch; the critic keeps
the startup family. A ``self`` distillation teacher copies the actor's
weights at the family's first job, and Reef does not checkpoint that copy.

The Slime component fails the job for a family that:

- needs its own family flags, which only the startup family parses;
- configures the critic, as SAO does;
- is a second distillation family;
- declares a wire key with a dtype that another family of the run declares
  differently;
- fails its own validation against the startup flags.

A learning-rate schedule sets the rate of each actor optimizer step through
Slime's before-train-step hook. Reef runs any
``--custom-megatron-before-train-step-hook-path`` first. The component
records the schedule's progress with each saved checkpoint, so the next job
continues it. A job that selects a schedule fails when the run uses
``--decoupled-lr``.

The distillation base
---------------------

SDFT, SDPO and on-policy distillation distil a teacher on the student's own
samples. They compute the same per-token divergence and differ in who the
teacher is and which divergence they minimize. Both are settings of one
implementation, ``reef/train/slime_backend/distill/``. Each recipe's family
is a thin subclass of ``DistillAlgorithm``: it names itself, sets its
defaults in a ``DistillAlgorithmSettings`` subclass, and its
``objective.py`` forwards ``<name>_loss`` and ``<name>_actor_pre_train`` to
``distill.objective``.

**Wire row.** Seven columns: the policy row, ``teacher_tokens`` and
``sample_weight``. ``teacher_tokens`` holds the teacher's prompt IDs followed
by the student's response IDs; the component renders them from the sample's
teacher request. ``sample_weight`` is the factor on the sample's mean
divergence: 1, unless the recipe's stages set ``distill_sample_weight`` on
the sample.

**Flags.** Each family takes these flags under its own prefix
(``--<name>-``): ``teacher``, ``divergence``, ``top-k``, ``top-k-source``,
``top-k-distribution``, ``teacher-update-rate``, ``teacher-checkpoint``,
``importance-sampling-cap``, ``importance-sampling-level``,
``skip-response-tokens`` and ``jsd-beta``. Reef stamps them on ``args`` under
``distill_*`` names, which the worker hooks read whatever the prefix was.

**Teacher.**

- ``self``: the student's own weights read the privileged prefix. With
  ``teacher-update-rate`` 1 the teacher is the current weights, below 1 a
  slow-moving copy, and at 0 a frozen snapshot.
- ``separate``: another checkpoint that fits the actor's model, loaded
  beside the actor's weights.

The pre-train hook switches the teacher's weights in, runs one forward-only
pass over the teacher sequences of the job's data, and switches the actor
back.

**Divergence.** The forward KL, the reverse KL or the generalized JSD:

- over the teacher's whole distribution (``top-k`` 0); or
- over K IDs per position: the teacher's own top-K, or the current
  student's, which needs one more forward before the step and zero dropout.
  The two distributions are either renormalized over the K IDs, with the
  reverse KL then estimated at the sampled token, or compared with one
  bucket for the rest of the vocabulary, as SDPO's reference does.

The base registers no family. The operations over the vocab shards of tensor
parallel (log-sum-exp, the log probabilities at IDs on any shard, the top-K
IDs) are in ``reef/train/slime_backend/vocab_parallel.py``. The distillation
base, score centering and OpenClaw-RL's teacher share them.

Teacher engine
--------------

A Slime distillation family can score its ``separate`` teacher on a frozen
SGLang engine instead of a checkpoint in the actor. The teacher may then
differ from the student in size and parallel layout, as long as it uses the
student's token IDs. The wire row, the loss and the report schema do not
change. `OPD <../user-guide/recipes/opd.rst#teacher-engine>`__ shows the
deployment's ``teacher`` section.

What a family author relies on:

- With a ``teacher`` section, ``reef serve`` starts the engine as the
  service ``distill-teacher`` and adds ``--<family>-teacher-url``,
  ``--<family>-teacher-model-path`` and ``--<family>-teacher-timeout`` to
  the Slime arguments. Without the section, ``training.options`` can set
  these flags for an engine that you run yourself.
- The Slime component scores each sample's rendered teacher sequence on the
  engine before the batch reaches the workers. A scoring failure fails the
  job and leaves no partially scored batch.
- Reef checks the engine's tokenizer against the student's vocabulary at
  startup. The chat templates may differ: the component renders the teacher
  sequence with the student's chat template.
- The engine supports the sampled reverse KL and the teacher-selected top-K
  forward, reverse and JSD objectives, at rollout temperature 1. The
  full-vocabulary and student-selected top-K representations need the
  checkpoint teacher. An unsupported combination fails validation.
- ``perf/distill_teacher_time`` is the scoring wall time in seconds.

Score centering
---------------

Score centering is a correction that Reef adds to a family's own
policy-gradient loss; it is not a loss family. It implements `Score
Centering Stabilizes Off-policy Reinforcement Learning
<https://arxiv.org/abs/2609.20807>`_ (Appendix A, equations 9-14). When the
rollout engine's distribution ``q`` differs from the trainer's ``p``, for
example with a quantized engine or stale weights, a policy gradient drifts
toward ``q``. Score centering subtracts that drift.

It is off unless ``score-centering`` is set:

.. code:: yaml

   training:
     options:
       score-centering: true
       score-centering-top-k: 128

+-----------------------------------+---------+-------------------------------------+
| Flag                              | Default | Meaning                             |
+===================================+=========+=====================================+
| ``score-centering``               | off     | add the term to the family's loss   |
+-----------------------------------+---------+-------------------------------------+
| ``score-centering-top-k``         | 128     | sampler log-probs read per position |
+-----------------------------------+---------+-------------------------------------+
| ``score-centering-min-tail-mass`` | 1e-6    | floor applied to both tail masses   |
+-----------------------------------+---------+-------------------------------------+

Declare support
~~~~~~~~~~~~~~~

Only a family that declares the weight that its loss puts on the sampled
token's score accepts score centering. SAO, ``importance_sampling`` and the
distillation base's sampled reverse-KL mode support it. The loss must have
the form ``-A_t * sg[f(p_t / q_t)] * log p_t``, with ``q`` the rollout
engine's probability:

.. code:: python

   def policy_gradient_weight(self, args):
       # SAO: the ratio masked to its trust region.
       return PolicyGradientWeight("masked", lower=1 - args.eps_clip, upper=1 + args.eps_clip_high)

   def policy_gradient_weight(self, args):
       # importance_sampling: the ratio truncated at Slime's --tis-clip.
       return PolicyGradientWeight("truncated", upper=args.tis_clip)

``PolicyGradientWeight`` (``reef.train.slime_backend.algorithm``) has three
kinds: ``none`` (``f = 1``), ``truncated`` (``min(r, upper)``) and ``masked``
(``r`` strictly inside ``(lower, upper)``, else 0). The default, ``None``,
refuses score centering.

Reef adds this term to the family's loss at every trained response position:

.. code:: text

   A * sum_{v in H} sg[q_v * f(p_v / q_v) - alpha * p_v] * log p_v

   rho   = max(1 - q(H), eps) / max(1 - p(H), eps)
   alpha = rho * f(1 / rho)

``H`` is the sampler's recorded top-K IDs. The sampler's tail is approximated
as ``rho * p``, and ``sg`` stops the gradient. The term uses the loss's own
advantages and the same per-sample mean. It is zero when ``q = p``.

Requirements
~~~~~~~~~~~~

- **Samples.** Each trained sample must carry the top-K that its sampler
  recorded, at least ``score-centering-top-k`` entries per trained position,
  in ``extra.reef.training`` (``topk_indices`` and ``topk_log_probs``).
  Serving records no tensors, so a sample that serving recorded has none,
  and the component fails the job. The Spindle full-parameter component's
  task rollouts record them when its ``score_centering`` option is on.
- **Task jobs on Slime.** A task job trains the ``slime`` family, which
  refuses this term. It uses Slime's own ``--use-score-centering`` instead
  (see `Task jobs on Slime
  <../reference/configuration-training.rst#task-jobs-on-slime>`__).
- **Checks.** Before each job trains, the component keeps the first
  ``top-k`` entries of every trained position. It refuses a sample with
  missing or short rows, duplicate or negative IDs, non-finite log
  probabilities, a head mass above one, or a head whose entry for the
  sampled token disagrees with ``rollout_log_probs``.
- **Flags.** The Slime component refuses a family that declares no weight,
  and ``--context-parallel-size`` above 1. For a family on Slime's policy
  loss it also refuses ``--use-tis``, ``--get-mismatch-metrics``,
  ``--use-opsm`` and ``--custom-pg-loss-reducer-function-path``.
- **Distribution.** The recorded log probabilities are taken after
  temperature and before the top-k, top-p and min-p filters. With those
  filters on, the correction centers against the unfiltered distribution.
- **Size.** At a head of 128, a response token carries about 3.5 KB of JSON
  instead of about 35 bytes.

Sampled on-policy distillation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a family that uses ``DistillAlgorithm``, select the ``reverse``
divergence, a positive teacher ``top-k`` and the ``renormalized`` top-K
distribution. The reverse KL is then estimated at the sampled token, and its
advantage is the detached ``log teacher(y) - log student(y)``. For example,
add these options to an SDFT configuration with the score-centering options
above:

.. code:: yaml

   training:
     options:
       sdft-divergence: reverse
       sdft-top-k: 128
       sdft-top-k-distribution: renormalized
       sdft-importance-sampling-level: token
       sdft-importance-sampling-cap: 2.0

Use the ``sdpo-`` prefix for SDPO. Centering accepts token-level truncated
importance sampling, or no importance weighting
(``importance-sampling-cap: 0``). It refuses sequence-level importance
weights, full-vocabulary KL, top-K-plus-tail divergences, forward KL and JSD.
OpenClaw-RL's separate top-K OPD surrogate is not supported.

``score-centering-top-k`` sets the correction's head, and the family's
``top-k`` sets the teacher pass. They need not match. Teacher top-K log
probabilities cannot replace the sampler's recorded head.

Metrics
~~~~~~~

Each step reports these aggregates, as sums of per-sample means:

- ``score_centering_term``: the reduced term; ``loss`` includes it;
- ``score_centering_correction_l1``: the L1 norm of the centering
  coefficients;
- ``score_centering_sampler_head_mass`` and
  ``score_centering_trainer_head_mass``;
- ``score_centering_tail_ratio``;
- ``score_centering_tail_clipped``: the fraction of positions where a tail
  fell below the floor.
