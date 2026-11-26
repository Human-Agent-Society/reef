Training service: job parameters and objectives
===============================================

A weight method supplies job parameters and an objective. The training
service runs the objective once per job, then trains its loss on the
prepared data. See `harness methods <write-a-harness-method.rst>`__ for
harness evolution.

Job parameters
--------------

``WeightTrainingRecipe.default_parameters()`` returns parameters for each
sample kind. Each job contains trajectories or tasks, never both.
Backend code defines the accepted keys and values:

.. code:: python

   from reef.recipe import WeightTrainingRecipe


   class MyMethodRecipe(WeightTrainingRecipe):
       @classmethod
       def default_parameters(cls):
           return {
               "trajectory": {"objective": "sft"},
               "task": {"objective": "my_pkg.objectives:PolicyObjective"},
           }

Later sources override earlier ones:

1. Recipe defaults for the sample kind.
2. Deployment ``recipe.config.job_parameters``.
3. Training request ``parameters``.

Backend defaults fill omitted values. Return deterministic defaults:
parameters contribute to job identity and must survive retries and restarts.
``objective`` accepts a registered name or ``package.module:Objective``;
``sft`` is built in. One component can run different objectives across jobs.

Override ``configured_parameters`` to apply deployment settings to selected
sample kinds. An omitted kind requires explicit request parameters.
Parameters do not select the backend; the deployment selects it.

Parameters that Slime and Tinker take
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Meaning
   * - ``objective``
     - Recipe-defined
     - Objective name or import reference
   * - ``scheduling.epochs``
     - 1
     - Positive number of passes
   * - ``scheduling.shuffle``
     - ``false``
     - Deterministically reorder each pass
   * - ``scheduling.remainder``
     - ``partial``
     - Train, drop or reject a shorter final step: ``partial``, ``drop``, ``error``
   * - ``scheduling.packing_length``
     - ``null``
     - Pack rollouts by token count before batching
   * - ``save_interval``
     - Backend-defined
     - Optimizer steps between checkpoints; Slime task and dataset jobs count rollouts
   * - ``learning_rate_schedule``
     - Keep active schedule
     - Named optimizer-step schedule

Packing never splits a rollout. A rollout above the packing length forms
its own pack. Batch size, shuffle and remainder then apply to packs.

A learning-rate schedule declares ``name``, ``peak_learning_rate``,
``warmup_steps``, ``decay_style``, ``decay_steps``, ``min_learning_rate`` and
``initial_learning_rate``. Reusing the active schedule continues its step
count across jobs and restarts. A different schedule starts at step 0;
use a new name to restart the same curve. Optimizer state is retained.

Objective state continues only across jobs with the same objective.
See `Tinker <../user-guide/tinker.rst>`__ and
`Slime task jobs <../reference/configuration-training.rst#task-jobs-on-slime>`__
for backend-specific parameters.

Failure behavior
~~~~~~~~~~~~~~~~

Training checks each job before submission. Unsupported parameters or
objectives fail the trainer cycle without consuming samples.
Multiple epochs require ``supports_multiple_epochs=True``.

An exception after submission fails the job; its samples remain trained.
For backends that support train-from, submit the root snapshot to
``POST /reef/scenarios/{scenario}/train-from`` to restart from base weights
without training state.

The objective
-------------

Subclass ``SlimeObjective`` from ``reef.train.slime_backend.objectives`` or
``TinkerObjective`` from ``reef.train.tinker_backend.objectives``.
Each backend has its own loss family and ``TrainingSignal``.

Declare:

- ``name``: nonempty objective name.
- ``loss_family``: a registered family of the same backend.
- ``supports_multiple_epochs``: default ``False``.
  Old-policy log probabilities are computed once, before the first pass.
- ``rollout_unit``: ``sample`` by default, or ``comparison_set`` to keep
  comparison rows together.

Implement ``prepare(data, state)``. It receives the whole job's data,
so group statistics remain independent of optimizer-step partitioning.
Do not mutate the input or commit state. Return ``TrainingSignal`` with:

- ``action``: ``train`` or ``skip``.
- ``next_algorithm_state``: state saved with the final checkpoint.
  A skipped job saves no checkpoint and does not keep this state.
- ``metrics``: measurements reported with the job.
- ``advantages``: one value per trajectory, in data order, or ``None``
  when the loss does not use advantages.

A minimal Slime objective:

.. code:: python

   from reef.train.slime_backend.objectives import SlimeObjective
   from reef.train.slime_backend.schedule import next_steps
   from reef.train.slime_backend.signals import TrainingSignal
   from reef.train.types import JobData, trajectories


   class SdftObjective(SlimeObjective):
       name = "sdft"
       loss_family = "sdft"

       def prepare(self, data: JobData, state) -> TrainingSignal:
           steps = next_steps(state)
           return TrainingSignal(
               "train", {"steps": steps}, {"steps": steps, "samples": len(trajectories(data))}
           )

Keep model-dependent work in the loss family.
Objective modules must import without torch, Ray or Slime at module scope:
non-training services also resolve objectives to read their loss family.

Register an objective
~~~~~~~~~~~~~~~~~~~~~

Register by reference in your package's ``__init__.py``:

.. code:: python

   from reef.train.slime_backend.objectives import register_objective

   register_objective("my", "my_pkg.slime:MyObjective")

The reference loads on first resolution. Its module must define the objective
and register its loss family with ``register_loss_family``.
Tinker uses the equivalent registries in ``reef.train.tinker_backend``.
A package supporting both backends registers each separately.
Jobs can also use an unregistered objective's dotted reference.

Malformed references, name conflicts and unknown names raise ``ValueError``.
References that fail to import or identify no objective class raise
``PluginConfigError``. Tests can register objective and loss-family instances.

Rendering
---------

For trajectory samples, training applies the model's chat template and
computes needed log probabilities with the job's starting weights.
They are on-policy only when the sample stage enforces that condition.
Match the serving engine's
`chat template arguments <../reference/configuration-inference.rst#chat-template-arguments>`__.

Sample metadata can provide:

- ``extra.reef.teacher``: a teacher request followed by student reply tokens.
- ``extra.reef.teacher_candidates``: alternative teacher requests.
- ``extra.reef.max_tokens``: maximum trainable sample length.

A prompt that rewrites earlier turns can split a trajectory into segments.
Each keeps its group and reward, with
``extra.reef.training.segment = {"index": i, "count": n}``.
Read that metadata when counting trajectories.

Training skips unrenderable or overlength samples, overlength teacher
sequences, and teacher samples with multiple segments.
``unrenderable_samples`` counts these exclusions.
Task rollouts retain the training engine's exact tokens and log probabilities.

Checkpoints
-----------

Training validates each checkpoint before storage.
Control commits it; the activation rule decides whether it serves.
See `Admission checks <components.rst#admission-checks>`__ and
`Activation rule <recipe-control-service.rst#activation-rule>`__.
