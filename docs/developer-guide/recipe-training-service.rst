Training service: job parameters and objectives
===============================================

The training service runs each training job on the deployment's training
component. A weight method plugs in there: the recipe gives each job its
parameters, and the parameters name the method's objective. This page defines
the job parameters, the objective contract, and how the component renders
samples.

:For: authors of a weight method
:Before you start: the plug-in map in `Write a recipe <write-a-recipe.rst>`__
:You will have: the job parameters of your recipe, a registered objective, and the failure behavior of a job

Only the training service builds the training component. Control reads the
backend's facts and sends each job through the jobs queue. So a method's
training code runs in the training service alone:

- The objective's ``prepare`` runs in the training component's process,
  once per job, on the job's full data.
- A Slime loss family's hooks run in Slime's Megatron workers, in each
  optimizer step (see `Loss families <loss-families.rst>`__).
- A Tinker loss runs in the training component's process.
- A harness ``Method`` runs in the ``harness`` training component's process,
  once per job (see `Write a harness method
  <write-a-harness-method.rst>`__).

Job parameters
--------------

Reef submits one job each time job data is ready. A job holds samples of one
kind: trajectories or tasks. The job's parameters are opaque to Reef. Each
training backend owns their names and shapes.

``WeightTrainingRecipe.default_parameters()`` returns the recipe's
parameters by sample kind. Normally it names the method as ``objective``:

.. code:: python

   from reef.recipe import WeightTrainingRecipe


   class MyMethodRecipe(WeightTrainingRecipe):
       @classmethod
       def default_parameters(cls):
           return {
               "trajectory": {"objective": "sft"},
               "task": {
                   "objective": "my_pkg.objectives:PolicyObjective",
                   "learning_rate_schedule": {"name": "policy", "peak_learning_rate": 1e-6, "warmup_steps": 10},
               },
           }

The parameters of a job are merged in this order; a later source wins:

1. ``default_parameters()`` for the kind of the job's samples;
2. ``recipe.config.job_parameters`` of the deployment, for every kind;
3. the ``parameters`` of the job's training request, when it has one.

The component's own defaults fill the rest, such as its learning rate.

Rules:

- ``default_parameters()`` must return the same value at every call. The
  parameters are part of the job's identity, and a retry after a restart
  computes the job again.
- ``objective`` is a registered name or a dotted
  ``package.module:Objective``. Use a dotted reference for an objective of
  another package. ``sft`` is built in on each backend.
- A recipe can give each kind its own objective. The same component
  switches the objective from one job to the next.
- Override ``configured_parameters`` when ``job_parameters`` must reach only
  some kinds. A kind that its result leaves out trains only for a request
  that gives its parameters.
- The job's backend is always the deployment's training component. The
  parameters do not name it.

Parameters that Slime and Tinker take
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``objective``: the method, as an objective name or reference.
- ``scheduling``: how the component trains the job's data.

  - ``epochs`` (default 1): the passes over the job's data, a positive int.
  - ``shuffle`` (default ``false``): reorder the rollouts in every pass,
    deterministically from the data's rows.
  - ``remainder`` (default ``"partial"``): what to do with a last step
    that has fewer rollouts, or packs, than the batch size. ``"partial"``
    trains it as a smaller step, ``"drop"`` leaves it out, and ``"error"``
    refuses the job's data.
  - ``packing_length`` (default ``null``): pack the rollouts into packs of
    at most this many tokens before the component cuts the steps. One step
    then trains batch-size packs. A pack never splits a rollout, and a
    longer rollout is a pack of its own. ``shuffle`` and ``remainder``
    apply to the packs.
- ``save_interval``: save a checkpoint after each ``save_interval``
  optimizer steps of the job.
- ``learning_rate_schedule``: a ``LearningRateSchedule``, with ``name``,
  ``peak_learning_rate``, ``warmup_steps``, ``decay_style``,
  ``decay_steps``, ``min_learning_rate`` and ``initial_learning_rate``.

A ``LearningRateSchedule`` counts optimizer steps:

- A job that selects the active schedule continues it, also on a retry or
  after a restart.
- A job that selects another schedule starts that schedule at step 0. To
  start the same curve again, give it another ``name``.
- A job that selects no schedule keeps the active one.
- Starting a schedule does not reset the optimizer state.

A job continues the objective state of the previous job only when it trains
the same objective. A job of another objective starts with an empty state.

`Train with Tinker <../user-guide/tinker.rst>`__ and `Task jobs on Slime
<../reference/configuration-training.rst#task-jobs-on-slime>`__ list the other
parameters of each backend.

Failure behavior
~~~~~~~~~~~~~~~~

- Control asks the training service to check each job before it queues the
  job. The component refuses a parameter that it does not take and a method
  that it cannot train. Control does not submit a refused job, and the job
  consumes no sample. The trainer's cycle fails with the component's
  message.
- The component refuses ``scheduling.epochs`` above 1 for an objective that
  does not declare ``supports_multiple_epochs``.
- An exception during training ends the job as failed. The samples of a
  failed job count as trained: they do not train again.
- To train again from the base weights with no training state, call
  ``POST /reef/scenarios/{scenario}/train-from`` with the scenario's root
  snapshot.

The objective
-------------

An objective owns a method's data preparation and names its loss family.
Each backend has its own base class, table and ``TrainingSignal``; the two
backends share no code.

- Slime: subclass ``reef.train.slime_backend.objectives.SlimeObjective``.
  The loss family is a ``SlimeAlgorithm`` (`Loss families
  <loss-families.rst>`__).
- Tinker: subclass ``reef.train.tinker_backend.objectives.TinkerObjective``.
  The loss family is a ``TinkerLoss`` (``reef.train.tinker_backend.losses``).

Both base classes have the same members:

- ``name`` (required): the non-empty name that a job's ``objective`` gives.
- ``loss_family`` (required): the name of a registered loss family of the
  same backend.
- ``supports_multiple_epochs`` (default ``False``): whether the loss stays
  valid when a job trains its data more than once. The component computes
  old-policy log probabilities once, before the first pass.
- ``rollout_unit`` (default ``"sample"``): the rows that one optimizer step
  keeps together, ``"sample"`` or ``"comparison_set"``.
- ``prepare(data, state)`` (required): return the backend's
  ``TrainingSignal`` for the job's full data. Do not mutate ``data`` and do
  not commit state.

A ``TrainingSignal`` holds:

- ``action``: ``"train"`` or ``"skip"``. A skipped job trains nothing.
- ``next_algorithm_state``: the objective's state after this job. The
  component commits it with the job's last checkpoint.
- ``metrics``: the objective's telemetry.
- ``advantages``: one value per trajectory, or ``None`` for a loss family
  that reads none.

``prepare`` sees the complete job data, so statistics over a group do not
depend on how the backend partitions the data. Everything that depends on
the model stays in the loss family.

SDFT's objective is the minimal case:

.. code:: python

   from reef.train.slime_backend.objectives import SlimeObjective
   from reef.train.slime_backend.schedule import next_steps
   from reef.train.slime_backend.signals import TrainingSignal
   from reef.train.types import JobData, trajectories


   class SdftObjective(SlimeObjective):
       name = "sdft"
       loss_family = "sdft"
       supports_multiple_epochs = False

       def prepare(self, data: JobData, state) -> TrainingSignal:
           samples = trajectories(data)
           steps = next_steps(state)
           return TrainingSignal("train", {"steps": steps}, {"steps": steps, "samples": len(samples)})

The module that defines an objective must import no torch, Ray or Slime at
module scope. On Slime, every service of the deployment resolves the
recipe's objective to read its loss family, so each service imports that
module. Only the training service calls ``prepare``.

Register an objective
~~~~~~~~~~~~~~~~~~~~~

A method makes one registration call for each backend that it supports, in
its package ``__init__``. The call names the objective by reference and
imports nothing:

.. code:: python

   # my_pkg/__init__.py
   from reef.train.slime_backend.objectives import register_objective

   register_objective("my", "my_pkg.slime:MyObjective")

The first ``resolve_objective("my")`` imports ``my_pkg.slime``. That module
defines the objective and, under ``@register_loss_family``, the loss family
that the objective names:

.. code:: python

   # my_pkg/slime/__init__.py
   from reef.train.slime_backend.algorithm import SlimeAlgorithm
   from reef.train.slime_backend.loss_families import register_loss_family
   from reef.train.slime_backend.objectives import SlimeObjective


   @register_loss_family
   class MyAlgorithm(SlimeAlgorithm):
       loss_family = "my"
       ...


   class MyObjective(SlimeObjective):
       name = "my"
       loss_family = "my"
       ...

Tinker registers the same way, with ``register_objective`` of
``reef.train.tinker_backend.objectives``, ``register_loss_family`` of
``reef.train.tinker_backend.loss_families``, and the base classes
``TinkerObjective`` and ``TinkerLoss``.

- A package that supports both backends imports each ``register_objective``
  under its own name (see ``recipes/tttd/__init__.py``). A method can compute
  its advantages in one function that names no backend, and call it from
  both objectives.
- A job can also name an unregistered objective by its dotted reference.
- A test can register an instance: ``register_objective(objective.name,
  objective)`` and ``register_loss_family(family)``.
- ``register_objective`` raises ``ValueError`` for a malformed reference and
  for a name that holds another objective. ``resolve_objective`` raises
  ``ValueError`` for an unknown name, and ``PluginConfigError`` for a
  reference that does not import or names no objective class.

Rendering
---------

A sample holds an ATIF trajectory, not tokens. The training component
renders each sample with the model's chat template
(``reef.core.chat_rendering``). It computes the log probabilities that the
objective needs with the weights of the job's start. They are exact for an
on-policy stage and an approximation for a stage that accepts a lag.

A record-to-sample stage can add what the component renders with:

- ``extra.reef.teacher``: the teacher request of a distillation sample. The
  component renders the teacher sequence: the teacher prompt, then the
  student's reply tokens.
- ``extra.reef.teacher_candidates``: one teacher request for each hint, as
  OpenClaw-RL writes them.
- ``extra.reef.max_tokens``: the longest sample that may train, as SFT
  writes it.

Behavior that an objective must handle:

- **Segments.** When a call's prompt does not extend the tokens of the call
  before it, the trajectory trains as several segments. Each segment is one
  sample with ``extra.reef.training.segment``: ``{"index": i, "count": n}``.
  The segments keep the trajectory's group and reward. An objective that
  counts trajectories must read ``segment``.
- **Left-out samples.** The component leaves out a sample that cannot
  render, that is longer than its ``max_tokens``, whose teacher sequence is
  longer than the request's ``max_tokens``, or that has a teacher and
  several segments. The job metric ``unrenderable_samples`` counts them.
- **Task rollouts.** A task rollout of the training component already holds
  the exact tokens and log probabilities of its own engine. The component
  trains them as they are.

The component must render as the serving engine did. Give it the engine's
chat template arguments (see `Chat template arguments
<../reference/configuration-inference.rst#chat-template-arguments>`__).

Checkpoints
-----------

The training service checks each checkpoint of a job with the component's
admission check before it stores the checkpoint (see `Admission checks
<components.rst#admission-checks>`__). Control commits the stored checkpoint
as an artifact of the scenario, and the `activation rule
<recipe-control-service.rst#activation-rule>`__ decides whether it serves.
