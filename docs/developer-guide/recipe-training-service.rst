Training service: job parameters and objectives
===============================================

A weight objective prepares each job's data before loss computation.
Harness evolution uses `methods <write-a-harness-method.rst>`__ instead.

Job parameters
--------------

``WeightTrainingRecipe.default_parameters()`` maps sample kinds to parameters:

.. code:: python

   from reef.recipe import WeightTrainingRecipe


   class MyMethodRecipe(WeightTrainingRecipe):
       @classmethod
       def default_parameters(cls):
           return {
               "trajectory": {"objective": "sft"},
               "task": {"objective": "my_pkg.objectives:PolicyObjective"},
           }

Request parameters override ``recipe.config.job_parameters``, which override
recipe defaults. Backend defaults fill omissions.
Defaults must be deterministic because they contribute to job identity.
Each job has one sample kind; missing defaults require explicit request parameters.

Parameters that Slime and Tinker take
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``objective``: registered name or class reference.
- ``scheduling``: epochs (default 1), shuffle (false), remainder (``partial``)
  and optional packing length. Packing never splits a rollout.
- ``save_interval``: optimizer steps between checkpoints.
  Slime task and dataset jobs count rollout iterations instead.
- ``learning_rate_schedule``: named schedule. Reusing it continues its step count;
  changing it resets the count while retaining optimizer state.

See `Tinker <../user-guide/tinker.rst>`__ and
`Slime <../reference/configuration-training.rst>`__ for backend constraints.
Unsupported jobs fail before submission without consuming samples;
after submission, failure leaves samples consumed.

The objective
-------------

Subclass ``SlimeObjective`` or ``TinkerObjective`` from the backend's ``objectives`` module.
Declare ``name`` and a registered ``loss_family``.
Implement ``prepare(data, state)`` without mutating inputs or committing state:

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

``TrainingSignal`` returns ``action`` (train or skip), metrics, next state and
optional advantages in trajectory order. Only the final checkpoint stores next state;
skipped jobs save neither checkpoints nor state.

The objective sees the whole job before optimizer partitioning.
Multiple epochs require ``supports_multiple_epochs=True``; old-policy probabilities
are computed once. ``rollout_unit="comparison_set"`` keeps comparison rows together.

Keep torch, Ray and Slime imports out of objective module scope.
Register by reference with the backend's ``register_objective(name, reference)``;
the referenced module also registers its loss family.
Bad names raise ``ValueError``; import/class errors raise ``PluginConfigError``.

Rendering and checkpoints
-------------------------

Training uses the model's chat template and starting weights.
Match `serving template arguments <../reference/configuration-inference.rst#chat-template-arguments>`__.
Unrenderable or overlength inputs and multi-segment teacher samples are skipped.
``unrenderable_samples`` counts exclusions. Task rollouts retain exact engine tokens.

Checkpoints pass `component validation <components.rst#admission-checks>`__ before storage.
Control commits them; activation separately decides whether they serve.
