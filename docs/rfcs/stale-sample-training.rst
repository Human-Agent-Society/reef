Bounded stale-sample training
=============================

:Status: Deprecated

.. warning::

   This historical RFC predates the issue-based RFC process and will be removed
   in a future cleanup.

   Reef used to fence each training job to the exact serving version that
   produced its rollouts, and it discarded all older samples. This RFC adds
   one bounded staleness window, the recipe setting ``max_staleness``
   (default ``0``). An earlier version of this RFC checked the window in the
   training runtime. The current design checks it only in the record-to-record
   stage, before a batch forms. Section 3 records that change.

.. _1-the-problem:

1. The problem
--------------

Under the exact fence, a rollout could train only while the weights that
produced it still served. Each commit made every in-flight rollout
ineligible. The fence wasted in-flight work and made asynchronous rollout
generation lockstep.

The training path already owns how a sample is consumed. The staleness check
should only bound which producing versions reach that path. It should not
inspect the recipe or objective, change the payload, or add an importance
weight.

.. _2-decision:

2. Decision: the record-to-record stage owns the staleness check
----------------------------------------------------------------

``max_staleness`` is a field of ``WeightTrainingRecipe``
(`reef/recipe/base.py <../../reef/recipe/base.py>`__, environment fallback
``REEF_MAX_STALENESS``). The recipe gives it to the pipeline as
``PipelineContext.max_staleness``. ``0`` keeps exact-version training.

The check has three parts:

1. **Served versions.** At each batch reservation, the scenario gives the
   version that it serves now (``Scenario.serving_runtime_load_id``). The
   surface's ``served_runtime_load_id`` resolves it:

   +----------------------------+--------------------------------------------+
   | Committed head             | Served version                             |
   +============================+============================================+
   | Live head                  | Its ``runtime_load_id``                    |
   +----------------------------+--------------------------------------------+
   | Per-scenario LoRA adapter  | The committed head version                 |
   +----------------------------+--------------------------------------------+
   | Durable head               | The runtime's ``current_runtime_load_id``  |
   +----------------------------+--------------------------------------------+
   | Initial head (no version)  | ``None``: the check does not run           |
   +----------------------------+--------------------------------------------+

   ``Record2Record.observe_serving_runtime_load_id`` adds each new version to
   a list. A version of another incarnation starts the list again.
2. **Lag.** The lag of a producing version is the number of versions that
   this scenario served after it. The same rule applies to full weights and to
   per-scenario LoRA. A sample is stale when its lag exceeds
   ``max_staleness``, when it has no recorded version, or when the stage did
   not see its version. A sample with several token spans counts every span's
   version. ``stale_sets`` in
   `reef/record2record/staleness.py <../../reef/record2record/staleness.py>`__
   does the arithmetic.
3. **Drop.** Before each batch, ``drop_stale_sets`` ends the held sets that
   have a stale sample (``drop_held_set``). The check does not run while a
   batch is reserved. The trainer releases the records of dropped sets in the
   next commit, so they do not train after a restart.

Batches can mix samples from different versions. Each sample is classified
independently, and a stale sample drops its whole record set. After the
check, payload conversion, validation, and the numerical training path run
unchanged.

.. _3-change-from-runtime-admission:

3. Change from runtime admission
--------------------------------

The earlier design checked the window in the training runtime, after the
batch formed. It split one payload key into an exact execution fence
(``expected_runtime_load_id``, stamped from the serving version) and a
per-sample admission check before backend side effects. That design put the
same rule in several places, and each backend needed its own enforcement.

The current design removes runtime admission:

- The scheduler admission functions and the ``TrainingExecution`` context
  admission.
- The ``runtime.max_staleness`` config and
  ``ExecutorTrainingRuntime.max_staleness`` / ``TrainingRuntime.max_staleness``.
- Tinker's ``stale`` payload flag.
- ``ScenarioHistoryStore.lag`` and ``RuntimeLoadId.lag_behind``.

The payload key ``expected_runtime_load_id`` is now
``parent_runtime_load_id``. It is training metadata: the version that the job
trains from. The training job ID excludes it, so a retry after a newer
publication keeps the same job ID and replays its marker. The payload still
carries each sample's ``producing_runtime_load_ids`` (and
``producing_runtime_load_spans`` when present).

The ``stale`` training outcome remains for backend-specific races, such as a
Tinker incumbent that changed during the step. The dispatcher still drops
such a batch without advancing the scenario step and writes a consumption
receipt.

.. _4-trade-offs:

4. Trade-offs
-------------

- **Rebuilt trainer.** The list of served versions is only in memory. A
  trainer that rebuilds knows only the versions that it sees after the rebuild. Retained
  samples from earlier versions are stale, even when they are within the
  window.
- **No execution fence.** Nothing checks the window between batch build and
  job execution. A commit in that interval can make a reserved batch one or
  more versions older than ``max_staleness`` when it trains.

.. _5-alternatives-considered:

5. Alternatives considered
--------------------------

+----------------------------------+----------------------------------+
| Alternative                      | Why not                          |
+==================================+==================================+
| Keep the gate; make the harness  | Regeneration re-runs the         |
| regenerate stale samples         | environment and discards valid   |
|                                  | recorded behavior                |
|                                  | log-probabilities; it also       |
|                                  | leaves rollout generation        |
|                                  | lockstep.                        |
+----------------------------------+----------------------------------+
| Admit unlimited lag              | A queue of arbitrarily old       |
|                                  | samples can waste train steps    |
|                                  | and hides overload; a bounded    |
|                                  | window keeps the failure mode    |
|                                  | visible.                         |
+----------------------------------+----------------------------------+
| Reef-side importance reweighting | Duplicates training math in a    |
| at admission                     | second layer with independent    |
|                                  | numerical choices; the backend   |
|                                  | already owns the objective.      |
+----------------------------------+----------------------------------+
| Runtime admission with an exact  | The earlier version of this RFC. |
| execution fence                  | It duplicated the rule across    |
|                                  | the scheduler and each backend,  |
|                                  | and it discarded batches after   |
|                                  | the pipeline had built them.     |
+----------------------------------+----------------------------------+

.. _6-rollout-of-the-change:

6. Rollout of the change
------------------------

1. Keep ``max_staleness: 0`` while qualifying the exact-version baseline.
2. Enable a small window (for example ``max_staleness: 2``) and compare the
   existing objective telemetry against that baseline.
