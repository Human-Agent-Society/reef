Bounded stale-sample training without giving up the version fence
=================================================================

:Status: Deprecated

.. warning::

   This historical RFC predates the issue-based RFC process and will be removed
   in a future cleanup.

   Today every training job is fenced to the exact serving runtime load ID that
   produced its rollouts; anything older is discarded. That fence makes
   version lag zero **by construction**. This RFC adds one shared, bounded
   staleness window: keep the fence's correctness job (serving identity at
   execution time), stop using it as the admission policy for sample
   provenance, and let the existing training path own admitted stale data. It
   is implemented as the shared opt-in ``max_staleness`` setting (default ``0``).

.. _1-where-the-gate-lives-today:

1. Where the gate lives today
-----------------------------

Three independently reasonable mechanisms combine to produce this behavior:

1. **Provenance becomes the fence.**
   `executor_runtime.prepare_training_step <../../reef/runtime/adapters/executor_runtime.py>`__
   collects the batch's producing runtime load IDs, requires exactly one, and
   stamps it as the job's ``expected_runtime_load_id``. The samples' *provenance*
   is promoted into the job's *identity*.
2. **The bridge rejects mismatch.**
   `bridge._run_train_step <../../reef/train/slime_backend/reef_adapters/bridge.py>`__
   compares ``expected_runtime_load_id`` against the currently published serving
   version and returns ``outcome="stale"`` on any difference.
3. **The dispatcher discards.** On ``stale``,
   `dispatcher._process <../../reef/dispatcher.py>`__ calls ``reject_pending()``:
   the reserved rollouts are dropped rather than retried. A retry would carry
   the same producing version and fail the same comparison.

As a result, a rollout is trainable **only** while the weights that
produced it are still serving. The moment a commit publishes a new version,
every in-flight rollout becomes ineligible.

.. _2-why-that-is-the-wrong-resting-point:

2. Why exact matching is too restrictive
----------------------------------------

The training path already owns how a sample is consumed. Staleness admission
should only bound *which* producing versions reach that path; it should not
inspect the configured recipe or objective, reinterpret the payload, or add
another importance weight. An unconditional exact-provenance gate wastes
in-flight work and makes asynchronous rollout generation lockstep.

The exact gate remains the safest default. This RFC separates correctness
fencing from freshness admission while keeping ``max_staleness: 0`` inert.

.. _3-proposal--split-the-fence-from-the-freshness-policy:

3. Proposal: split the fence from the freshness policy
------------------------------------------------------

``expected_runtime_load_id`` currently answers two different questions with one
comparison:

+----------------------+-----------------------+----------------------+
| Question             | Nature                | Right owner          |
+======================+=======================+======================+
| "Is this job         | fencing / correctness | bridge, exact match, |
| executing against    |                       | unchanged            |
| the serving state it |                       |                      |
| was prepared         |                       |                      |
| against?"            |                       |                      |
+----------------------+-----------------------+----------------------+
| "Is this sample      | admission policy      | shared training      |
| fresh enough to      |                       | config, bridge       |
| learn from?"         |                       | enforcement          |
+----------------------+-----------------------+----------------------+

.. _31-fence-on-serving-identity-not-sample-provenance:

3.1 Fence on serving identity, not sample provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the window enabled, ``prepare_training_step`` stamps
``expected_runtime_load_id`` from the runtime's own ``serving_runtime_load_id()`` at
preparation time instead of from the samples. The bridge's exact-match check
still catches the race it exists for (the serving version moved between
preparation and execution). A parallel ``producing_runtime_load_ids`` list in the
shared training payload retains each sample's provenance. At ``W = 0``, Reef
keeps the old producing-version fence and does not add a serving probe.

.. _32-a-staleness-window-before-bridge-side-effects:

3.2 A staleness window before bridge side effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared training setting ``max_staleness: W`` defaults to ``0``, so the change
is inert until opted into. After the exact execution fence and before packing,
training, or checkpoint side effects, a sample is admissible iff its producing
version is within ``W`` published steps of the current serving version **within
the same incarnation**:

- ``lag = serving_step − producing_step``, parsed from canonical
  ``WeightVersion`` tokens; ``0 ≤ lag ≤ W`` admits.
- A sample from a different incarnation has undefined lag: never admitted for
  training, dropped with a counted metric rather than silently.
- A sample whose lag exceeds ``W`` is dropped at bridge admission, also counted
  (``staleness/samples_dropped``), so the discard rate is observable instead of
  implicit.

Mixed-provenance batches become legal (the "exactly one producing version"
assertion relaxes to "every version admissible under the window"). Each sample
is classified independently; if any sample is inadmissible, the reserved batch
is dropped atomically.

.. _33-the-training-objective-stays-untouched:

3.3 The training objective stays untouched
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Admission receives only the configured window, the serving version, and each
sample's producing version. It does not branch on recipe or loss-family names.
After admission, the existing payload conversion, validation, and numerical
training path runs unchanged.

.. _4-interactions:

4. Interactions
---------------

.. _41-one-job-per-version-and-the-marker-protocol:

4.1 One-job-per-version and the marker protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``RUNNING → CHECKPOINT → COMPLETE`` progression and replay ordering stay
unchanged. For enabled-window jobs, ``job_id`` hashes the retry-stable sample
payload, rollout ID, and staleness window but excludes the volatile preparation
fence. Otherwise a lost acknowledgement after publication would re-prepare
against the newly published serving version, produce a different job ID, and
miss the COMPLETE marker. A fresh execution still checks the exact fence
before any side effect; marker replay remains first.

.. _42-recovery:

4.2 Recovery
~~~~~~~~~~~~

On clean restart the bridge republishes the recovered checkpoint under its
recorded serving token, so lag remains comparable after recovery. A genuinely
new serving incarnation never compares equal and all
older samples are dropped as cross-incarnation. The recovery-pair invariant,
checkpoint retention, and the critic checkpoint are unaffected because none of
them consume provenance.

.. _43-what-does-not-change:

4.3 What does *not* change
~~~~~~~~~~~~~~~~~~~~~~~~~~

- The bridge's exact-match execution fence. The dispatcher still rejects a
  ``stale`` result without advancing a scenario step, but now writes a durable
  compaction receipt and a warning before deleting source payloads.
- Payload validation and numerical training implementations.
- Deployments that keep ``W = 0`` retain the current exact-version semantics and
  avoid the extra serving-version probe.

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

.. _6-rollout-of-the-change:

6. Rollout of the change
------------------------

1. Keep ``max_staleness: 0`` while qualifying the exact-version baseline.
2. Enable a small window (e.g. ``W = 2``) and observe fresh, admitted-stale, and
   dropped counts.
3. Compare the existing objective telemetry against the exact-version
   baseline.
