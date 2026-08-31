Operate a deployment
====================

What to check while a deployment runs, how to read and steer its version chain, how to keep an eye on training, and what survives a restart.

.. page::
   :for: whoever runs a Reef deployment
   :needs: a running deployment from `Evolve your model <evolve-your-model.rst>`__ or `Evolve your harness <evolve-your-harness.rst>`__, its URL and token
   :outcome: the routine checks, the version operations, and the recovery rules

The examples use ``REEF_URL`` and ``REEF_TOKEN`` as in `HTTP API <../reference/http-api.rst>`__.

Check health and status
-----------------------

.. code:: bash

   curl -f "$REEF_URL/healthz"
   curl -sS -H "Authorization: Bearer $REEF_TOKEN" "$REEF_URL/reef/status"

``/healthz`` answers as soon as the HTTP service is up; it says nothing about training. ``/reef/status`` is the training side: the last asynchronous error, model preload failures, and for every scenario its step counter, the weight version being served, checkpoint storage state, whether a batch is waiting, the processor's state, and whether inference is admitted or paused for a weight update. It is the first place to look when requests keep being served by an old version.

The service and every process ``reef serve`` started write logs under ``run_dir`` (``/tmp/reef-stack/`` by default), one ``<service>.log`` and one ``<service>.pid`` each.

Read the version chain
----------------------

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" \
     "$REEF_URL/reef/scenarios/code-repair/versions"

Newest first. Each row names the version, its parent, whether it is a durable checkpoint (``checkpoint``, ``restorable``), what produced it (``operation``: ``creation``, ``training``, ``rollback``, ``recovery``), whether it is the one currently served, and for training rows the step's ``metrics``. A version can be live (``artifact_kind: live_weights``: the engine has the weights, the repository has only the record) or saved (a Git LFS commit).

For harness scenarios, ``GET /reef/harness/versions`` lists the same chain oldest first with each step's gate metrics, and ``GET /reef/harness?version=<id>`` returns any listed tree.

Pin a version
-------------

A client that must keep answering from one version sends ``x-reef-artifact-version: <id>`` with its requests. Pinning is per request and changes nothing on the server; a pin that conflicts with the scenario's binding is refused with 409.

Roll back
---------

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_version": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/rollback"

Rollback republishes the target as a new commit and makes it current; history is not rewritten, and the step counter keeps increasing. Only versions marked ``restorable`` qualify: durable checkpoints. Live weight versions that were never checkpointed cannot be restored, and the bundled Ray/Slime runtime does not implement checkpoint restoration, so rollback currently applies to harness artifacts; for weights, redeploy from the checkpoint you want.

Set the checkpoint cadence
--------------------------

``checkpoint_every_n_versions`` (default ``1``) decides how many accepted updates go by between durable checkpoints. Between checkpoints, new weights live only in the engine's memory: their versions are recorded, but their bytes are not. A restart restores the last checkpoint, and the step counter, algorithm state, and record progress continue from the log. Raise the cadence only when checkpoint writes are the bottleneck and losing live versions on a restart is acceptable.

On a training deployment, checkpoint retention runs under ``--reef-checkpoint-policy`` (``latest`` or ``best_reward``) with storage-fraction limits; when storage is blocked the step is deferred rather than failed, and ``/reef/status`` shows ``checkpoint_storage``.

Track experiments with W&B
--------------------------

Tracking is optional and off by default; ``observability.wandb`` in `Configuration <../reference/configuration.rst#experiment-tracking>`__ lists every key. Export ``WANDB_API_KEY`` before starting the stack; there is no key field, and the Slime driver refuses ``--wandb-key`` so a credential never enters a command line or a run config.

What you see in W&B: one group per scenario and one run per scenario, plus a new run after every rollback, so each post-rollback branch is its own curve. Every training result lands on the run-local ``train/step`` axis carrying the monotonic ``reef/step`` that joins it to the commit log, and the commit metrics record ``experiment/run_id``, so a Reef version leads to its run and the run's ``reef/training_job_id`` leads back. Tracking failures are logged and never fail a training step or its commit.

Restart and recovery
--------------------

What survives a restart, provided the storage paths are persistent:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - State
     - Guarantee
   * - records
     - persisted before the processor sees them; never trained twice
   * - the commit log
     - append-only per scenario; the fsynced append is the commit point
   * - checkpointed versions
     - in the Git LFS repository; the recovered head is what is served
   * - algorithm state and record progress
     - restored from the log's head record
   * - live weights
     - not recoverable; the last checkpoint is restored
   * - a training step in flight
     - not recoverable; the batch is replayed after the step is settled

The record store, commit logs, and repository live under ``.reef/`` by default (``agent_record_dir``, ``artifact_repository``, ``artifact_work_dir``, ``artifact_cache_dir``). On ephemeral storage none of the guarantees above hold past its loss. Each scenario needs one Reef writer; run a second deployment on other ports and storage paths rather than two services on one store. `Architecture <../getting-started/architecture.rst#durability>`__ describes the commit ordering behind the table.
