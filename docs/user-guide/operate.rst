Operate a deployment
====================

Check health, change served artifacts and recover training. Start a
`weight <evolve-your-model.rst>`__ or `harness <evolve-your-harness.rst>`__ deployment.
Set ``REEF_URL`` and ``REEF_TOKEN`` for these commands.

Check health and status
-----------------------

.. code:: bash

   curl -f "$REEF_URL/healthz"
   curl -sS -H "Authorization: Bearer $REEF_TOKEN" "$REEF_URL/reef/status"

``/healthz`` checks HTTP readiness. Status shows jobs and asynchronous errors;
inspect ``last_training_commit`` and ``processor`` when training appears stuck.
Managed services write logs under ``reef.run-dir`` (default ``.reef/run/``).
For W&B, see `experiment tracking
<../reference/configuration-observability.rst#experiment-tracking>`__.

Read the artifacts
------------------

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

Artifacts appear newest first. Their IDs are opaque. Skipped and rejected jobs
appear only in ``GET /reef/scenarios/{scenario}/operations``.
Use ``GET /reef/harness/artifacts`` for harness step results.

New scenarios start from the configured base model or harness seed. For explicit
starts and imports, see the `artifact API <../reference/http-api-artifacts.rst>`__.
Imports change neither serving nor training starts; harness sync changes both.

Set what a scenario serves
--------------------------

Replace the served selection with artifact IDs and integer traffic weights:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {"<id>": 100}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

Nonempty weights must sum to 100. The highest-weight artifact is active.
``{"a": 95, "b": 5}`` sends 5% of traffic to ``b``.
A zero weight keeps an artifact available through its dedicated URL.
``{}`` stops serving but keeps training; inference returns 404.

Add ``component`` for a composite scenario. An empty mapping without it stops
all components. Serving selections survive restarts.

Poll ``GET /reef/scenarios/{scenario}/inference/status`` before sending requests.
Add ``?model=<artifact-id>`` to check another served artifact.
Resend ``served`` to retry failed startup immediately.

Tinker can serve several artifacts per scenario; adapter engines support one.
Full weights require one remote component per checkpoint, configured through
``inference.remote.slot-urls``. Removed components retire after ``inference.timeout-s``
plus 5 seconds (default 305), measured from route publication. Work or release
failures can delay retirement; no API reports when the slot is free.
If replacement startup fails, wait and resend the mapping.

Activation rule
---------------

The deployment's ``activation.module`` controls new artifacts:

- ``always`` (default): activate, except harness changes held for review.
- ``never``: require manual activation through ``served``.
- ``evaluated``: require an evaluation to reach ``min_mean_score``.

This weight job uses the default training start A with ``activation.module: never``:

.. code:: mermaid

   flowchart TB
       accTitle: Serving and training can use different artifacts
       accDescr: Artifact A serves requests. A job from the default start A commits checkpoint B, the next default training start. Activating and loading B changes serving without changing the training start.
       A["Artifact A"] -->|"Train and<br/>commit"| B["Artifact B"]
       A -->|Serves| R["Agent<br/>requests"]
       B -->|"Default<br/>start"| J["Next<br/>training job"]
       B -.->|"Activate<br/>and load"| R

Set ``{"activation": {"module": "never"}}`` in a scenario update to override
the deployment rule. Set ``activation`` to null to restore that rule.
See `activation configuration
<../reference/configuration-recipe.rst#the-activation-section>`__.

Train from an artifact
----------------------

Change the next default training start:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

Add ``component`` for a composite scenario. This cancels training from the previous
default start and begins a new tracking run. Serving does not change.
Tinker restores weights and optimizer state; harness training uses the tree.
Without optimizer state, training starts a new optimizer.

A `manual job <../reference/http-api-training.rst#manual-training>`__ can name
``train_from`` for that job alone. It leaves the default unchanged; later
train-from requests do not cancel it.

Slime refuses train-from with HTTP 400. To resume an earlier checkpoint, create
a new deployment and scenario. Set new ``save`` and ``save-hf`` directories,
``training.options.load`` to the old directory, and ``ckpt-step`` to the saved rollout.
Do not move the existing scenario: its recorded start belongs to the old deployment.

Go back to an older artifact
----------------------------

Restore serving from a previous operation:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" -d '{"commit_number": 12}' \
     "$REEF_URL/reef/scenarios/code-repair/restore"

Read commit numbers from the operations endpoint. Commit 0 selects starting artifacts.
Restore creates no artifact and leaves training starts unchanged.
Set ``activation.module: never`` to prevent the next training artifact replacing it.
Poll inference startup before sending requests.

Set the checkpoint cadence
--------------------------

Set ``recipe.config.job_parameters.save_interval`` for intermediate saves:

- Tinker and Slime sample jobs count optimizer steps.
- Slime task and dataset jobs count rollout iterations.

Both save at job end; without an interval, they save only there.
Slime's native ``save-interval`` does not set Reef's cadence, but Megatron
requires it when ``save`` is configured.

Slime retention preserves the next job's start. A job fails if its next checkpoint
cannot fit; submitted samples remain consumed. Set absolute limits with
``training.options.reef-checkpoint-max-storage`` and ``reef-checkpoint-min-free-space``.

Restart and recovery
--------------------

Keep `stores and backend state <storage.rst#back-up-a-deployment>`__ persistent.
Reef restores committed selections and saved training progress. Unsaved work
can run again. Settled harness steps commit their result; unsettled steps rerun.

``artifact_head_sync: pending`` or ``conflict`` means repository synchronization
is incomplete. A conflict does not undo the commit, but blocks later commits.
See the `recovery contract <../advanced_topics/state-model.rst#commit-ordering>`__.

Deleting a scenario removes it immediately and permanently reserves its ID.
Background cleanup cancels jobs and archives state; failed cleanup retries hourly.
For failover, use `split services <split-services.rst#control-restarts>`__.
