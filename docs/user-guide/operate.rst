Operate a deployment
====================

Set ``REEF_URL`` and ``REEF_TOKEN`` for your running deployment.
Replace ``code-repair`` below with your scenario.

Check health and status
-----------------------

.. code:: bash

   curl -f "$REEF_URL/healthz"
   curl -sS -H "Authorization: Bearer $REEF_TOKEN" "$REEF_URL/reef/status"

Health checks HTTP readiness. Status shows jobs and errors.
Service logs are under ``reef.run-dir`` (default ``.reef/run/``).

Read the artifacts
------------------

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

New scenarios start from the base model or harness seed. See the
`artifact API <../reference/http-api-artifacts.rst>`__ for fields, imports and explicit starts.

Set what a scenario serves
--------------------------

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" -d '{"served": {"<id>": 100}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

Add ``component`` for composite scenarios. Nonempty weights sum to 100.
``{}`` stops serving. See `traffic splits <../reference/http-api-serving.rst>`__.

Poll ``GET /reef/scenarios/{scenario}/inference/status`` before requests.
Resend ``served`` to retry failed startup.

Tinker supports several artifacts per scenario. Adapter engines support one.
Full weights need one remote component per checkpoint (``inference.remote.slot-urls``).
Removed components retire ``inference.timeout-s`` plus 5 seconds after route publication
(default 305 seconds). Work or release failures can delay this. No API reports free slots.
If replacement startup fails, wait and resend.

Activation rule
---------------

``activation.module`` defaults to ``always``. Use ``never`` for manual activation
or ``evaluated`` for a score threshold. Harness review requirements still apply.
See `activation settings <../reference/configuration-recipe.rst#the-activation-section>`__.

This weight job uses default start A with ``activation.module: never``.

.. code:: mermaid

   flowchart TB
       accTitle: Serving and training can use different artifacts
       accDescr: Artifact A serves requests. A job from the default start A commits checkpoint B, the next default training start. Activating and loading B changes serving without changing the training start.
       A["Artifact A"] -->|"Train and<br/>commit"| B["Artifact B"]
       A -->|Serves| R["Agent<br/>requests"]
       B -->|"Default<br/>start"| J["Next<br/>training job"]
       B -.->|"Activate<br/>and load"| R

Train from an artifact
----------------------

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

Add ``component`` for composite scenarios. This cancels training from the previous
default start. Serving stays unchanged.
For one job, use `manual train_from <../reference/http-api-training.rst#manual-training>`__.
For serving rollback, use `restore <../reference/http-api-serving.rst#restore-a-scenario-selection>`__.

Slime rejects train-from with 400. To resume an earlier checkpoint, create a new
deployment and scenario. Use new ``save``/``save-hf`` directories.
Set ``training.options.load`` to the old directory and ``ckpt-step`` to the saved rollout.
Keep the existing scenario in its original deployment.

Set the checkpoint cadence
--------------------------

Set ``recipe.config.job_parameters.save_interval``. Tinker and Slime sample jobs
count optimizer steps. Slime task and dataset jobs count rollout iterations.
Both also save at job end. Omit the interval to save only there.

Megatron requires native ``save-interval`` when ``save`` is configured.
Slime retention preserves the next job's start. If a checkpoint cannot fit, the job
fails and submitted samples remain consumed. Set ``training.options.reef-checkpoint-max-storage``
and ``reef-checkpoint-min-free-space`` for absolute limits.
A Megatron checkpoint holds the optimizer state, about 8 times the HF export without LoRA.
``--no-save-optim`` keeps the weights only, about 2 times, but every job then restarts Adam,
so it needs ``--reef-accept-optimizer-restart``.

Restart and recovery
--------------------

Keep `persistent stores and checkpoints <storage.rst#back-up-a-deployment>`__.
Unsaved training work can repeat. ``artifact_head_sync: conflict`` blocks later
commits without undoing the existing commit. See the `recovery contract
<../advanced_topics/state-model.rst#commit-ordering>`__.
Use `split services <split-services.rst#control-restarts>`__ for failover.
