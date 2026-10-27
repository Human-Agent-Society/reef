Troubleshooting
===============

Symptoms, their usual cause, and the fix. Each entry names where to look.

.. page::
   :for: anyone whose deployment, request, or learning step did not do what they expected
   :needs: access to the deployment's logs, ``reef.run-dir/*.log`` (``.reef/run/`` by default; the tutorials use ``work/reef.log``)
   :outcome: the cause found, or the right log to read

Starting Reef
-------------

**reef serve cannot find the config.** A relative ``-c`` path is resolved against the current working directory. Pass an absolute path, or run from the directory that holds the stack file.

**A setting is not what the config says.** The first lines of the launcher log list every resolved setting with its source (``file``, ``command line``, ``environment``, ``automatic``, ``default``), so a command-line flag or ``REEF_*`` variable that overrode the file shows up there; ``reef serve ... --print-config`` prints the full list, defaults included, without starting anything. A file that repeats a key, spells one field two ways, or sets ``null`` on a field that is not optional is refused before startup, naming the field and lines.

**A service never reports ready.** Its ``ready`` probe keeps failing; the stack waits that service's ``ready_timeout`` before giving up. The default is 30 seconds for the Reef HTTP service and 3600 seconds for a managed engine or a training stack, which download weights. Read that service's log under ``reef.run-dir``. For a training stack, the usual causes are a model that is still downloading, serving engines that fail to start on the inference component (read ``inference.log``), or GPUs already in use.

**Boot fails naming a config key.** A ``reef.*`` key that the selected recipe has no field for stops the start rather than being ignored. Recipe fields are listed in `Bundled recipes <recipes.rst>`__; ``harness_evolve`` takes none in the flat section and is configured through a preset.

**Boot fails naming a credential in the tree.** A harness-evolution seed, proposal, or recovered state holding a literal key (``apiKey``, ``token``, and their plural and list forms) is refused, because tree state is persisted and published. Rotate the key, remove it from the entry, and keep credentials in ``inference.upstream-api-key`` or an ``api_key_env``.

Requests
--------

An inference request reaches Reef on ``/v1/chat/completions``, ``/v1/responses``
or ``/v1/messages``; all three take the provider's own request body.

**401 invalid service token.** The ``Authorization: Bearer`` value (or ``x-api-key``, when no Authorization header is sent) is not one of ``reef.token`` / ``reef.tokens``. An unset ``${REEF_TOKEN}`` in the config becomes an empty entry, which is dropped; a deployment with no tokens at all accepts every request.

**400 missing or empty x-reef-scenario.** Every inference, report, and harness read needs the header.

**404 unknown scenario.** The deployment sets ``allow_implicit_scenario_creation: false`` and the scenario has not been created; ``POST /reef/scenarios`` creates it. With implicit creation on, the same typo silently creates a second scenario instead: check ``GET /reef/scenarios`` when traffic seems to vanish. ``POST /reef/train`` answers 404 with either setting and creates nothing.

**409 on an inference request.** Either the ``x-reef-release-id`` header names a version that conflicts with the scenario's binding, or, on a training deployment, the engine answered with a runtime load ID other than the one frozen for the request. The second case is a backend contract violation and should not happen with the bundled stack; ``/reef/status`` shows the current runtime load ID.

**A streaming request is refused on a training scenario.** Streaming through a training deployment requires the token-capturing backend (``inference_handler_factory`` set to the SGLang chat backend, as in the bundled configs); the plain HTTP proxy backend cannot stream there.

**The provider's error came back as 400.** Provider 4xx responses are relayed with the provider's message; read it, the request body is usually the problem. A provider 5xx becomes 502.

**503 this deployment runs no inference engine.** The deployment sets
``inference.backend: none``, so it trains without an engine and serves no
requests. This is expected. Serve a published checkpoint from a separate
deployment, or select an engine with ``inference.backend``. See `Train without
an inference engine
<../reference/configuration.rst#train-without-an-inference-engine>`__.

**503 inference retry deadline exceeded (300s).** Buffered inference exhausted
``inference.retry-timeout-s``. It follows ``inference.timeout-s`` when not set
separately, and both default to 300 seconds. If one valid generation can take
longer—for example, a Designer call against a slow local model—raise the
request timeout:

.. code:: bash

   reef serve ... --inference.timeout-s 1800

If ``inference.retry-timeout-s`` is explicitly configured, make sure it is
also long enough. ``inference.retry-initial-s`` and
``inference.retry-max-s`` control only the delay between attempts; increasing
them does not give one generation more time.

Reports and training
--------------------

**A report was accepted but nothing trains.** Check whether the recipe's trigger is reached (``pipeline.task2job.config.job_samples`` samples, or a complete TTTD rollout grid), whether training reported a data-contract error, and whether the receipts were already consumed by an earlier step. Reports cannot set training eligibility flags, and missing references are rejected at admission. ``GET /reef/scenarios/{scenario}/contract`` shows the recipe contract; ``/reef/status`` shows whether job data is ready (``job_data_ready``), and its ``processor`` entry compares ``ready_samples`` with ``job_samples``. When more samples wait than ``pipeline.task2job.config.buffer_size``, Reef releases the oldest record sets untrained and logs a warning.

**400 on a report.** The recipe declares a report schema and the body violates it: a missing ``score``, a boolean where a number is expected, a missing ``metadata`` field. `Bundled recipes <recipes.rst>`__ lists each schema.

**409 on a report.** The client-chosen ``agent_record_id`` was sent before with different content. Use a new id, or resend identical content.

**A training job fails and /reef/status reports an error.** The service log has the traceback. Slime rejecting ``--wandb-key`` or ``--use-wandb`` means tracking must be configured under ``observability.wandb`` instead. A mismatch between the recipe's loss family and the Slime flags is refused at startup by design.

**A Slime job fails with checkpoint storage is full.** Slime admits a new checkpoint under the checkpoint root only when it fits. Retention first deletes older checkpoints in the order of ``--reef-checkpoint-policy`` (``latest`` or ``best_reward``). It never deletes the latest checkpoint or the checkpoint the job starts from. The error names the reason:

- the storage cap: ``--reef-checkpoint-max-storage-fraction`` (default 0.8) or ``--reef-checkpoint-max-storage``
- the free-space floor: ``--reef-checkpoint-min-free-space-fraction`` (default 0.1) or ``--reef-checkpoint-min-free-space``
- ``unowned checkpoint assets``: files in the checkpoint directories that Slime did not save

Free disk space, change the limits, or move the extra files out of the checkpoint root.

**After a restart, a job trains its data again.** Reef keeps each submitted job, and the training component resumes a job that did not end. Slime's workers load their latest saved checkpoint again. A job that stopped after Megatron saved its checkpoint records that checkpoint and does not train again. A job that stopped before the save trains its data again from the latest checkpoint. A Tinker job continues after its last saved checkpoint.

**A rollback is refused on a Slime scenario.** Slime's workers continue from their own latest checkpoint and cannot load a published release, so Reef refuses the rollback. Tinker scenarios can roll back.

Harness evolution
-----------------

**reports are accepted but no evolve step runs.** Check the configured training mode and ``job_samples`` (by default ``evolution.batch_size``). Successful and failed outcomes both contribute samples; there is no score-window filter. Inspect training errors for missing inputs or incompatible trajectory data.

**startup fails installing the harness binary.** With ``evolution.binary`` unset, startup installs the descriptor's pinned binary. Install the vendor tool named in the error (``npm`` for pi), fix the reported vendor failure, or set ``evolution.binary`` to an existing executable.

**no skill mutation passed the selection.** A step ran and the candidate did not win. Read the step's episodes in the service log. Both sides scoring nothing means the episodes could not run: the adapter binary could not be launched (``evolution.binary``, when set, must point at a real binary), the endpoint rejected ``tool_choice: "auto"`` (vLLM needs ``--enable-auto-tool-choice --tool-call-parser hermes``), or an episode exceeded the 600 second timeout. Both sides scoring the same means the proposal did not change the outcome; ``selection: always`` publishes every applied mutation if that is what you want.

**GET /reef/harness returns 404.** Nothing has been published yet, or the scenario's recipe serves no files. The catalog at ``GET /reef/harness/releases`` lists what exists.

**reef-pi captures no receipts, so report has nothing to send.** The installed ``reef-client`` is older than 0.2.0 and reads a header the service no longer sends. ``pip install -U "reef-client>=0.2.0"``.

**reef-<adapter> exits with no Reef URL in the tree's model binding files.** The wrapper finds Reef through the file the adapter's model binding renders its endpoint into: ``pi-agent/models.json``, ``opencode/opencode.json``, ``claude/settings.json``, ``dsh/profiles/headless/cordis.patch.yml`` (and ``dsh/profiles/web/cordis.patch.yml`` for ``reef-dsh web``), ``hermes/config.yaml``, ``native/models.json``. The published tree carries no endpoint on purpose; write the binding with your Reef URL there before running the wrapper. A codex tree (``codex/config.toml``) is rewritten the same way; codex calls Reef's ``/v1/responses`` route.

Docs and links
--------------

**A docs link points at GitHub instead of a page.** Only files under
``docs/*.rst`` are site pages; historical RFCs, examples, and code are linked
on GitHub, where they are read.
