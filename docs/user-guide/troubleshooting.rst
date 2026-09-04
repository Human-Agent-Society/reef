Troubleshooting
===============

Symptoms, their usual cause, and the fix. Each entry names where to look.

.. page::
   :for: anyone whose deployment, request, or learning step did not do what they expected
   :needs: access to the deployment's logs, ``run_dir/*.log`` (``/tmp/reef-stack/`` by default; ``work/reef.log`` in the examples)
   :outcome: the cause found, or the right log to read

Starting Reef
-------------

**reef serve cannot find the config.** A relative ``-c`` path is resolved against the Reef checkout root, not the current directory. Pass an absolute path, or run from the checkout root.

**A service never reports ready.** Its ``ready`` probe keeps failing; the stack waits ``ready_timeout`` seconds (3600 by default) before giving up. Read that service's log under ``run_dir``. For a training stack, the usual causes are a model that is still downloading, a ``reef.inference_url`` override that does not match where Slime bound its router (leave it unset; Reef takes the address from the training actor), or GPUs already in use.

**Boot fails naming a config key.** A ``reef.*`` key that the selected recipe has no field for stops the start rather than being ignored. Recipe fields are listed in `Bundled recipes <recipes.rst>`__; ``harness_evolve`` takes none in the flat section and is configured through a preset.

**Boot fails naming a credential in the tree.** A harness-evolution seed, proposal, or recovered state holding a literal key (``apiKey``, ``token``, and their plural and list forms) is refused, because tree state is persisted and published. Rotate the key, remove it from the entry, and keep credentials in ``reef.upstream_api_key`` or an ``api_key_env``.

Requests
--------

**401 invalid service token.** The ``Authorization: Bearer`` value is not one of ``reef.token`` / ``reef.tokens``. An unset ``${REEF_TOKEN}`` in the config becomes an empty entry, which is dropped; a deployment with no tokens at all accepts every request.

**400 missing or empty x-reef-scenario.** Every inference, report, and harness read needs the header.

**404 unknown scenario.** The deployment sets ``allow_implicit_scenario_creation: false`` and the scenario has not been created; ``POST /reef/scenarios`` creates it. With implicit creation on, the same typo silently creates a second scenario instead: check ``GET /reef/scenarios`` when traffic seems to vanish.

**409 on an inference request.** Either the ``x-reef-release-id`` header names a version that conflicts with the scenario's binding, or, on a training deployment, the engine answered with a runtime load ID other than the one frozen for the request. The second case is a backend contract violation and should not happen with the bundled stack; ``/reef/status`` shows the current runtime load ID.

**A streaming request is refused on a training scenario.** Streaming through a training deployment requires the token-capturing backend (``inference_backend_factory`` set to the SGLang chat backend, as in the bundled configs); the plain HTTP proxy backend cannot stream there.

**The provider's error came back as 400.** Provider 4xx responses are relayed with the provider's message; read it, the request body is usually the problem. A provider 5xx becomes 502.

Reports and training
--------------------

**A report was accepted but nothing trains.** In order of likelihood: the recipe's step trigger is not reached yet (``batch_size`` reports, or a full ``groups_per_step × rollouts_per_group`` grid for ``tttd``); the report carries no finite ``score`` and the recipe requires one; ``metadata.training.eligible`` is ``false``; or the referenced receipts were already consumed by an earlier step, in which case the report is accepted and ignored. ``GET /reef/scenarios/{scenario}/contract`` shows what the recipe consumes; ``/reef/status`` shows whether a batch is ready.

**400 on a report.** The recipe declares a report schema and the body violates it: a missing ``score``, a boolean where a number is expected, a missing ``metadata`` field. `Bundled recipes <recipes.rst>`__ lists each schema.

**409 on a report.** The client-chosen ``agent_record_id`` was sent before with different content. Use a new id, or resend identical content.

**The training step fails and /reef/status reports an error.** The service log has the traceback. A driver rejecting ``--wandb-key`` or ``--use-wandb`` means tracking must be configured under ``observability.wandb`` instead. A mismatch between the recipe's loss family and the Slime flags is refused at startup by design.

**After a restart the previous live weights are gone.** Weights between checkpoints exist only in engine memory; a restart restores the last checkpoint and the step counter continues from there. Keep ``checkpoint_every_n_versions`` at 1 unless you can afford to lose live versions. A ``RUNNING`` job marker left by a crash mid-step needs an operator to decide whether the job completed before the stack is restarted.

Harness evolution
-----------------

**every task passed: nothing batched, no evolve step runs.** The recipe learns from failures: only reports scoring at or below ``data.max_score`` (``0.0`` in the example) batch. Use a model that fails a task, raise ``max_score``, or add tasks the model gets wrong.

**no skill mutation won a gate.** A step ran and the candidate did not win. Read the step's episodes in the service log. Both sides scoring nothing means the episodes could not run: the adapter binary is not on ``PATH`` (or ``evolution.binary`` is wrong), the endpoint rejected ``tool_choice: "auto"`` (vLLM needs ``--enable-auto-tool-choice --tool-call-parser hermes``), or an episode exceeded the 600 second timeout. Both sides scoring the same means the proposal did not change the outcome; ``selection: always`` publishes every applied mutation if that is what you want.

**GET /reef/harness returns 404.** Nothing has been published yet, or the scenario's recipe serves no files. The catalog at ``GET /reef/harness/releases`` lists what exists.

**reef-pi captures no receipts, so report has nothing to send.** The installed ``reef-client`` is older than 0.2.0 and reads a header the service no longer sends. ``pip install -U "reef-client>=0.2.0"``.

**reef-<adapter> exits with no Reef URL in the tree's model binding files.** The wrapper finds Reef through the file the adapter's model binding renders its endpoint into: ``pi-agent/models.json``, ``opencode/opencode.json``, ``claude/settings.json``, ``dsh/profiles/headless/cordis.patch.yml``, ``hermes/config.yaml``, ``native/models.json``. The published tree carries no endpoint on purpose; write the binding with your Reef URL there before running the wrapper. A codex tree (``codex/config.toml``) is rewritten the same way, but codex speaks the Responses dialect and Reef serves no ``/v1/responses`` route yet, so its calls capture nothing until it does.

Docs and links
--------------

**A docs link points at GitHub instead of a page.** Only files under
``docs/*.rst`` are site pages; historical RFCs, examples, and code are linked
on GitHub, where they are read.
