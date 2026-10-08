Troubleshooting
===============

Symptoms, their usual cause, and the fix. Each entry names where to look.

.. page::
   :for: anyone whose deployment, request, or learning step did not do what they expected
   :needs: access to the deployment's logs, ``run_dir/*.log`` (``.reef/run/`` by default; a legacy ``/tmp/reef-stack`` stack keeps its own directory, and the tutorials use ``work/reef.log``)
   :outcome: the cause found, or the right log to read

Starting Reef
-------------

**reef serve cannot find the config.** A relative ``-c`` path is resolved against the current working directory. Pass an absolute path, or run from the directory that holds the stack file.

**A setting is not what the config says.** The first lines of the launcher log list every resolved setting with its source (``file``, ``command line``, ``environment``, ``automatic``, ``default``), so a command-line flag or ``REEF_*`` variable that overrode the file shows up there; ``reef serve ... --print-config`` prints the full list, defaults included, without starting anything. A ``schema-version: 2`` file that repeats a key, spells one field two ways, or sets ``null`` on a field that is not optional is refused before startup, naming the field and lines.

**A service never reports ready.** Its ``ready`` probe keeps failing; the stack waits that service's ``ready_timeout`` before giving up. The default is 30 seconds for the Reef HTTP service and 3600 seconds for a managed engine or a training stack, which download weights. Read that service's log under ``run_dir``. For a training stack, the usual causes are a model that is still downloading, an ``inference.url`` override that does not match where Slime bound its router (leave it unset; Reef takes the address from the training actor), or GPUs already in use.

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

**A report was accepted but nothing trains.** Check whether the recipe's trigger is reached (``batch_size`` samples, or a complete TTTD rollout grid), whether training reported a data-contract error, and whether the receipts were already consumed by an earlier step. Reports cannot set training eligibility flags, and missing references are rejected at admission. ``GET /reef/scenarios/{scenario}/contract`` shows the recipe contract; ``/reef/status`` shows whether a batch is ready.

**400 on a report.** The recipe declares a report schema and the body violates it: a missing ``score``, a boolean where a number is expected, a missing ``metadata`` field. `Bundled recipes <recipes.rst>`__ lists each schema.

**409 on a report.** The client-chosen ``agent_record_id`` was sent before with different content. Use a new id, or resend identical content.

**The training step fails and /reef/status reports an error.** The service log has the traceback. A driver rejecting ``--wandb-key`` or ``--use-wandb`` means tracking must be configured under ``observability.wandb`` instead. A mismatch between the recipe's loss family and the Slime flags is refused at startup by design.

**After a restart the previous live weights are gone.** Weights between checkpoints exist only in engine memory; a restart restores the last checkpoint and the step counter continues from there. Keep ``checkpoint_every_n_versions`` at 1 unless you can afford to lose live versions.

**A restart fails with ambiguous training job <id>.** Reef stopped or failed while that Slime training job trained or saved its checkpoint. So the job marker ``.reef-latest-job.json`` in the HF checkpoint directory (``--save-hf``) still says ``RUNNING``. The checkpoint directories cannot always show whether the job saved its optimizer step. They also cannot show whether the job before it was committed. If Reef trains the job's batch again, it can apply the step twice. So the Slime driver refuses to start. The error lists:

- the marker's rollout, ``N`` below
- what Megatron's tracker, ``latest_checkpointed_iteration.txt`` in the Megatron checkpoint directory (``--save``), says: the iteration it names, ``P`` below, or that it is missing or unreadable
- the paths for rollout ``N`` that exist, including its checkpoint record
- whether the run trains per-scenario LoRA adapters

Reef keeps no copy of the settled marker that ``RUNNING`` replaced. So recovery is manual. The checkpoint root is the directory that holds the HF and Megatron checkpoint directories, the marker, and ``.reef-retention``. Unless all the conditions below hold, restore the checkpoint root from a copy. Use a copy from a time when no job ran and the marker said ``COMPLETE``. That is after the last committed job and before the next job started.

The preflight also refuses a copy whose marker says ``REJECTED``. So if the job before this one was rejected, the copy must be older than that job. Reef does not take these copies. Automatic recovery is tracked in `#333 <https://github.com/Human-Agent-Society/reef/issues/333>`__.

For a first job, restore an empty checkpoint root. This reset discards the interrupted job's batch. No marker existed before the first job, so Reef starts under a new runtime load ID, as on a first start. In a full-weight run, Reef then drops the batch as stale. The reset also removes the Megatron checkpoint, the scenario history and the adapter snapshots, so no weights keep the job's step.

*Full-weight run, the job saved nothing.* Recover by hand only when all of these hold:

1. The error says ``per-scenario LoRA: no``. The tracker names an iteration ``P`` below ``N``. The error lists no paths for rollout ``N``. ``--load`` is the same directory as ``--save``, as in the bundled configurations.

   A per-scenario LoRA run always needs the copy. Its settled marker also names the scenario and the scenario's runtime load ID. Also, a save that started can already have rewritten the scenario's adapter snapshot in ``reef_adapter_slots``.
2. ``.reef-retention/records/`` under the checkpoint root has a record for ``P``. The file name is ``P`` padded to 20 digits, then ``.json``. No record is newer. Its ``job_id`` is the job for rollout ``P``.
3. That job was committed. A rejected job also leaves a checkpoint record and moves the tracker. Its rollout holds the declined candidate. Retention can also have deleted the committed rollout before it.

   Each scenario has a commit log in ``agent_record_dir`` on the Reef service's host. The default is ``.reef/agent-record``. A committed training step records its job ID as ``training_job_id``. A rejected step does not. But a rejected step still has the ID in another field of its line. So search for the exact pair:

   .. code:: bash

      grep -l '"training_job_id":"<job_id>"' <agent_record_dir>/*.commits.jsonl

   A printed file name means that the job was committed. No output means that the job was rejected, or that its commit cannot be confirmed.
4. The ``RUNNING`` marker has a ``parent_runtime_load_id``: the runtime load ID that rollout ``P`` is served under.

Then replace the marker with the settled marker of rollout ``P``. For example, with ``P`` = 4 and ``--save-hf /data/ckpt/hf/{rollout_id}``:

.. code:: json

   {
     "status": "COMPLETE",
     "job_id": "<job_id in /data/ckpt/.reef-retention/records/00000000000000000004.json>",
     "rollout_id": 4,
     "checkpoint_path": "/data/ckpt/hf/4",
     "runtime_load_id": "<parent_runtime_load_id in the RUNNING marker>",
     "commit_acknowledged": true
   }

The preflight then accepts the checkpoint directories. Reef does not count the interrupted job's batch as trained. Do not use the interrupted job's ID as ``job_id``. If you do, Reef takes its batch as already trained.

Do not delete the marker to get past the error. The checkpoint directories stay, so the trainer loads the iteration that the tracker names. In a full-weight run without a marker, Reef publishes these weights under a new runtime load ID. They can be a rejected candidate, or they can already hold the interrupted job's step. Reef also drops the interrupted job's batch as stale. The empty-root reset for a first job loses that batch too. But it leaves no checkpoint, so Reef starts as on a first start.

In a per-scenario LoRA run, admission compares the batch with the scenario's publications in ``reef_scenarios.json``. It does not compare the batch with the serving runtime load ID. So Reef admits the batch again. The batch can then train on an adapter snapshot that already holds the job's step.

Harness evolution
-----------------

**reports are accepted but no evolve step runs.** Check the configured training mode and batch size. Successful and failed outcomes both contribute samples; there is no score-window filter. Inspect training errors for missing inputs or incompatible trajectory data.

**startup fails installing the harness binary.** With ``evolution.binary`` unset, startup installs the descriptor's pinned binary. Install the vendor tool named in the error (``npm`` for pi), fix the reported vendor failure, or set ``evolution.binary`` to an existing executable.

**no skill mutation won a gate.** A step ran and the candidate did not win. Read the step's episodes in the service log. Both sides scoring nothing means the episodes could not run: the adapter binary could not be launched (``evolution.binary``, when set, must point at a real binary), the endpoint rejected ``tool_choice: "auto"`` (vLLM needs ``--enable-auto-tool-choice --tool-call-parser hermes``), or an episode exceeded the 600 second timeout. Both sides scoring the same means the proposal did not change the outcome; ``selection: always`` publishes every applied mutation if that is what you want.

**GET /reef/harness returns 404.** Nothing has been published yet, or the scenario's recipe serves no files. The catalog at ``GET /reef/harness/releases`` lists what exists.

**reef-pi captures no receipts, so report has nothing to send.** The installed ``reef-client`` is older than 0.2.0 and reads a header the service no longer sends. ``pip install -U "reef-client>=0.2.0"``.

**reef-<adapter> exits with no Reef URL in the tree's model binding files.** The wrapper finds Reef through the file the adapter's model binding renders its endpoint into: ``pi-agent/models.json``, ``opencode/opencode.json``, ``claude/settings.json``, ``dsh/profiles/headless/cordis.patch.yml`` (and ``dsh/profiles/web/cordis.patch.yml`` for ``reef-dsh web``), ``hermes/config.yaml``, ``native/models.json``. The published tree carries no endpoint on purpose; write the binding with your Reef URL there before running the wrapper. A codex tree (``codex/config.toml``) is rewritten the same way; codex calls Reef's ``/v1/responses`` route.

Docs and links
--------------

**A docs link points at GitHub instead of a page.** Only files under
``docs/*.rst`` are site pages; historical RFCs, examples, and code are linked
on GitHub, where they are read.
