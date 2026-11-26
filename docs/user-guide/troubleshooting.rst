Troubleshooting
===============

Start with ``GET /reef/status`` and the affected service's log under
``reef.run-dir`` (default ``.reef/run/``). Match the symptom below, then check
the response body for the specific cause.

Starting Reef
-------------

**The config cannot be found.** Relative ``-c`` paths use the current directory.
Pass an absolute path or run from the directory containing the file.

**A setting differs from the file.** Run ``reef serve ... --print-config``.
It shows resolved values and their sources without starting the deployment.
Check command-line overrides and ``REEF_*`` variables. Duplicate keys,
conflicting spellings and null values for required fields prevent startup.

**A service never becomes ready.** Read its log and check its ``ready`` probe.
The usual deadlines are 30 seconds for Reef HTTP and 3600 seconds for managed
engines or training stacks. Training can extend the HTTP deadline.
Check downloads, GPU availability and engine startup errors in ``inference.log``.

**Startup names an invalid setting.** Unknown settings are errors, not ignored
options. Check the `settings reference <../reference/configuration.rst>`__.
Harness options belong under ``recipe.config.evolution``.

**Startup finds a credential in a harness tree.** Reef refuses literal keys
and credential-shaped strings because snapshot trees are stored in Git.
Rotate exposed credentials and remove them from the tree.
Use ``inference.upstream-api-key`` or the adapter's ``api_key_env`` instead.
Candidate patches have the same restriction.

Requests
--------

**401: invalid token.** Check ``Authorization: Bearer`` against ``reef.token``
or ``reef.tokens``. Without Authorization, Reef also accepts ``x-api-key``.
Empty expanded token values are dropped. A deployment with no tokens accepts
requests without authentication.

**400: missing scenario.** Send ``x-reef-scenario`` for inference, reports and
harness reads, or use an API path that identifies the scenario.

**404: unknown scenario or no served model.** Create the scenario before inference
with ``POST /reef/scenarios`` or a harness install. Check
``GET /reef/scenarios`` and its ``settings`` response.
An empty ``served`` mapping, ``serves: false`` or ``inference.backend: none``
leaves it without serving.

Reports can create missing scenarios when implicit creation is enabled.
A typo can therefore create another scenario. ``POST /reef/train`` never creates one.

**503: model startup or selected weights unavailable.** Poll
``GET /reef/scenarios/{scenario}/inference/status``.
``inference_starting`` includes ``Retry-After: 30``.
For failed startup, fix the reported cause and resend the ``served`` mapping.
See `Inference <split-services.rst#inference>`__ for ownership and retry behavior.

**409: the engine repeatedly aborts.** Reef rereads routing state and retries once.
Check the engine log if the second attempt also aborts.

**502: inference failed.** Read the engine message in the response body.
The engine may be unreachable, return 5xx or send an invalid response.
Provider 4xx errors retain their status, including context-length errors.

**504: inference timed out.** The default ``inference.timeout-s`` is 300 seconds.
Increase it when valid generations need more time:

.. code:: bash

   reef serve ... --inference.timeout-s 1800

Reports and training
--------------------

**A report was accepted but training does not start.** Check:

- The trigger: enough ``job_samples`` or a complete recipe-specific group.
- ``job_data_ready`` and ``processor`` in ``/reef/status``.
- The scenario's contract at ``GET /reef/scenarios/{scenario}/contract``.
- Training errors, consumed receipts and samples marked ``stale``.

Reports cannot override training eligibility. Missing references are rejected
when the report arrives.

**400 on a report.** The body violates the recipe's schema, such as a missing
score or invalid metadata. Check the `recipe guide <recipes.rst>`__.

**409 on a report.** The same ``agent_record_id`` already has different content.
Use a new ID or resend the original content unchanged.

**A training job failed.** Read its traceback in the service log.
Configure W&B through ``observability.wandb``; Slime's tracking flags are refused.
The recipe's loss family must match its Slime loss flags.

**Checkpoint storage is full.** Slime preserves the latest checkpoint and the
job's start. Check these limits in ``training.options``:

- ``reef-checkpoint-max-storage-fraction`` (default 0.8) or ``reef-checkpoint-max-storage``.
- ``reef-checkpoint-min-free-space-fraction`` (default 0.1) or ``reef-checkpoint-min-free-space``.

Retention uses ``reef-checkpoint-policy``: ``latest`` or ``best_reward``.
Free space or adjust the limits. If the error names ``unowned checkpoint assets``,
move unrelated files out of the checkpoint directories.

**A job repeats work after restart.** Jobs resume from saved checkpoints.
Unsaved optimizer work can run again. A Slime job that saved its checkpoint
before stopping records that checkpoint without retraining. Tinker resumes after
its last saved checkpoint.

**Slime refuses train-from.** Slime continues its workers' latest checkpoint.
Activation can change serving, but train-from returns 400.
See `Train from an artifact <operate.rst#train-from-an-artifact>`__ for starting
a new deployment from an earlier Slime checkpoint.

Harness evolution
-----------------

**Reports arrive but no step starts.** Check ``training_mode`` and
``pipeline.sample2job.config.job_samples`` (default 1).
Both successful and failed reports count. Inspect training errors for invalid
trajectory data or missing inputs.

**The harness binary cannot install.** Install the tool named in the error,
such as npm for pi. Otherwise, set ``evolution.binary`` to an existing executable.
Without that setting, the first job installs the adapter's pinned binary.

**No change is published.** Inspect ``candidate_runs`` and the step page in
``GET /reef/harness/artifacts``. A tie or unchanged tree keeps the current tree.
For failures, check the binary, tool-call support and
``evolution.evaluation.task_timeout_s`` (default 600 seconds).
``policy: none`` publishes valid candidates without evaluation.
See `History <evolve-your-harness.rst#history>`__.

**GET /reef/harness returns 404.** The scenario may be missing, have an empty
seed without a published tree, or expose no files. Read the error message.
Create the scenario explicitly and inspect ``GET /reef/harness/artifacts``.

**No receipts are captured.** Upgrade older clients:
``uv pip install --upgrade 'reef-client>=0.2.0'``.
Versions before 0.2.0 read an obsolete receipt header.

**A pulled agent omits x-reef-scenario.** Raw tree files contain no endpoint
binding. Install with ``reef-client harness connect`` or request
``?form=client&service_url=<reef>&root=<directory>``.
Custom clients must supply the scenario and service token.

**Sync reports conflicting file changes.** Your changes and the published tree
changed the same content. Your files remain untouched. Resolve the conflict,
then sync again. ``sync --discard`` discards your local changes.

**Run asks for a scenario.** Several scenarios are installed.
Name one, for example ``reef-client harness run code-repair``.

**Run or sync says it kept your files.** A training step prevents importing
your changes now. The next run or sync retries them.
A refused post-session upload appears after ``after the last session:``.

Docs and links
--------------

Code, examples and repository documents open on GitHub.
Only RST pages included in the documentation navigation render as site pages.
