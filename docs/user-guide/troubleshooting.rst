Troubleshooting
===============

Check ``GET /reef/status``, the response body and the affected service's log
under ``reef.run-dir`` (default ``.reef/run/``).

Starting Reef
-------------

- **Missing config:** ``-c`` paths are relative to the working directory.
- **Unexpected values:** run ``reef serve ... --print-config``. Flags and
  ``REEF_*`` variables can override the file. Duplicate, conflicting or
  invalid keys prevent startup; see `configuration <../reference/configuration.rst>`__.
- **Readiness timeout:** check downloads, GPUs and ``inference.log``.
  Deadlines default to 30 seconds for HTTP and 3600 for managed engines/training.
  Training can extend the HTTP deadline.
- **Credential found in a harness tree:** remove and rotate it. Trees enter
  Git; use ``inference.upstream-api-key`` or the adapter's ``api_key_env``.

Requests
--------

.. list-table::
   :header-rows: 1

   * - Status
     - Check
   * - 401
     - Match the Bearer token to ``reef.token`` or ``reef.tokens``.
   * - 400, missing scenario
     - Supply ``x-reef-scenario`` or a scenario-specific route.
   * - 404
     - Create the scenario before inference. Check its ``served`` mapping and inference backend.
   * - 503, startup
     - Poll ``/reef/scenarios/{scenario}/inference/status``. Fix failed startup and resend ``served``.
   * - 409, repeated abort
     - Check the engine log; Reef already retried once.
   * - 502
     - Check the engine/provider error in the response body.
   * - 504
     - Check ``inference.timeout-s`` (default 300 seconds).

Provider 4xx errors retain their status, including context-length errors.
See `inference errors <../reference/http-api-inference.rst#request>`__.
Reports can implicitly create scenarios; a misspelled name can create another.
``POST /reef/train`` never creates one.

Reports and training
--------------------

**Accepted report, no training:** inspect ``job_data_ready``, ``processor`` and
training errors in status. Check ``job_samples``, group completeness and
``GET /reef/scenarios/{scenario}/contract``. Consumed or stale samples do not
become fresh training data. Reports cannot override eligibility.

**Rejected report:** 400 means the report violates the recipe's schema; 409 means its
``agent_record_id`` already names different content. Resend unchanged content
or use another ID. See the `recipe guide <recipes.rst>`__.

**Failed job:** read its traceback. Match the recipe's loss flags.
Configure W&B through ``observability.wandb``; Slime tracking flags are rejected.

**Full checkpoint storage:** free space or adjust the
`checkpoint retention limits <operate.rst#set-the-checkpoint-cadence>`__.
Retention preserves the latest checkpoint and the job's start.
An ``unowned checkpoint assets`` error identifies unrelated files to move out.

**Repeated work after restart:** only saved checkpoints preserve progress.
Unsaved optimizer work can run again. See `recovery
<operate.rst#restart-and-recovery>`__.

**Slime rejects train-from:** use a new deployment from the desired checkpoint;
see `Train from an artifact <operate.rst#train-from-an-artifact>`__.
Changing activation changes serving only.

Harness evolution
-----------------

- **No step:** check ``training_mode``, ``job_samples`` (default 1) and training errors.
- **Missing binary:** install the named prerequisite, such as npm, or set
  ``evolution.binary``. Otherwise Reef installs the pinned agent on first use.
- **No published change:** inspect ``candidate_runs`` and the step page in
  ``GET /reef/harness/artifacts``. Ties and unchanged trees publish nothing.
  Check model tool-call support and ``evolution.evaluation.task_timeout_s``
  (default 600). See `History <evolve-your-harness.rst#history>`__.
- **Harness 404:** check the scenario, seed and artifact list. Empty trees
  expose no files until something is published.
- **Missing receipts:** run ``uv pip install --upgrade reef-client``.
  Old clients can read an obsolete receipt header.
- **Missing scenario header:** raw trees have no endpoint binding. Install
  with ``reef-client harness connect`` or use the `client form
  <../reference/http-api-harness.rst#harness-artifacts>`__.
- **Sync conflict:** resolve changed files and sync again. ``sync --discard``
  drops local edits.
- **Scenario prompt:** name the installation, for example
  ``reef-client harness run code-repair``.
- **Files kept after sync:** a training step can block imports. The next
  run/sync retries. Post-session errors appear after ``after the last session:``.

Docs and links
--------------

Only navigated RST pages render on this site. Code, examples and other
repository documents open on GitHub.
