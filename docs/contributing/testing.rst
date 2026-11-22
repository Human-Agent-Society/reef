Testing
=======

Reef's tests are one repository suite. Most of it runs without a GPU; the parts
that import the training runtime need the supported container.

Place a test by the responsibility it covers. The `test directory map
<https://github.com/Human-Agent-Society/reef/blob/main/tests/AGENTS.md>`_
defines the component, recipe, service and smoke suites. Read it before you add
or move a test, then check placement::

    python .github/scripts/check_test_layout.py

Pytest uses importlib mode. Import shared helpers through ``tests.support`` or
another scoped ``tests.<owner>`` module, so test packages do not shadow
libraries.

Run the full suite
------------------

Run the suite in the supported container, in parallel, as CI does:

.. code:: bash

   GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
     pytest tests/ -n 8

Without the training dependencies, pytest cannot collect the full suite. Many
torch-dependent tests use ``pytest.importorskip`` and skip, but others,
including ``tests/slime_backend/test_slime_train_group.py``, import Slime
during collection.

``pyproject.toml`` selects ``--dist worksteal``: the workers share single
tests, not files, and an idle worker takes queued tests from a busy one. The
``dev`` extra installs ``pytest-xdist``. Use ``-n 0`` for a serial run when you
diagnose a failure.

Any two tests, including two tests of one file, may run at the same time in
different processes, in any order. Each test must pass alone:

- Keep files and other external resources in ``tmp_path`` and allocate ports
  dynamically.
- Do not depend on state that another test sets.
- Use a module-scoped fixture only for read-only setup. Each worker builds
  its own copy.

What CI runs
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Suite
     - How it runs
   * - Source
     - ``pytest tests -m 'not sandbox' -n 8`` on Python 3.12, on an 8-vCPU
       Blacksmith runner, with PostgreSQL 16.
   * - Sandbox
     - ``pytest tests -m sandbox``, serially, on a GitHub-hosted runner with
       ``REEF_REQUIRE_SANDBOX=1``. A missing bubblewrap binary or a failed
       nested-jail preflight fails the job instead of skipping the tests.
       Local runs still skip these tests on unsupported hosts.
   * - Installed wheel
     - The modules of `Installed-wheel selection`_ with ``-n 4``, on a 4-vCPU
       Blacksmith runner.

The source and sandbox suites both collect all of ``tests/`` with complementary
marker selections, so each test belongs to exactly one of them.

- Ready pull requests run these suites after lint passes, with combined
  coverage. Drafts run only lint, type checks and static Dockerfile checks.
- Documentation builds and harness smoke tests run when relevant files change.
- A new push cancels the older run. Main pushes and manual workflow dispatches
  run the full matrix. Use a PR workflow rerun to satisfy that PR's required
  checks.
- The ``test (3.12)`` check completes both matrices. A failed, cancelled or
  skipped matrix cannot pass it. The same check combines and validates
  coverage.

See the `maintenance model
<https://github.com/Human-Agent-Society/reef/blob/main/.github/MAINTAINER.md#5-continuous-integration>`_.

Installed-wheel selection
-------------------------

``tests/packaging/wheel-tests.txt`` lists the contract modules that also run
against installed packages. Selection is independent of their owner directory.
Update the list when you move a listed module or add a wheel-compatible
contract. The layout check rejects missing and duplicate entries.

Run these tests outside the checkout, with Reef and its dependencies installed
from wheels. Expose only the checkout's ``tests/`` and ``recipes/`` directories
there, as the CI job does. Do not add the checkout root to ``PYTHONPATH``: its
``reef/`` directory would hide the installed package.

Run one area
------------

Choose a component suite or one test module. For example, artifact contracts
need no GPU:

.. code:: bash

   pytest tests/artifact/ -q
   pytest tests/artifact/test_reef_artifacts.py -q

Training job and serving contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These run in the regular CPU suite, with fake components and engines:

.. code:: bash

   .venv/bin/python -m pytest tests/train/test_local_jobs.py \
     tests/train/test_jobs.py tests/inference/test_serving.py -q

They cover job identity, durable job submission and resumption after a
restart, the trainer's job loop, and loading published snapshots into serving
engines. Real Slime training and engine weight loads need a supported GPU run.

Opt-in suites
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Suite
     - Needs
     - Covers
   * - Real Ray processes
     - ``REEF_TEST_RAY=1`` and the Python dependencies of ``.[slime]``
     - Service process and engine control on private local Ray clusters with
       CPU engines: injected process death, service address publication,
       child cleanup.
   * - PostgreSQL
     - ``uv pip install -e '.[postgres]'`` and ``REEF_TEST_POSTGRES_URL``, the
       URL of a dedicated test database whose role may create schemas
     - Record storage on PostgreSQL. Each test creates and drops a random
       ``reef_test_*`` schema. Without the variable the cases skip; with it,
       connection or driver errors fail.
   * - E2B
     - ``REEF_TEST_E2B=1``, ``REEF_E2B_TEST_MODEL``, ``E2B_API_KEY`` and
       ``OPENROUTER_API_KEY``. Paid.
     - pi and Codex runs in hosted sandboxes through the ordinary executor
       factory: model traffic through a localhost tunnel, trajectory
       read-back, immutable inputs, timeout cleanup, harness evaluation and
       GEPA scoring.
   * - Modal
     - ``REEF_TEST_MODAL=1`` and Modal credentials (``MODAL_TOKEN_ID`` and
       ``MODAL_TOKEN_SECRET``, or ``~/.modal.toml``). A few cents of CPU time.
     - One jailed process in an image for a small npm program: the tunnel to
       this host, the workspace copy in and back, the read-only filesystem
       outside it, and the timeout.

.. code:: bash

   REEF_TEST_RAY=1 PYTHONPATH="$PWD" NO_PROXY='*' \
     .venv/bin/python -m pytest tests/reef_service/deploy/test_service_executor_ray.py \
       tests/inference/sglang/test_inference_control.py -q

   REEF_TEST_POSTGRES_URL=postgresql://... \
     pytest tests/storage/test_postgres_records.py

   REEF_TEST_E2B=1 REEF_E2B_TEST_MODEL=deepseek/deepseek-v4.1-flash \
     .venv/bin/python -m pytest tests/smoke/test_e2b_run.py -q

   REEF_TEST_MODAL=1 .venv/bin/python -m pytest tests/smoke/test_modal_sandbox.py -q

Notes on the E2B run:

- The model must support both Chat Completions and Responses through
  OpenRouter and be available to the account in its region.
- Provider credentials stay in a local test HTTP server. Only a synthetic
  prompt and a run token enter the sandbox. Never commit the keys.
- Default CI skips these paid tests. ``tests/harness/test_e2b_executor.py``,
  ``tests/harness/test_e2b_tunnel.py`` and
  ``tests/sandbox/test_modal_executor.py`` cover the contracts and failure
  paths without provider access.

Native inference check
~~~~~~~~~~~~~~~~~~~~~~

The native inference check runs Reef's own SGLang and vLLM integrations on a
GPU. It is a command, not a pytest test, so ``pytest tests/`` and CI do not
run it. `Integration qualification
<../developer-guide/serving-runtimes.rst#integration-qualification>`__ lists the
cases that an engine integration must pass. The check needs:

- one unused GPU;
- a Reef GPU environment with SGLang, vLLM, Torch and Transformers. The run
  writes its own small checkpoints, so it downloads no model;
- an absolute path to a new or empty directory outside the checkout, for
  the output;
- for the ``routing`` profile only, a Semantic Router binary
  (``--router-binary``; ``--help`` lists the router options).

From the checkout, run one profile with one native engine::

    python -m tests.smoke.native_inference --profile features --native-engine sglang \
      --output /scratch/native-features-sglang

``--native-engine`` selects the engine behind Reef's inference component.
Each profile except ``recovery`` also starts the other engine as a plain
OpenAI server. To cover both integrations, run each profile with each
native engine. ``--profile`` selects the checks:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Profile
     - What it checks
   * - ``features``
     - A published full-weight artifact with request pinning, beside the
       other engine: traffic splits, requests that name an artifact, a
       rollout request, log probabilities, stored records, and a load
       during a stream.
   * - ``routing``
     - Both engines as plain endpoints behind a Semantic Router that the
       run starts. The native engine answers with its base model, without
       request pinning.
   * - ``recovery``
     - The health watcher replaces engines that the run removes through
       the component's control API. Before the first activation, they
       stay paused; after it, the watcher loads the artifact again.
   * - ``lifecycle``
     - Reef's public API with a real serving worker and two components:
       creation and import, two checkpoints at the same time, a split, a
       promotion and a rollback, retirement and reuse of a component, the
       refusal of a third checkpoint, and watcher recovery. Then two
       harness artifacts that use the same provider model.

The command exits with status 0 only when every check passes. It saves
``report.json``, process logs, requests and stored records in the output
directory. The report lists the failures and the limits of the run;
``error.txt`` holds the traceback of a failed run. The command stops only
the processes that it started.

Each run uses small synthetic checkpoints on one GPU, which the two
``lifecycle`` components share. A run does not show:

- production performance or capacity: no profile measures them. The
  refusal in ``lifecycle`` shows only the limit of its two components;
- isolation between GPUs or hosts;
- recovery from a crashed process, a restarted component or service, or a
  failed weight load;
- training, or the trainer's rollout engines;
- control-plane publication, except in ``lifecycle``: ``features`` and
  ``recovery`` write their routes themselves;
- cache isolation for a real model. The checkpoints ignore their context,
  so ``features``, ``recovery`` and ``lifecycle`` check cached-token counts
  only. These show when an engine reuses its prefix cache and when it
  starts empty.

Markers
-------

``unit``, ``integration``, and ``acceptance``. Run one with ``pytest -m
<marker>``.

``sandbox`` selects tests that need real nested bubblewrap jails. Reproduce
the two source-suite groups with ``pytest tests -m 'not sandbox' -n 8`` and
``REEF_REQUIRE_SANDBOX=1 pytest tests -m sandbox``. The second command requires
a Linux host that supports nested jails.

Coverage
--------

CI measures coverage on Python 3.12. Each suite uploads its raw coverage data,
and the final check combines both into ``coverage.xml``. Only the combined
report is checked against ``[tool.coverage.report] fail_under`` in
``pyproject.toml``: the check fails when total coverage falls below that floor.
Reproduce it the way CI does:

.. code:: bash

   pytest tests -n 8 --cov --cov-report=term

``pytest-cov`` ships in the ``dev`` extra. The floor applies to the whole
package, so a partial run reports far less than CI does. Measure against the
full suite before you read a number as a regression. ``[tool.coverage.run]``
omits the three Megatron modules that execute only inside a live CUDA worker.
