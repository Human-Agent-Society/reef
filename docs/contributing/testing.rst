Testing
=======

Reef's tests are one repository suite. Most of it runs without a GPU; the parts
that import the training runtime need the supported container.

Find and place tests by the responsibility they cover. The
`test directory map <https://github.com/Human-Agent-Society/reef/blob/main/tests/AGENTS.md>`_
defines the component, recipe, service, and smoke suites. Read it before adding
or moving a test. Check placement with::

    python .github/scripts/check_test_layout.py

Pytest uses importlib mode. Import shared helpers through ``tests.support`` or
another scoped ``tests.<owner>`` module, so test packages do not shadow libraries.

PostgreSQL record tests require ``uv pip install -e '.[postgres]'`` and a dedicated
test database. Set ``REEF_TEST_POSTGRES_URL`` to its PostgreSQL URL, then run
``pytest tests/storage/test_record_store_contract.py tests/storage/test_postgres_records.py``.
Each test creates and drops a randomly named ``reef_test_*`` schema, so the test
role needs schema-creation permission. Without the variable these integration
cases skip; with it configured, connection or driver errors fail the tests.
CI supplies PostgreSQL 16 and runs these cases on all supported Python versions.

Run the full suite
------------------

.. code:: bash

   pytest tests/

Run it in the supported container environment. Many torch-dependent tests use
``pytest.importorskip`` and skip when torch is unavailable. Others, including
``tests/slime_backend/test_slime_train_group.py``, import Slime during
collection. Without the training dependencies, pytest cannot collect the full
suite.

Training job and serving contracts run in the regular CPU suite:

.. code:: bash

   .venv/bin/python -m pytest tests/runtime/test_jobs.py tests/train/test_local_jobs.py \
     tests/train/test_jobs.py tests/runtime/test_serving.py -q

These cover job identity, durable job submission and resumption after a
restart, the trainer's job loop, and loading published snapshots into serving
engines. They use fake components and engines. Real Slime training and
engine weight loads require a supported GPU run.

Service process and engine control also have opt-in CPU tests that use real
Ray processes:

.. code:: bash

   REEF_TEST_RAY=1 PYTHONPATH="$PWD" NO_PROXY='*' \
     .venv/bin/python -m pytest tests/reef_service/deploy/test_service_executor_ray.py \
       tests/inference/sglang/test_inference_control.py -q

Install the Python dependencies from ``.[slime]`` first. These tests use
private local Ray clusters and CPU engines. They inject process death and
check service address publication, child cleanup and engine control.

CI runs source and installed-wheel tests on Python 3.12 on
Blacksmith. Source tests use eight workers on an 8-vCPU runner; installed-wheel
tests use four workers on a 4-vCPU runner. The source suite excludes tests marked
``sandbox``; a parallel GitHub-hosted matrix runs those tests serially on the
same Python version. Both matrices collect all of ``tests/`` and use
complementary marker selections, so each test belongs to exactly one group.
The sandbox jobs set ``REEF_REQUIRE_SANDBOX=1``: a missing bubblewrap binary or
failed nested-jail preflight fails CI instead of silently skipping isolation
checks. Local runs still skip these tests on unsupported hosts.

Ready pull requests automatically run these suites after lint passes,
including installed-wheel tests and combined coverage. Drafts run only
lint, type checks, and static Dockerfile checks. Documentation builds and
harness smoke tests run automatically when relevant files change.

New pushes cancel older runs and return to routine coverage. Matrix selection
rejects closed, draft, and superseded PR revisions. Main pushes and manual
workflow dispatches run the full matrix; use a PR workflow rerun to satisfy
that PR's required checks. See the
`maintenance model <https://github.com/Human-Agent-Society/reef/blob/main/.github/MAINTAINER.md#5-continuous-integration>`_.

``pyproject.toml`` selects ``--dist worksteal``: the workers share single
tests, not files, and an idle worker takes queued tests from a busy one. The
``dev`` extra installs the same test runner plugin, so an activated
development environment reproduces the parallel run:

.. code:: bash

   GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
     pytest tests/ -n 8

Use ``-n 0`` for a serial run when diagnosing a failure. Any two tests,
including two tests of the same file, may run at the same time in different
processes, in any order. Each test must pass alone:

- Keep files and other external resources in ``tmp_path`` and allocate ports
  dynamically.
- Do not depend on state that another test sets.
- Use a module-scoped fixture only for read-only setup. Each worker builds
  its own copy.

CI installs dependencies with ``uv pip`` and caches downloads and built wheels
separately for each job and Python version. It still creates a fresh installed
environment on each runner, including the CPU-only package boundary checks.
The source and sandbox suites can restore older download caches for the same
OS, architecture, and Python version after workflow or dependency edits; uv
still resolves and installs the requested versions.

The ``test (3.12)`` check gates completion of both matrices. A failed,
cancelled, or skipped matrix cannot pass it. The same gate combines and
validates coverage.

Installed-wheel selection
-------------------------

``tests/packaging/wheel-tests.txt`` lists the contract modules that also run
against installed packages. Selection is independent of their owner directory.
Update the list when moving a listed module or adding a wheel-compatible
contract. The layout check rejects missing and duplicate entries.

Run these tests outside the checkout, with Reef and its dependencies installed
from wheels. Expose only the checkout's ``tests/`` and ``recipes/`` directories
there, as the CI job does. Do not add the checkout root to ``PYTHONPATH``: its
``reef/`` directory would hide the installed package being tested.

Run one area
------------

Hosted run acceptance is opt-in and creates paid E2B sandboxes and small
OpenRouter model requests. Set ``E2B_API_KEY`` and ``OPENROUTER_API_KEY`` in the
test process (never commit them), then run::

    REEF_TEST_E2B=1 REEF_E2B_TEST_MODEL=deepseek/deepseek-v4.1-flash \
      .venv/bin/python -m pytest tests/smoke/test_e2b_run.py -q

The tests exercise pi and Codex through the ordinary executor factory, model
traffic through a localhost tunnel, trajectory and residue read-back, immutable
inputs, timeout cleanup, and harness evaluation and GEPA scoring of remote
runs. The selected model must support
both Chat Completions and Responses through OpenRouter and be available to the
account in its region. Provider credentials remain in a local test HTTP server;
only a synthetic prompt and a run token enter the sandbox. Default CI skips
these paid tests; ``test_e2b_executor.py`` and ``test_e2b_tunnel.py`` cover the
contracts and failure paths without provider access.

The Modal sandbox has its own opt-in acceptance, which costs a few cents of
Modal CPU time and needs Modal credentials (``MODAL_TOKEN_ID`` and
``MODAL_TOKEN_SECRET``, or ``~/.modal.toml``)::

    REEF_TEST_MODAL=1 .venv/bin/python -m pytest tests/smoke/test_modal_sandbox.py -q

It builds an image for a small npm program and checks one jailed process: the
tunnel to this host, the workspace copy in and back, the read-only filesystem
outside it, and the timeout. ``tests/sandbox/test_modal_executor.py`` covers the
settings, the sandbox lifecycle and the launch without provider access.

The native inference check is a command, not a pytest test, so
``pytest tests/`` and CI do not run it (`Integration qualification
<../developer-guide/executors.rst#integration-qualification>`__ describes
what it checks). It needs one unused GPU and a Reef GPU environment with
SGLang, vLLM, Torch and Transformers installed. It writes its own small
checkpoints, so it downloads no model. From the checkout, run one profile
into a new directory outside it::

    python -m tests.smoke.native_inference --profile features --native-engine sglang \
      --output /scratch/native-features-sglang

``--native-engine`` selects the engine behind Reef's inference component.
``--profile`` selects what the run checks:

- ``features``: traffic splits, pins, rollouts, logprobs, records and the
  prefix cache of a published artifact.
- ``routing``: both engines behind a Semantic Router. It also needs a
  Semantic Router binary; ``--help`` lists its options.
- ``recovery``: the health watcher restores the component's engines after
  the run removes them with the component's authenticated control
  operation, once before the first activation and once while they serve an
  artifact. The run kills no process, restarts no component, and does not
  check the retry of a failed weight load.
- ``lifecycle``: changes to what a scenario serves, through Reef's public
  API, service application and serving worker. Two inference components of
  the native engine serve two full-weight checkpoints at the same time: by
  name, with requests that overlap, in a 50/50 split, and through a
  promotion and a rollback that load nothing. The run stops one checkpoint,
  waits until its component is free and serves it again there. Reef must
  refuse a third checkpoint. Last, the run removes the engines of one
  component, as ``recovery`` does: the health watcher restores its
  checkpoint while the other one answers. With two harness artifacts on one
  provider model, the run then serves both at the same time: by name, in a
  50/50 split, and with one at weight 0; then it stops serving one.

Both components of ``lifecycle`` share the one GPU, and the small
checkpoints ignore their context. The cached-token counts therefore show
only that a load empties a component's prefix cache and that a kept
checkpoint keeps it. The run measures no performance, and it shows no
isolation between GPUs or hosts.

``recovery`` starts only the native engine. Each other profile also starts
the other engine as a plain OpenAI server; ``lifecycle`` stops its
components first. To cover both integrations, run
each profile with each native engine. The command exits with
status 0 when every check passes. A run saves ``report.json``, process logs,
requests and stored records in the output directory. Check the report for
failures, and ``error.txt`` for a traceback when present. Invalid arguments
are reported before engines start. The run stops only its own processes.

Choose a component suite or one test module. For example, artifact contracts
need no GPU:

.. code:: bash

   pytest tests/artifact/ -q
   pytest tests/artifact/test_reef_artifacts.py -q

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

CI measures coverage on Python 3.12. Each suite uploads its raw coverage data;
the final gate requires both artifacts, combines their results, and produces
``coverage.xml``. Relative source paths allow data from the two runner
checkouts to refer to the same files. The suites defer the coverage threshold
to that combined report; neither partial report is a coverage gate. The
``[tool.coverage.report] fail_under`` in
``pyproject.toml`` is a gate: the run exits non-zero when total coverage falls
below the floor. Reproduce it the way CI does:

.. code:: bash

   pytest tests -n 8 --cov --cov-report=term

``pytest-cov`` ships in the ``dev`` extra. The floor applies to the whole
package, so a partial run reports far less than CI does; measure against the
full suite before reading a number as a regression. ``[tool.coverage.run]``
omits the four Megatron modules that only execute inside a live CUDA worker.
