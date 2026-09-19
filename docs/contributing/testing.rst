Testing
=======

Reef's tests are one repository suite. Most of it runs without a GPU; the parts
that import the training runtime need the supported container.

PostgreSQL record tests require ``uv pip install -e '.[postgres]'`` and a dedicated
test database. Set ``REEF_TEST_POSTGRES_URL`` to its PostgreSQL URL, then run
``pytest tests/reef_service/test_record_store_contract.py tests/reef_service/test_postgres_records.py``.
Each test creates and drops a randomly named ``reef_test_*`` schema, so the test
role needs schema-creation permission. Without the variable these integration
cases skip; with it configured, connection or driver errors fail the tests.
CI supplies PostgreSQL 16 and runs these cases on all supported Python versions.

Run the full suite
------------------

Managed deployment recovery also has opt-in CPU tests using real Ray processes:

.. code:: bash

   REEF_TEST_RAY=1 PYTHONPATH="$PWD:$PWD/tests" NO_PROXY='*' \
     .venv/bin/python -m pytest tests/reef_service/test_training_restart.py \
       tests/reef_service/test_model_supervision_ray.py -q

Install the Python dependencies from ``.[slime]`` first. These tests use private
local Ray clusters and fake CPU weights. They inject process death and check
durable publication, readiness, child cleanup and HTTP endpoint reconnection;
they do not validate Slime's real GPU checkpoint or collective transport.

LoRA and colocated contract tests run in the regular CPU suite:

.. code:: bash

   .venv/bin/python -m pytest tests/slime_backend/test_driver_runtime_env.py \
     tests/slime_backend/test_rollout_recovery.py \
     tests/slime_backend/test_sglang_engine.py \
     tests/reef_service/test_multi_scenario_bridge.py -q

These cover independent component selection, inference-owned startup offload,
paired memory transitions, cold reconstruction of scenario adapters and the
commit barrier, including retaining the frozen base between training steps.
The memory fixture rejects publication without resident weights and training
while inference KV/graphs remain resident. It models ordering, not GPU capacity;
real LoRA IPC/NCCL transport, CUDA memory use and combined-mode performance
require a supported GPU run.

.. code:: bash

   pytest tests/

Run it in the supported container environment. Many torch-dependent tests use
``pytest.importorskip`` and skip when torch is unavailable. Others, including
``tests/reef_service/test_slime_bridge.py``, import Slime and torch during
collection. Without the training dependencies, pytest cannot collect the full
suite.

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

Tests in the same file stay in one worker, preserving
module fixture reuse. In an activated development environment, install the same
test runner plugin and reproduce the parallel run:

.. code:: bash

   uv pip install pytest-xdist==3.8.0
   GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
     pytest tests/ -n 8 --dist loadfile

Use ``-n 0`` for a serial run when diagnosing a failure. Tests in different
files may run at the same time; use temporary directories and dynamically
allocated ports for their external resources.

CI installs dependencies with ``uv pip`` and caches downloads and built wheels
separately for each job and Python version. It still creates a fresh installed
environment on each runner, including the CPU-only package boundary checks.
The source and sandbox suites can restore older download caches for the same
OS, architecture, and Python version after workflow or dependency edits; uv
still resolves and installs the requested versions.

The ``test (3.12)`` check gates completion of both matrices. A failed,
cancelled, or skipped matrix cannot pass it. The same gate combines and
validates coverage.

Run one area
------------

Most tests under ``tests/reef_service`` need no GPU:

.. code:: bash

   pytest tests/reef_service/test_reef_artifacts.py -q

Markers
-------

``unit``, ``integration``, and ``acceptance``. Run one with ``pytest -m
<marker>``.

``sandbox`` selects tests that need real nested bubblewrap jails. Reproduce
the two source-suite groups with ``pytest tests -m 'not sandbox' -n 8 --dist
loadfile`` and ``REEF_REQUIRE_SANDBOX=1 pytest tests -m sandbox``. The second
command requires a Linux host that supports nested jails.

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

   pytest tests -n 8 --dist loadfile --cov --cov-report=term

``pytest-cov`` ships in the ``dev`` extra. The floor applies to the whole
package, so a partial run reports far less than CI does; measure against the
full suite before reading a number as a regression. ``[tool.coverage.run]``
omits the four Megatron modules that only execute inside a live CUDA worker.
