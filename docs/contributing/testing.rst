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

CI runs source and installed-wheel tests on Python 3.10, 3.11, and 3.12 with
four pytest workers. Tests in the same file stay in one worker, preserving
module fixture reuse. In an activated development environment, install the same
test runner plugin and reproduce the parallel run:

.. code:: bash

   uv pip install pytest-xdist==3.8.0
   GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
     pytest tests/ -n 4 --dist loadfile

Use ``-n 0`` for a serial run when diagnosing a failure. Tests in different
files may run at the same time; use temporary directories and dynamically
allocated ports for their external resources.

CI installs dependencies with ``uv pip`` and caches downloads and built wheels
separately for each job and Python version. It still creates a fresh installed
environment on each runner, including the CPU-only package boundary checks.

Run one area
------------

Most tests under ``tests/reef_service`` need no GPU:

.. code:: bash

   pytest tests/reef_service/test_reef_artifacts.py -q

Markers
-------

``unit``, ``integration``, and ``acceptance``. Run one with ``pytest -m
<marker>``.

Coverage
--------

CI measures coverage on Python 3.12, combining all workers' results, and
``[tool.coverage.report] fail_under`` in
``pyproject.toml`` is a gate: the run exits non-zero when total coverage falls
below the floor. Reproduce it the way CI does:

.. code:: bash

   pytest tests -n 4 --dist loadfile --cov=reef --cov-report=term

``pytest-cov`` ships in the ``dev`` extra. The floor applies to the whole
package, so a partial run reports far less than CI does; measure against the
full suite before reading a number as a regression. ``[tool.coverage.run]``
omits the four Megatron modules that only execute inside a live CUDA worker.
