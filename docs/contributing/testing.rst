Testing
=======

Reef's tests are one repository suite. Most of it runs without a GPU; the parts
that import the training runtime need the supported container.

Run the full suite
------------------

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
