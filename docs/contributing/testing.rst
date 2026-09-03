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

CI runs the suite under coverage, and ``[tool.coverage.report] fail_under`` in
``pyproject.toml`` is a gate: the run exits non-zero when total coverage falls
below the floor. Reproduce it the way CI does:

.. code:: bash

   pytest tests --cov=reef --cov-report=term

``pytest-cov`` ships in the ``dev`` extra. The floor applies to the whole
package, so a partial run reports far less than CI does; measure against the
full suite before reading a number as a regression. ``[tool.coverage.run]``
omits the four Megatron modules that only execute inside a live CUDA worker.
