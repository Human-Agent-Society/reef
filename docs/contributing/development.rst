Development
===========

Set up a checkout and check your change before review. Use the
`codebase map <codebase-structure.rst>`__ to find the owning package.

Setup
-----

Use ``uv`` and Python 3.12 or newer. CI uses 3.12.
Install Git LFS for artifact and checkpoint tests.
Reuse an existing ``.venv`` or set up a new checkout with these commands.

.. code:: bash

   git submodule update --init --recursive
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -e ".[dev]" -e ./third_party/reef-client
   pre-commit install

For Slime work, use the supported environment in
`docker/README.md <https://github.com/Human-Agent-Society/reef/blob/main/docker/README.md>`__
and install its dependencies:

.. code:: bash

   uv pip install -e ".[slime]"
   uv pip install --no-deps --group runtime

The ``slime`` extra supplies adapter dependencies; ``runtime`` pins Slime.
Keep ``--no-deps`` to preserve the container's CUDA-compatible stack.

Checks
------

Activate ``.venv`` and start with the smallest test that exercises the change:

.. code:: bash

   .venv/bin/python -m pytest tests/artifact/test_reef_artifacts.py -q

Use the `test directory map
<https://github.com/Human-Agent-Society/reef/blob/main/tests/AGENTS.md>`__
to place tests and find their fixtures. Each test must pass alone and in any
order; keep files in ``tmp_path`` and allocate ports dynamically.

Before requesting review for Python changes, run:

.. code:: bash

   pre-commit run --all-files
   .venv/bin/python -m mypy
   GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
     .venv/bin/python -m pytest tests/ -n 8

Full collection needs training dependencies, even when optional GPU checks
skip. The Git settings isolate Git LFS tests. Use ``-n 0`` to diagnose a
failure serially. The `CI workflow
<https://github.com/Human-Agent-Society/reef/blob/main/.github/workflows/ci.yml>`__
defines the test environment and suite selections. Report checks that could
not run.

Integration checks
------------------

Hosted-provider tests under `tests/smoke/
<https://github.com/Human-Agent-Society/reef/tree/main/tests/smoke>`__
list their opt-in variables and credential requirements.
For real inference engines, follow `Integration qualification
<../developer-guide/serving-runtimes.rst#integration-qualification>`__.
These checks require their own environments; the CPU suite does not validate
real engine behavior.

Documentation
-------------

Use Node.js 22. Run these commands from the repository root.

.. code:: bash

   cd docs/site
   npm ci
   npm run check:docs
   npm run lint
   npm run build

The checks validate links and documented contracts. Inspect changed pages
with ``npm run dev`` after editing their structure or layout.
