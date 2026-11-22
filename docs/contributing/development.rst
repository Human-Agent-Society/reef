Development
===========

:For: contributors
:Before you start: a Reef checkout, ``uv``, Git LFS, and Python 3.12 or newer
:You will have: a development environment with the checks that run in CI

Setup
-----

.. code:: bash

   git submodule update --init --recursive
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -e ".[dev]" -e ./third_party/reef-client
   pre-commit install

Training-related work also needs the training runtime:

.. code:: bash

   uv pip install -e ".[slime]"
   uv pip install --no-deps --group runtime

The ``slime`` extra installs the Python-side dependencies of
``reef/train/slime_backend/``. The ``runtime`` dependency group pins the
reviewed training-runtime commit. Keep ``--no-deps``: GPU development uses the
CUDA-compatible dependencies that the container already holds. `Installation
<../getting-started/installation.rst#gpu-image-for-weight-training>`__
describes the container.

``reef-client`` is a separate distribution, maintained at
`Human-Agent-Society/reef-client <https://github.com/Human-Agent-Society/reef-client>`__.
It implements Reef's wire protocol with the standard library only, does not
import ``reef``, and does not ship in the ``reef`` wheel. The harness in
``recipes/basic/`` talks to Reef through it. `Codebase structure
<codebase-structure.rst>`__ maps the other packages.

Run entry points as modules or scripts:

.. code:: bash

   python -m reef.deploy --help
   recipes/basic/run.sh

Checks
------

.. code:: bash

   pre-commit run --all-files
   .venv/bin/python -m mypy
   .venv/bin/python -m pytest tests/api -q

Start with the smallest suite that exercises your change; `Testing
<testing.rst>`__ describes the full suite and the opt-in suites. Keep provider
request bodies unchanged and preserve scenario isolation. A change to a public
interface also needs a contract test.
