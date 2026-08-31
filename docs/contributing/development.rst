Development
===========

.. page::
   :for: contributors
   :needs: a Reef checkout
   :outcome: a development environment with the checks that run in CI

Setup
-----

.. code:: bash

   git submodule update --init --recursive
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]" -e ./third_party/reef-client
   pip install --no-deps --group runtime
   pre-commit install

The ``runtime`` dependency group installs the reviewed training-runtime commit.
Keep ``--no-deps`` in this command because GPU development uses the
CUDA-compatible dependencies already installed in the container.

Package boundaries
------------------

- The ``slime`` extra installs the Python-side dependencies used by
  ``reef/train/slime_backend/``; the ``runtime`` dependency group pins the
  training runtime itself.
- ``reef-client`` is a separate distribution maintained at
  `Human-Agent-Society/reef-client <https://github.com/Human-Agent-Society/reef-client>`__.
  It implements Reef's wire protocol using only the standard library, does not
  import ``reef``, and does not ship in the ``reef`` wheel. The harness in
  ``recipes/basic/`` talks to Reef through it.

Package ownership and repository-level homes are in `Codebase structure
<codebase-structure.rst>`__.

Run entry points as modules or scripts:

.. code:: bash

   python -m reef.service.deploy --help
   recipes/basic/run.sh

Checks
------

.. code:: bash

   pre-commit run --all-files
   PYTHONPATH=.:third_party/reef-client uv run \
     --with pytest \
     --with aiohttp \
     --with huggingface_hub \
     pytest tests/reef_service -q

Keep provider request bodies unchanged and preserve scenario isolation. A change
to a public interface also needs a contract test.
