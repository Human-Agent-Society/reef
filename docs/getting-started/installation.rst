Install Reef on a laptop or GPU server
======================================

The Reef service requires Python 3.12 or newer and Git LFS.

Laptop, no GPU
--------------

For inference through a hosted model or harness evolution, install the release.

.. code:: bash

   git lfs install
   pip install reef-infra

To use the bundled recipes and tutorials, install from a checkout instead.

.. code:: bash

   git clone https://github.com/Human-Agent-Society/reef.git
   cd reef
   pip install -e .

GPU image, for weight training
------------------------------

For local Slime training, build the supported image from the checkout root.

.. code:: bash

   docker build -f docker/Dockerfile.reef -t reef .

See the `GPU setup <../../docker/README.md>`__ for prerequisites and startup.
The `development guide <../contributing/development.rst#setup>`__ covers reinstalling from source inside the container.

Client only
-----------

To connect to an existing deployment, install the client in your agent's environment.

.. code:: bash

   pip install reef-client

Run your first request
----------------------

Follow the `inference and feedback quickstart <quickstart.rst>`__.
