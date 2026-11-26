Install Reef on a laptop or GPU server
======================================

Reef requires Python 3.12 or newer. Choose an installation for your workload:

- **Harness updates with a hosted model:** `Laptop, no GPU`_.
- **Local weight training:** `GPU image, for weight training`_.
- **Connect to an existing deployment:** `Client only`_.

Laptop, no GPU
--------------

Install Git LFS, then install a released version of Reef:

.. code:: bash

   pip install reef-infra
   git lfs install

To use the examples or develop Reef, install from a checkout:

.. code:: bash

   git clone https://github.com/Human-Agent-Society/reef.git
   cd reef
   pip install -e .

The distribution is ``reef-infra``; Python imports use ``reef``.
Git stores snapshot history, and Git LFS stores large artifact files.
The checkout includes ``recipes/`` and ``tutorials/``; the wheel does not.

GPU image, for weight training
------------------------------

For local Slime training, build the supported image from the checkout root:

.. code:: bash

   docker build -f docker/Dockerfile.reef -t reef .

The image includes the compatible CUDA, inference and training dependencies.
See `docker/README.md <../../docker/README.md>`__ for prerequisites and startup.

To update the source installation inside the container, run:

.. code:: bash

   git submodule update --init --recursive
   pip install -e ".[slime]"
   pip install --no-deps --group runtime

Keep ``--no-deps`` to preserve the image's CUDA packages.
The ``slime`` extra installs adapter dependencies. The ``runtime`` group pins
the training runtime.

Client only
-----------

To connect a harness to an existing deployment, install the client:

.. code:: bash

   pip install reef-client

The client uses the Python standard library and does not install the Reef service.

Run your first request
----------------------

- `Inference and feedback quickstart <quickstart.rst>`__: record a request
  and submit feedback.
- `Evolve your model <../user-guide/evolve-your-model.rst>`__: train with
  local GPUs or Tinker.
