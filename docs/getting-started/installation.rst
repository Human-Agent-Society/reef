Install Reef on a laptop or GPU server
======================================

Installation depends on what you want Reef to evolve:

- **A harness, against a hosted model.** Install the package on your
  laptop; no GPU involved. See `Laptop, no GPU`_.
- **Model weights, and optionally a harness.** Set up docker image that
  contains GPU management stack. See `GPU image, for weight training`_.
- **Neither — you only call a Reef deployment.** See `Client only`_.

Laptop, no GPU
--------------

Enough to serve, record, report, and evolve a harness against a hosted model.

For a released version:

.. code:: bash

   pip install reef-infra        # the import package is `reef`

Or, to work on Reef itself:

.. code:: bash

   git clone https://github.com/Human-Agent-Society/reef.git && cd reef
   pip install -e .
   git lfs install

``git lfs`` is required: Reef keeps release history in Git, with weight files
under LFS. The checkout includes the core ``recipe`` example, the paper-backed
methods under ``recipes/``, and the harness evolution demo under
``tutorials/evolve-your-harness/``.

GPU image, for weight training
------------------------------

Weight training runs Ray, a Slime driver, and Reef together, against
CUDA-specific builds of torch, SGLang, Megatron-Core, and FlashAttention. The
supported way to get them is the image:

.. code:: bash

   docker build -f docker/Dockerfile.reef -t reef .

`docker/README.md <../../docker/README.md>`__ covers the GPU prerequisites and
how to start the container; `Evolve your model <../user-guide/evolve-your-model.rst>`__
picks up from there.

Inside the container, to install from source instead of using what the image
already carries:

.. code:: bash

   git submodule update --init --recursive
   pip install -e ".[slime]"
   pip install --no-deps --group runtime

Keep ``--no-deps``: the CUDA-specific packages are already in the image, and
resolving them again replaces working builds. The ``slime`` extra installs only
the Python-side dependencies; the ``runtime`` group pins the training runtime
itself.

Client only
-----------

For a harness that just needs to talk to a Reef deployment, install the
stdlib-only wire client:

.. code:: bash

   pip install reef-client

Run your first request
----------------------

Continue with the `inference and feedback quickstart <quickstart.rst>`__
to start the service, capture a receipt, and submit feedback. If you installed
the GPU image for weight training, follow `Train model weights from agent
feedback <../user-guide/evolve-your-model.rst>`__ for the training stack.
