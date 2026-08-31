Installation
============

Laptop, no GPU
--------------

Enough to serve, record, report, and evolve a harness against a hosted model.

.. code:: bash

   git clone https://github.com/Human-Agent-Society/reef.git && cd reef
   pip install -e .
   git lfs install

``git lfs`` is required: Reef keeps version history in Git, with weight files
under LFS. This install runs the base ``recipe`` kind and ``harness_evolve``.

Client only
-----------

A client that only talks to a Reef deployment needs neither the repository nor
the install above — just the stdlib-only wire client:

.. code:: bash

   pip install reef-client

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
