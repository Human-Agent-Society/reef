Installation
============

Harness evolution and a recording proxy run on a laptop; weight training needs
the GPU environment.

Laptop, no GPU
--------------

Enough to serve, record, report, and evolve a harness against a hosted model.

.. code:: bash

   pip install reef-infra        # the import package is `reef`
   git lfs install

or, to work on Reef itself:

.. code:: bash

   git clone https://github.com/Human-Agent-Society/reef.git && cd reef
   pip install -e .
   git lfs install

``git lfs`` is required: Reef keeps version history in Git, with weight files
under LFS. This install runs the base ``recipe`` kind and ``harness_evolve``.

A client that only talks to a Reef deployment needs neither the repository nor
this install — just the stdlib-only wire client:

.. code:: bash

   pip install reef-client

Weight training
---------------

Weight training runs Ray, a Slime driver, and Reef together, against
CUDA-specific builds of torch, SGLang, Megatron-Core, and FlashAttention. The
supported way to get them is the image:

.. code:: bash

   docker build -f docker/Dockerfile.reef -t reef \
     --build-arg REEF_VERSION="$(git describe --tags | sed s/^v//)" .

`docker/README.md <../docker/README.md>`__ covers the GPU prerequisites. Then
get inside it, mounting the model directory ``reef.model_path`` points at and
the directory Reef keeps its state in:

.. code:: bash

   docker run --gpus all --network host --ipc host --shm-size 32g -it \
     -v ~/models:/root/models \
     -v ~/reef-run:/var/lib/reef \
     -v "$PWD":/workspace/Reef \
     reef bash

``--network host`` lets the Slime driver, SGLang, and Reef reach each other on
localhost; ``--ipc host --shm-size 32g`` is what the training stack needs for
shared memory. ``recipes/openclawrl/examples/openclawrl/run.sh`` runs the same
invocation non-interactively.

Inside that container, to install from source instead of using what the image
already carries:

.. code:: bash

   git submodule update --init --recursive
   pip install -e ".[slime]"
   pip install --no-deps --group runtime

Keep ``--no-deps``: the CUDA-specific packages are already in the image, and
resolving them again replaces working builds. The ``slime`` extra installs only
the Python-side dependencies; the ``runtime`` group pins the training runtime
itself.

What the bundled example needs
------------------------------

``recipes/sao/examples/sao/serve.yaml`` declares ``training.num_gpus: 2`` and
``cuda_visible_devices: "0,1"``. ``training.num_gpus`` must match the devices
you actually expose, and the model at ``reef.model_path`` must be present or
downloadable.

Harness evolution also needs the coding-agent binary its adapter drives — for
``pi``, ``npm i -g @earendil-works/pi-coding-agent@0.84.2`` — and an
OpenAI-compatible endpoint for the model under test.

See also
--------

- `Basic concepts <concepts.rst>`__ — run the loop end to end.
- `Evolve your harness <evolve-your-harness.rst>`__ — the no-GPU lane.
- `Evolve your model <evolve-your-model.rst>`__ — the weight-training lane.
