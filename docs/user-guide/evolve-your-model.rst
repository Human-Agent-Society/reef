Train model weights from agent feedback
=======================================

Weight training runs three processes. Reef hot-swaps each accepted update into
the serving engine, so inference keeps answering across the update.

+--------------------------+-----------------------------------------------+
| Process                  | Owns                                          |
+==========================+===============================================+
| Ray and the Slime driver | GPUs, the optimizer, checkpoints              |
+--------------------------+-----------------------------------------------+
| Reef                     | requests, records, the release chain          |
+--------------------------+-----------------------------------------------+
| Engine                   | SGLang, serving the current weights           |
+--------------------------+-----------------------------------------------+

When it fits
------------

Use this lane when the model itself has to change. If your recipe evolves a
harness or another text artifact, see `Evolve your harness
<evolve-your-harness.rst>`__.

Before you start
----------------

Weight training needs the supported GPU environment: Ray, a Slime driver, and
CUDA-specific builds of torch, SGLang, and Megatron. ``pip install -e .`` from
the quickstart brings none of them; build the image as described in
`Installation <../getting-started/installation.rst#gpu-image-for-weight-training>`__, then get
inside it, mounting the model directory ``reef.model_path`` points at and the
directory Reef keeps its state in:

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

The cookbook ``recipes/sao/examples/sao/serve.yaml`` declares
``training.num_gpus: 2`` and ``cuda_visible_devices: "0,1"``.
``training.num_gpus`` must match the devices you actually expose, and the
model at ``reef.model_path`` must be present or downloadable.

Start from a config
-------------------

Each weight-training example ships a complete ``serve.yaml`` that starts the
three processes in the required order. Copy the closest one and edit it.

- `SAO rollout training <recipes/sao.rst>`__ uses
  ``recipes/sao/examples/sao/serve.yaml``, the smallest: two GPUs, one actor
  with the critic colocated on it, one rollout engine.
- `TTT-Discover test-time training <recipes/tttd.rst>`__ uses
  ``recipes/tttd/examples/tttd/serve.yaml``, two GPUs with LoRA training.
- `OpenClaw-RL conversation learning <recipes/openclawrl.rst>`__ uses
  ``recipes/openclawrl/examples/openclawrl/serve.yaml``, the paper's seven-GPU
  stack with a PRM engine and a student model.

`Configuration <../reference/configuration.rst>`__ covers interpolation, command-line
overrides, and log locations.

What to review
--------------

.. config::

   reef.model_path | a local HF model directory or a repo id, downloaded on start
   reef.recipe | the recipe this deployment serves. Recipe fields such as ``batch_size`` sit beside it
   reef.token | the bearer token the service accepts
   training.num_gpus | GPUs handed to Ray and Slime, with ``training.cuda_visible_devices``
   training.global_batch_size | samples in one optimizer step
   training.checkpoint_dir | where checkpoints land, with the ``reef.artifact_*`` paths
   training.slime_flags | GPU layout, optimizer, sequence length, loss settings

Three things to get right:

1. **Batch sizes must agree.** A recipe's ``batch_size`` must equal
   ``training.global_batch_size``. A mismatch leaves a partial optimizer batch
   or makes the driver reject the update.
2. **The recipe and the loss flags must describe the same objective.** The driver
   checks this at startup. Each recipe page names its loss family, and
   `Loss families <../developer-guide/loss-families.rst#family-to-driver-flags>`__
   maps each one to its flags.
3. **The serving backend must capture training tensors.** Weight training needs
   the exact token ids, loss mask, and rollout log probabilities produced during
   inference. The bundled SGLang training backend records them in
   ``response.training``.

Keep the ``slime_flags`` from the closest working config and change only what
your model or recipe needs.

Run the example
---------------

.. code:: bash

   export REEF_TOKEN=reef-local     # the token recipes/sao/examples/sao/serve.yaml declares

   reef serve -c recipes/sao/examples/sao/serve.yaml \
     --reef.model_path ~/models/Qwen2.5-1.5B-Instruct

Any config value can be overridden on the command line. Startup takes several
minutes; wait for all three services to report ready.

.. code:: bash

   curl -f http://127.0.0.1:8900/healthz
   curl -H "Authorization: Bearer reef-local" http://127.0.0.1:8900/reef/status

``/healthz`` says the HTTP service is up. Read ``/reef/status`` when something
is wrong: it reports asynchronous training and preload failures, current weight
versions, inference admission, processor state, and checkpoint storage.

Watch it learn
--------------

Send inference through Reef and report feedback per response, exactly as in
`Quickstart <../getting-started/quickstart.rst>`__. When the recipe has enough eligible records
it builds a batch, asks Slime for an optimizer step, checkpoints the result,
publishes it to the engine, and records a new version.

.. code:: bash

   export SCENARIO=<the x-reef-scenario you sent>   # examples/sao uses sao-smoke

   curl -sS -H "Authorization: Bearer reef-local" \
     http://127.0.0.1:8900/reef/scenarios/$SCENARIO/releases
   # {"scenario": "...", "releases": [{"operation": "training", "current": true, ...}, ...]}

You are training when a new ``training`` entry appears, which happens once
``batch_size`` eligible records have arrived. Later requests use the current
version without a restart. Keep ``checkpoint_every_n_versions`` at its default
of ``1`` unless you want versions without durable checkpoints. Between
checkpoints, weight bytes live only in engine memory and a restart falls back to
the last checkpoint.

A full-weight deployment handles one training scenario for its lifetime. To
train another, run another stack on different ports.

Gate a candidate
----------------

By default a successful training step is published. To measure the exported
checkpoint first, add an ``evaluation`` section. The fields are in
`Configuration <../reference/configuration.rst#the-evaluation-section>`__, the plugin
interface and a worked example in `Write a recipe
<../developer-guide/write-a-recipe.rst#gate-a-candidate>`__.

Several scenarios on one base model
-----------------------------------

One stack can serve and train one LoRA adapter per scenario against a single
frozen base model. Nothing changes per request: send ``x-reef-scenario`` and
Reef routes to that scenario's own adapter revision.

.. code:: text

   --megatron-lora-rank=32 --megatron-lora-alpha=32
   --megatron-lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
   --max-loaded-loras=3

Add those to the ``slime-driver`` entry's command in your config's ``services``
list, and size the engine's adapter table for the scenarios you will train, plus
one slot for the revision being published. The training thread takes turns
between scenarios; each keeps its own adapter and optimizer state, and a restart
recovers every one of them. ``/reef/status`` lists each scenario with its
``adapter_runtime_load_id``.

An adapter the scenario did not train, such as one from an offline SFT run,
is admitted the same way. It enters the release chain only if
``adapter_config.json`` declares a
``peft_type``, the weights are present, and its base model matches the one the
engine holds.

Connect your agent
------------------

Use the `HTTP API reference <../reference/http-api.rst>`__ for inference,
feedback reports, and release queries. To compare learning signals before
choosing a training config, see `Choose a recipe for agent learning
<recipes.rst>`__.
