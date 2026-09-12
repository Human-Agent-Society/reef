Train model weights from agent feedback
=======================================

Weight training coordinates the driver, model workers and HTTP service. Reef
hot-swaps each accepted update into the serving engine, so inference keeps
answering across the update.

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
inside it, mounting the model directory ``inference.model-path`` points at and the
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

The cookbook ``recipes/sao/examples/sao/serve.yaml`` requests one actor GPU
and one rollout GPU through Slime flags. Reef manages the shared Ray runtime,
and Slime schedules its model workers there. Its ``run.sh`` defaults the local
Ray pool to two visible devices; an external cluster controls its own pool.
The model at ``inference.model-path`` must be present or downloadable.

Start from a config
-------------------

Each weight-training example ships a complete ``serve.yaml`` that starts the
services in the required order and manages the shared Ray runtime. SAO omits
``services``: Reef assembles the Slime driver and HTTP process, discovers the
inference connection through the bridge, and stops the stack on exit. Custom
stacks can still declare processes explicitly. Copy the closest example and edit it.

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

   inference.model-path | a local HF model directory or a repo id, downloaded on start
   recipe.implementation | the recipe this deployment serves; its fields live in ``recipe.config``
   service.token | the bearer token the service accepts
   training.config.num_gpus | example-specific GPU count passed to Slime topology flags; some examples set the flags directly
   training.config.global_batch_size | samples in one optimizer step
   training.config.checkpoint_dir | where checkpoints land, with the ``storage.artifact-*`` paths
   training.options | GPU layout, optimizer, sequence length, loss settings

Three things to get right:

1. **Batch sizes must agree.** A recipe's ``batch_size`` must equal
   ``training.config.global_batch_size``. A mismatch leaves a partial optimizer batch
   or makes the driver reject the update.
2. **The recipe and the loss flags must describe the same objective.** The driver
   checks this at startup. Each recipe page names its loss family, and
   `Loss families <../developer-guide/loss-families.rst#family-to-driver-flags>`__
   maps each one to its flags.
3. **The serving backend must capture training tensors.** Weight training needs
   the exact token ids, loss mask, and rollout log probabilities produced during
   inference. The bundled SGLang training backend records them in
   ``response.training``.

Keep the ``training.options`` mapping from the closest working config and change only what
your model or recipe needs.

Run the example
---------------

.. code:: bash

   export REEF_TOKEN=reef-local     # the token recipes/sao/examples/sao/serve.yaml declares

   reef serve -c recipes/sao/examples/sao/serve.yaml \
     --inference.model-path ~/models/Qwen2.5-1.5B-Instruct

Any config value can be overridden on the command line. Startup takes several
minutes; wait for the driver and HTTP service to report ready.

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

Add these native flags to ``training.options`` (without leading ``--``),
and size the engine's adapter table for the scenarios you will train, plus
one slot for the revision being published. The training thread takes turns
between scenarios; each keeps its own adapter and optimizer state, and a restart
recovers every one of them. ``/reef/status`` lists each scenario with its
``adapter_runtime_load_id``.

An adapter the scenario did not train, such as one from an offline SFT run,
is admitted the same way. It enters the release chain only if
``adapter_config.json`` declares a
``peft_type``, the weights are present, and its base model matches the one the
engine holds.

What a published adapter contains
---------------------------------

Every accepted update publishes a directory holding the adapter alone, never a
merged or full-model checkpoint. This is what makes saving every update
affordable: a rank-32 adapter is a few megabytes where the base model is
gigabytes. ``checkpoint_every_n_versions`` can therefore stay at 1, and the
release chain keeps every revision instead of only some of them.

.. code:: text

   adapter_config.json          standard PEFT config: peft_type, r, lora_alpha,
                                target_modules, lora_dropout, base_model_name_or_path
   adapter_model.safetensors    the adapter tensors, and only those
   reef-adapter.json            Reef's training metadata for this revision

The first two files are a plain Hugging Face PEFT adapter, so a published
revision loads outside Reef with nothing but ``transformers`` and ``peft``:

.. code:: python

   from peft import PeftModel
   from transformers import AutoModelForCausalLM

   base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
   model = PeftModel.from_pretrained(base, "/path/to/checkpoint-4")

``reef-adapter.json`` is what lets you check a revision, not just load it. It
records:

- the base model, plus a checksum of its tokenizer and chat-template files;
- the PEFT settings the export wrote;
- the dtype of the tensors it wrote;
- the scenario and step that produced it;
- a SHA-256 for each of the two PEFT files.

PEFT loaders ignore files they do not recognize, so this one does not affect
loading the adapter elsewhere.

Reef checks this file whenever it is present. A checksum that no longer matches
its file, a setting that contradicts ``adapter_config.json``, or a listed file
that is missing all cause Reef to refuse the adapter instead of serving weights
it cannot account for. Adapters from elsewhere have no such file and are
accepted without one; a deployment that should serve only its own exports can
require it.

Connect your agent
------------------

Use the `HTTP API reference <../reference/http-api.rst>`__ for inference,
feedback reports, and release queries. To compare learning signals before
choosing a training config, see `Choose a recipe for agent learning
<recipes.rst>`__.
