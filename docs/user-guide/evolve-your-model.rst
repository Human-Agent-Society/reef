Train model weights from agent feedback
=======================================

Train model weights with Slime on your own GPUs. Reef records agent calls,
uses feedback to build training jobs, and serves the checkpoints you select.

For hosted LoRA training from a CPU host, use `Tinker <tinker.rst>`__.
To change rules, skills or tools while keeping the model fixed, use
`harness evolution <evolve-your-harness.rst>`__.

Before you start
----------------

Build the `GPU training image
<../getting-started/installation.rst#gpu-image-for-weight-training>`__.
A base Reef installation does not include Slime, Ray or the CUDA training stack.
Mount persistent directories for the model and deployment state:

.. code:: bash

   docker run --gpus all --network host --ipc host --shm-size 32g -it \
     -v ~/models:/root/models \
     -v ~/reef-run:/var/lib/reef \
     -v "$PWD":/workspace/Reef \
     reef bash

The model at ``inference.model-path`` must be present or downloadable.
The container's host network lets Reef, Ray and inference reach each other.

Start from a config
-------------------

Copy the closest working ``serve.yaml``. Keep its training options until
both the model and recipe work.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Example
     - Configuration
     - GPUs
   * - `SAO <recipes/sao.rst>`__
     - ``recipes/sao/examples/imo_answerbench/serve.yaml``
     - 2
   * - `TTT-Discover <recipes/tttd.rst>`__
     - ``recipes/tttd/examples/tttd/serve.yaml``
     - 4, with LoRA
   * - `OpenClaw-RL <recipes/openclawrl.rst>`__
     - ``recipes/openclawrl/examples/openclawrl/serve.yaml``
     - 7, including the judge

Run the example
---------------

From the checkout in the container:

.. code:: bash

   export REEF_TOKEN=reef-local
   reef serve -c recipes/sao/examples/imo_answerbench/serve.yaml \
     --inference.model-path ~/models/Qwen2.5-1.5B-Instruct

Startup can take several minutes. In another terminal, check readiness:

.. code:: bash

   curl -f http://127.0.0.1:8900/healthz
   curl -H "Authorization: Bearer reef-local" http://127.0.0.1:8900/reef/status

``/healthz`` checks the HTTP service. ``/reef/status`` reports training errors,
preload failures and the current jobs. ``reef serve`` stops the managed stack
when it exits. See `Configuration <../reference/configuration.rst>`__ for
command-line overrides and log locations.

Watch it learn
--------------

Send inference and feedback as in the `quickstart
<../getting-started/quickstart.rst>`__. Once enough eligible samples arrive,
Reef submits a job. Slime trains it and saves checkpoints. Reef evaluates
candidates and publishes selected artifacts.

.. code:: bash

   export SCENARIO=sao-smoke
   curl -sS -H "Authorization: Bearer reef-local" \
     "http://127.0.0.1:8900/reef/scenarios/$SCENARIO/artifacts"

A new ``training`` artifact confirms that a job produced a published checkpoint.
Activation decides whether later requests use it. Check
`checkpoint cadence <operate.rst#set-the-checkpoint-cadence>`__ when a job
needs intermediate checkpoints.

Slime continues from its latest saved checkpoint, including checkpoints that
Reef did not select. It refuses ``train-from`` on an existing scenario.
Activating an older artifact changes serving only. A Slime stack trains one
scenario for its lifetime; use another stack and ports for another scenario.

What to review
--------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Setting
     - Check
   * - ``inference.model-path``
     - The local model path or Hugging Face repository ID.
   * - ``recipe.implementation``
     - The recipe to run; its settings are under ``recipe.config``.
   * - ``reef.token``
     - The token clients send to Reef.
   * - ``pipeline.sample2job.config.job_samples``
     - The stored samples required to submit a job.
   * - ``training.config.global_batch_size``
     - The optimizer batch size in examples that expose this setting.
   * - ``training.options``
     - Slime's GPU layout, optimizer, sequence length and loss flags.
   * - Checkpoint and ``storage.artifact-*`` paths
     - Persistent storage with room for checkpoints and exports.

A weight recipe requires ``job_samples``. Set it to the example's batch size
for its usual one-step jobs. Trajectory rendering can produce more optimizer
samples than stored samples; see `SFT <recipes/sft.rst>`__.
``pipeline.batch_size`` is not a valid setting.

The recipe and Slime loss flags must agree. Reef checks this at startup.
Use the recipe's example or the `loss-family mapping
<../developer-guide/loss-families.rst#family-to-slime-flags>`__.

Match training and serving chat-template arguments. For example, set
``training.options.apply-chat-template-kwargs`` to
``'{"enable_thinking": false}'`` when serving disables thinking.
Samples that cannot render do not train; ``unrenderable_samples`` counts them.

Choose what serves
------------------

The default activation rule serves each new artifact. Choose ``evaluated``
to require a score threshold, or ``never`` for manual activation.
See `Activation rule <operate.rst#activation-rule>`__.
Training continues from its latest checkpoint under all three rules.

Train a LoRA adapter
--------------------

To train an adapter on a frozen base model, add these Slime options:

.. code:: yaml

   training:
     options:
       megatron-lora-rank: 32
       megatron-lora-alpha: 32
       megatron-lora-target-modules: [linear_qkv, linear_proj, linear_fc1, linear_fc2]

Use ``inference.backend: sglang``. Reef's vLLM integration serves full weights
only. Each published artifact contains the adapter, and SGLang loads it under
a new name.

``inference.max-loaded-adapters`` defaults to 1. With one slot, generation
pauses while the old adapter is replaced. With two slots, the new adapter
can load beside it. For several scenarios on one base model, use
`Tinker with inference engines <tinker.rst#serve-on-inference-engines>`__.

Reef also accepts imported adapters. Their configuration must declare
``peft_type``, their weights must exist, and their base model must match serving.

What a published adapter contains
---------------------------------

The artifact's ``adapter/`` directory contains:

- ``adapter_config.json``: the PEFT configuration.
- ``adapter_model.safetensors``: the adapter tensors.
- ``reef-adapter.json``: training metadata and checksums.

The first two files load outside Reef with Transformers and PEFT:

.. code:: python

   from peft import PeftModel
   from transformers import AutoModelForCausalLM

   base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
   model = PeftModel.from_pretrained(base, "/path/to/snapshot/adapter")

When ``reef-adapter.json`` exists, Reef checks its files, checksums and settings.
A mismatch refuses the adapter. External adapters can omit this metadata unless
the deployment requires it. PEFT ignores the extra file.

Connect your agent
------------------

Use the `HTTP API <../reference/http-api.rst>`__ for inference and feedback.
`Choose a recipe <recipes.rst>`__ compares the supported learning signals.

Run Harbor tasks and report
---------------------------

With Docker available, run a task manifest through Reef:

.. code:: bash

   python -m reef.harness.harbor.rollouts \
     --reef-url http://127.0.0.1:8900 --scenario my-agent --model qwen3.8:27b \
     --manifest tasks/manifest.json --tasks-root tasks --side train

Each task runs with ``terminus-2`` unless ``--agent-json`` selects another agent.
Reef records its model calls. The verifier result becomes one report with the
score, inference receipts and task metadata. The recipe decides what trains.

- ``--instructions FILE`` appends instructions to each task.
- ``--label name=value`` tags calls and reports to distinguish experimental arms.
- ``--agent-host host.docker.internal`` lets an agent inside the container reach
  the capture proxy. The proxy then listens on all interfaces. On Linux,
  configure Docker's ``host-gateway`` for that name.
