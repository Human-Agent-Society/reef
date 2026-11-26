Train model weights from agent feedback
=======================================

Train model weights with Slime on your own GPUs. Reef records agent calls,
uses feedback to build training jobs, and serves the checkpoints you select.

For hosted LoRA training from a CPU host, use `Tinker <tinker.rst>`__.
To change rules, skills or tools while keeping the model fixed, use
`harness evolution <evolve-your-harness.rst>`__.

Before you start
----------------

Build and start the `GPU training image
<../getting-started/installation.rst#gpu-image-for-weight-training>`__.
Mount persistent model and checkpoint directories as described in
`docker/README.md <../../docker/README.md>`__.
The base installation has no Slime, Ray or CUDA training stack.

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

``/healthz`` checks HTTP readiness; ``/reef/status`` shows jobs and errors.
Exiting ``reef serve`` stops the managed stack.

Watch it learn
--------------

Send inference and feedback as in the `quickstart
<../getting-started/quickstart.rst>`__. Once enough eligible samples arrive,
Reef submits a job. Slime trains it, and each saved checkpoint becomes an
artifact. Activation decides whether it serves, using evaluation when configured.

.. code:: bash

   export SCENARIO=sao-smoke
   curl -sS -H "Authorization: Bearer reef-local" \
     "http://127.0.0.1:8900/reef/scenarios/$SCENARIO/artifacts"

Look for an artifact with ``operation: training``. Configure
`checkpoint cadence <operate.rst#set-the-checkpoint-cadence>`__ for intermediate saves.

Slime continues from its latest saved checkpoint, including checkpoints that
Reef did not select. It refuses ``train-from`` on an existing scenario.
Activating an older artifact changes serving only. A Slime stack trains one
scenario for its lifetime; use another stack and ports for another scenario.

What to review
--------------

Keep the example's GPU layout, loss flags and batch sizes until it works.
Review these changes for your workload:

- Set ``inference.model-path`` and persistent checkpoint/storage paths.
- Set required ``pipeline.sample2job.config.job_samples``. This counts stored
  samples; rendering can produce more optimizer samples (`SFT <recipes/sft.rst>`__).
- Match the recipe to its `Slime loss flags
  <../developer-guide/loss-families.rst#family-to-slime-flags>`__; mismatches fail startup.
- Match training and serving chat-template arguments. For disabled thinking,
  use ``training.options.apply-chat-template-kwargs: '{"enable_thinking": false}'``.
  Unrenderable samples do not train and increment ``unrenderable_samples``.

See `training options <../reference/configuration-training.rst>`__ for batch
and backend settings. ``pipeline.batch_size`` is invalid.

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

The artifact's ``adapter/`` directory holds PEFT's ``adapter_config.json``
and ``adapter_model.safetensors``, plus Reef metadata in ``reef-adapter.json``.
Transformers and PEFT can load the first two files directly.

When Reef metadata is present, its checksums and settings must match.
External adapters can omit it unless the deployment requires it.
See `artifact layouts <../reference/http-api-artifacts.rst#import-an-artifact>`__.

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

The default agent is ``terminus-2``. Each verifier result becomes a report
with its inference receipts; the recipe decides what trains.
Use ``--help`` for agent, instruction and tagging options.

For container-to-host calls, ``--agent-host host.docker.internal`` makes the
capture proxy listen on all interfaces. Configure Docker's host gateway on Linux.
