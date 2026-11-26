Train model weights from agent feedback
=======================================

Use Slime to train on your GPUs and serve saved checkpoints. For hosted LoRA
training, use `Tinker <tinker.rst>`__. To change an agent's files instead,
use `harness evolution <evolve-your-harness.rst>`__.

Run the example
---------------

Start the `GPU training image
<../getting-started/installation.rst#gpu-image-for-weight-training>`__ with
persistent model and checkpoint directories. The base installation has no
Slime or CUDA stack. From the checkout inside the container:

.. code:: bash

   export REEF_TOKEN=reef-local
   reef serve -c recipes/sao/examples/imo_answerbench/serve.yaml \
     --inference.model-path ~/models/Qwen2.5-1.5B-Instruct

This SAO example needs two GPUs. In another terminal, check startup:

.. code:: bash

   curl -f http://127.0.0.1:8900/healthz
   curl -H "Authorization: Bearer reef-local" http://127.0.0.1:8900/reef/status

Follow the `quickstart
<../getting-started/quickstart.rst>`__ scenario and feedback steps with this running
deployment. Use ``model: active``. Reef submits a job when enough eligible
samples arrive. Each saved checkpoint becomes an artifact. Activation decides
whether it serves, using evaluation when configured.

.. code:: bash

   curl -sS -H "Authorization: Bearer reef-local" \
     http://127.0.0.1:8900/reef/scenarios/hello-reef/artifacts

Look for ``operation: training``. See `checkpoint cadence
<operate.rst#set-the-checkpoint-cadence>`__ and `activation
<operate.rst#activation-rule>`__ to control saving and serving.

Adapt the configuration
-----------------------

`Choose a recipe <recipes.rst>`__ for your feedback. Start with its working
configuration and preserve its GPU layout, loss flags and batch sizes.

- Set ``inference.model-path`` and persistent checkpoint paths.
- Set ``pipeline.sample2job.config.job_samples``. This counts stored samples;
  rendering can produce more optimizer samples.
- Match training and serving chat templates. Unrenderable samples do not train.

See `training settings <../reference/configuration-training.rst>`__ for options
and the recipe's README for its complete run.

Slime trains one scenario per stack. It continues from its latest saved checkpoint,
including inactive checkpoints. Activating an older artifact changes serving only.
Slime refuses ``train-from`` on an existing scenario; use a `new deployment
<operate.rst#train-from-an-artifact>`__ to resume an earlier checkpoint.

Train a LoRA adapter
--------------------

Use the `TTT-Discover example <../../recipes/tttd/examples/tttd/README.md>`__
for Slime LoRA. Reef's SGLang integration serves adapters; its vLLM integration
serves full weights only.

``inference.max-loaded-adapters`` defaults to 1, which pauses generation during
replacement. A second slot lets the new adapter load beside the old one.
Imported adapters must declare ``peft_type``, contain weights and match the
serving base model. See `artifact imports
<../reference/http-api-artifacts.rst#import-an-artifact>`__.
