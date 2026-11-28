Train with Tinker
=================

Train LoRA adapters from a CPU host. Tinker trains and samples remotely;
Reef keeps records, feedback and artifacts. No local GPU stack is required.

Install and run
---------------

Use Python 3.12+, Git LFS and a Tinker account with model access.
From the checkout in an activated environment:

.. code:: bash

   uv pip install -e '.[tinker]'
   export TINKER_API_KEY="..."
   export REEF_TOKEN=reef-local
   export REEF_TINKER_STATE_DIR="$PWD/work/tinker-smoke"
   reef serve -c tutorials/tinker/serve.yaml

After ``/healthz`` is ready, run in another terminal with the same environment:

.. code:: bash

   python tutorials/tinker/smoke.py

This paid check generates four responses and trains one step with synthetic rewards.
Run it once per fresh state directory. It checks integration, not learning quality.
See the `example <../../tutorials/tinker/README.md>`__ for the complete setup.

For another deployment, set ``training.backend: tinker`` and a supported model.
Keep ``training.options.state-dir`` persistent and exclusive to one runtime.
Default trainer serving supports one training scenario; engine serving supports
several adapters on one base model.

What the backend supports
-------------------------

Use ``/v1/chat/completions`` with text messages. Tools, multimodal messages,
Anthropic routes, multiple outputs and unknown options are rejected.
Streaming buffers the full reply, then emits its reasoning and answer as SSE;
it provides no token-by-token latency.

Without a token limit, generation uses the remaining context window, or 1024
tokens when the service states no window. A full prompt returns 400
``context_length_exceeded``. Match training and serving chat templates;
unrenderable or over-context samples do not train.

Supported objectives are ``importance_sampling``, ``sft`` and
`TTTD <recipes/tttd.rst>`__. Set ``kl-coef: 0`` for the first two.
SAO and OpenClaw-RL are unsupported. See `training settings
<../reference/configuration-settings.rst#training>`__ for batch size, learning rate
and service selection.

``training.timeout-s`` bounds SDK calls; ``inference.timeout-s`` bounds sampling.
A timeout does not cancel an accepted remote update.

Jobs, checkpoints, and recovery
-------------------------------

New scenarios serve the base model. Jobs restore weights and optimizer state,
save at job end, and optionally save every ``recipe.config.job_parameters.save_interval``
optimizer steps. The next job uses the last checkpoint regardless of activation.

Artifacts contain remote checkpoint references, not weights. Keep the originating
account, project, service and checkpoints available. Local retention does not delete
remote checkpoints. Cross-service checkpoints are rejected.

Restart resumes from the last save. Uncertain API results can consume credits or
leave unreferenced checkpoints. See `train-from <operate.rst#train-from-an-artifact>`__
to change a training start.

Serve on inference engines
--------------------------

Hosted Tinker adapters can serve on SGLang:

.. code:: yaml

   inference:
     model-path: Qwen/Qwen3-8B
     backend: sglang
     num-gpus: 2
     max-loaded-adapters: 2
   training:
     backend: tinker
     options:
       state-dir: /var/lib/reef/tinker
       lora-rank: 32

Engine settings select engine serving. ``training.colocate`` is rejected.
Set adapter slots at least as high as the scenario count; one extra slot allows
updates beside served adapters. Match the base model, tokenizer and chat template.
SGLang must support ``lm_head`` adapters.

Reef downloads checkpoints and converts them to PEFT. Spindle LoRA checkpoints
cannot use this path because Spindle has no checkpoint archive API.
This topology has stub-engine tests only; no real SGLang run has validated it.

Serve from a remote GPU host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a GPU host with Reef and SGLang installed:

.. code:: bash

   export REEF_REMOTE_CONTROL_TOKEN=...
   export REEF_REMOTE_INFERENCE_TOKEN=...
   python -m reef.inference.remote --port 8000

Use different tokens. Supply the same variables on the Reef host and set
``inference.remote.url: https://gpu-host.example.com`` in the engine configuration.
Use HTTPS. Protect the control token like a host login: it can start model code
and replace weights.

The component accepts one controlling runtime. Stopping that runtime stops its
engines. After a component restart, Reef detects it within 30 seconds and reloads
adapters. See `remote settings <../reference/configuration-settings.rst#inference>`__.

Train all the parameters on Spindle
-----------------------------------

Deploy a `Spindle <https://github.com/modal-projects/spindle>`__ full-parameter
engine with the exact base repository and 40-digit commit SHA. Install
``reef-infra[modal]`` and supply Modal credentials, a Spindle API key through
``TINKER_API_KEY``, and ``HF_TOKEN`` with base-model read access.
Spindle's ``huggingface-secret`` must also permit writes to the release repository.

.. code:: yaml

   inference:
     model-path: org/model
   training:
     backend: spindle_full
     options:
       state-dir: /var/lib/reef/spindle-full
       base-url: https://<workspace>--spindle-server.modal.run
       base: hf://org/model@0123456789abcdef0123456789abcdef01234567
       release-repo: org/model-releases
       modal-environment: spindle
       learning-rate: 1.0e-6

Each checkpoint exports weights, tokenizer and templates to a new release commit.
Default serving uses Spindle; Reef's engines can load that revision.
Imports must match the base or contain a full-parameter manifest from the same
service. Other weights return 400. Use a separate engine and scenario per base.

Budget for GPU startup and export on every job. One live 4B smoke run validated
integration; it measured no learning quality.

Optional ``training.options.score-centering: true`` requires importance-sampling
task jobs and Spindle support for centered ``cispo`` and ``topk_sample_logprobs``.
Use temperature 1 without ``top_p`` or ``top_k``. Recorded samples and nonzero
``kl-coef`` are rejected; SFT is unchanged. See `score-centering settings
<../reference/configuration-settings.rst#training>`__.
