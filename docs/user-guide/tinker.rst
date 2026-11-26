Train with Tinker
=================

Train LoRA adapters from a CPU host. Tinker trains and samples remotely;
Reef keeps records, feedback and artifacts. No local GPU stack is required.

Install and run
---------------

You need Python 3.12+, Git LFS and a Tinker account with model access.
In an activated environment, run from the checkout:

.. code:: bash

   uv pip install -e '.[tinker]'
   export TINKER_API_KEY="..."
   export REEF_TOKEN=reef-local
   export REEF_TINKER_STATE_DIR="$PWD/work/tinker-smoke"
   reef serve -c tutorials/tinker/serve.yaml

After ``/healthz`` is ready, run in another terminal with the same environment:

.. code:: bash

   python tutorials/tinker/smoke.py

This paid smoke run generates four responses and trains one step with synthetic
rewards. Run it once per fresh state directory. It checks integration, not
learning quality. See the `example <../../tutorials/tinker/README.md>`__ for setup.

For another deployment, set ``training.backend: tinker`` and a supported
``inference.model-path``. Keep ``training.options.state-dir`` persistent and
exclusive to one runtime. Reef downloads tokenizer resources, not model weights.

``api-key-env``, ``project-id`` and ``base-url`` select credentials, project and
service. A checkpoint belongs to its originating service; Reef rejects a
checkpoint from another service. Spindle has no checkpoint archive API, so its
LoRA checkpoints cannot use Reef's inference engines. Its base-model sampler
starts a sampler pool on first use.

Default ``inference.serving: trainer`` supports one training scenario per
deployment. Engine serving supports several adapters on one base model.
See `Training settings <../reference/configuration-settings.rst#training>`__.

What the backend supports
-------------------------

Inference requests
~~~~~~~~~~~~~~~~~~

Use ``/v1/chat/completions`` with text ``system``, ``user`` and ``assistant``
messages. Assistant prefill is supported; tools, multimodal messages and
Anthropic routes are rejected.

Supported options are ``max_tokens`` or ``max_completion_tokens``, ``temperature``,
``top_p``, ``top_k``, ``seed``, ``stop`` and ``chat_template_kwargs.enable_thinking``.
Only one output is allowed. Unknown options fail explicitly.

Without a token limit, generation uses the remaining context window, or 1024
tokens if the service states no window. A full prompt returns 400
``context_length_exceeded``. Reasoning appears in ``reasoning_content`` for
responses. Streaming buffers the complete reply, then emits OpenAI SSE with
its reasoning and answer. It provides no token-by-token latency.

Match ``training.options.chat_template_kwargs`` to serving. Unrenderable or
over-context samples do not train; ``unrenderable_samples`` counts them.

Losses
~~~~~~

.. list-table::
   :header-rows: 1

   * - Objective
     - Requirements
   * - ``importance_sampling``
     - Advantages and rollout log probabilities; ``kl-coef: 0``.
   * - ``sft``
     - Target tokens; no advantages or rollout log probabilities. Set ``kl-coef: 0``.
   * - `TTTD <recipes/tttd.rst>`__
     - Its recipe supplies the objective, including frozen-base KL configured by ``kl-coef``.

SAO and OpenClaw-RL are not implemented. Unknown loss families fail before
training. See `objective extensions
<../developer-guide/recipe-training-service.rst#job-parameters>`__ for custom losses.

Job settings
~~~~~~~~~~~~

Under ``training.options``, set:

- ``batch-size``: samples per optimizer step, default 1.
- ``lora-rank`` and ``seed``: adapter initialization.
- ``learning-rate``: Adam rate, unless the recipe selects a schedule.

Adam otherwise uses SDK defaults: beta1 0.9, beta2 0.95, epsilon 1e-12,
no weight decay or gradient clipping.

``pipeline.sample2job.config.job_samples`` controls job submission. Complete record
sets and trajectory rendering can produce more optimizer samples.
A job's ``scheduling`` controls epochs, shuffle and its final partial batch;
see `Job parameters <../developer-guide/recipe-training-service.rst#job-parameters>`__.

``training.timeout-s`` bounds SDK futures and transport; ``inference.timeout-s``
bounds sampling. A timeout does not cancel an accepted remote update.

Task jobs
~~~~~~~~~

The `generated-task recipe
<../reference/configuration-generated-tasks.rst#generated-tasks>`__ submits Harbor tasks.
``rollout`` selects the agent and sandbox; ``rollout.rollouts_per_task`` sets
attempts per task. Rollouts use the job's start weights and Tinker's sampler.
Resumed jobs reuse saved rollouts.

Jobs, checkpoints, and recovery
-------------------------------

A new scenario serves the base model. The first job creates its adapter from
``seed``. With ``seed: null``, Reef saves and reuses one untrained adapter
under ``<state-dir>/base/``.

Jobs restore weights and optimizer state. They save training state and sampler
weights at job end, or also at ``recipe.config.job_parameters.save_interval``
optimizer steps. The next job uses the last checkpoint regardless of activation.

Artifacts contain ``tinker-checkpoint.json``, not weights. Keep their remote
checkpoints and originating account, project and service available. Local
retention never deletes remote checkpoints.

A restart resumes from the last save. An uncertain API result can consume
credits or leave an unreferenced checkpoint; retry uses a fresh session.
See `Train-from and recovery <operate.rst#train-from-an-artifact>`__ for changing starts.

Serve on inference engines
--------------------------

Hosted Tinker adapters can serve on GPU-backed SGLang engines:

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

Engine settings select ``inference.serving: engines``. ``training.colocate`` is
rejected. Reef downloads selected checkpoints, converts them to PEFT and caches
them under ``<state-dir>/sampler-weights/``.

Set ``max-loaded-adapters`` at least as high as the scenario count. One extra
slot lets an update load beside the served adapter; otherwise it replaces it.
Match the base model, tokenizer and chat-template arguments. SGLang must support
``lm_head`` adapters. Restart reloads served adapters before requests resume.

This topology has stub-engine tests only; no real SGLang run has validated it.

Serve from a remote GPU host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a GPU host with Reef and SGLang installed:

.. code:: bash

   export REEF_REMOTE_CONTROL_TOKEN=...
   export REEF_REMOTE_INFERENCE_TOKEN=...
   python -m reef.inference.remote --port 8000

Use different tokens. Set the same variables on the Reef host and add this
to the engine configuration above:

.. code:: yaml

   inference:
     remote:
       url: https://gpu-host.example.com

Reef starts no local component. The GPU host downloads the base model and
receives selected adapters. Protect the control token like a host login;
it can start model code and replace weights. Use HTTPS.

The component accepts one controlling runtime. Stopping that runtime stops
its engines. After a component restart, Reef detects it within 30 seconds
and reloads adapters. See `remote settings
<../reference/configuration-settings.rst#inference>`__ and the command's ``--help``.

Train all the parameters on Spindle
-----------------------------------

Use ``spindle_full`` to train a Hugging Face checkpoint on a
`Spindle <https://github.com/modal-projects/spindle>`__ full-parameter engine.
Before starting Reef:

- Deploy the engine with the exact base repository and 40-digit commit SHA.
- Supply the Spindle API key through ``TINKER_API_KEY`` or ``api-key-env``.
- Supply an ``HF_TOKEN`` that reads the base, plus Modal credentials
  (``reef-infra[modal]`` and a profile or ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET``).
- In Spindle's Modal environment, give the ``hf-secret`` secret (default
  ``huggingface-secret``) an ``HF_TOKEN`` that reads the base and writes ``release-repo``.

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

Each checkpoint exports weights, tokenizer and templates to a new
``release-repo`` commit. Reef stores references and restores optimizer state
for later training. Default serving uses Spindle; Reef's engines can load
the exported revision.

Imports must contain the base's files or its full-parameter manifest from the
same service with a full-weight reference. Other weights return HTTP 400.
Use another engine and scenario to train another base.

Budget for GPU startup and weight export on every job. Only one live 4B smoke
run has validated this backend; it measured no learning quality.

Score centering
~~~~~~~~~~~~~~~

For ``spindle_full`` task jobs, enable score centering to correct policy drift
from bounded importance weights (`method <https://arxiv.org/abs/2609.20807>`__):

.. code:: yaml

   training:
     options:
       score-centering: true
       score-centering-top-k: 128
       score-centering-weight-low: 1.0
       score-centering-weight-high: 1.0
       score-centering-min-tail-mass: 1.0e-6

The default bounds 1 and 1 give REINFORCE; 0 and 2 give truncated importance
sampling. The correction is zero under the full importance ratio.

- ``importance_sampling`` task jobs use Spindle's centered ``cispo`` loss.
  Recorded or uploaded samples lack the required sampler top-k and fail.
- Sample at temperature 1 without truncation. Other temperatures or explicit
  ``top_p``/``top_k`` return 400. Harbor's ``terminus-2`` sets none by default.
- SFT is unchanged. Other policy losses and nonzero ``kl-coef`` are rejected.
- Spindle must implement ``topk_sample_logprobs`` and centered ``cispo``.
  Missing support fails the job. Inspect ``score_centering:sum`` in its metrics.
