Train with Tinker
=================

Train LoRA adapters through Tinker from a CPU host. Reef keeps the records,
feedback and artifact history; Tinker trains and samples remotely.
The default deployment needs no local Ray, Slime or inference engine.

Use `Serve on inference engines`_ for SGLang serving, or
`Train all the parameters on Spindle`_ for full-parameter training.

Install and run
---------------

Use Python 3.12 or newer, Git LFS and a Tinker account with model access.
From an activated environment in the checkout:

.. code:: bash

   uv pip install -e '.[tinker]' -e ./third_party/reef-client
   export TINKER_API_KEY="..."
   export REEF_TOKEN=reef-local
   export REEF_TINKER_STATE_DIR="$PWD/work/tinker-smoke"
   reef serve -c tutorials/tinker/serve.yaml

After ``/healthz`` is ready, run in another terminal with the same environment:

.. code:: bash

   python tutorials/tinker/smoke.py

This paid smoke run generates four short responses and trains one optimizer step
with synthetic rewards. It checks the workflow, not model quality.
Run it once per fresh state directory. See the `smoke guide
<../../tutorials/tinker/README.md>`__ for the complete setup.

For your deployment, set ``training.backend: tinker`` and choose a supported
``inference.model-path``. Reef downloads tokenizer resources, not the model weights.
Keep ``training.options.state-dir`` persistent and exclusive to one runtime.
``api-key-env`` can name another credential variable; ``project-id`` selects a project.
Credentials do not enter resolved configurations or checkpoint manifests.

**Compatible services.** Set ``training.options.base-url`` for a compatible
service such as `Spindle <https://modal.com/docs/guide/spindle>`__.
Otherwise Reef uses hosted Tinker. Training and trainer-based sampling use the
same service.

- A checkpoint path such as ``tinker://<run>/weights/<name>`` resolves only on
  the service that made it. Each checkpoint manifest records the service, and
  the component refuses a checkpoint that another service holds.
- A service may lack some calls. Spindle serves no checkpoint archives, so
  Reef's own engines can serve only hosted Tinker's checkpoints.
- Reef makes a base-model sampling client at its first use (a KL term, a task
  rollout or a request). On Spindle such a client starts a sampler pool.

**Scenarios.** Default ``inference.serving: trainer`` supports one training
scenario per deployment. With Reef's serving engines, several scenarios can
train separate adapters on one base model.

What the backend supports
-------------------------

Inference requests
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Item
     - Support
   * - Route
     - ``/v1/chat/completions`` only. Anthropic routes fail explicitly.
   * - Messages
     - Text ``system``, ``user`` and ``assistant`` messages, including a final
       assistant prefill. Tool calls and multimodal messages fail explicitly.
   * - Options
     - One output per request; ``max_tokens`` or ``max_completion_tokens``,
       ``temperature``, ``top_p``, ``top_k``, ``seed``, ``stop`` and
       ``chat_template_kwargs.enable_thinking``. Unknown options fail
       explicitly.
   * - Reply length
     - Without a token limit, up to the rest of the model's context window, or
       1024 tokens when the service states no window. A prompt that fills the
       window gets a ``400`` whose message starts with
       ``context_length_exceeded``.
   * - Reasoning
     - A thinking model's reasoning is in ``reasoning_content`` and its answer
       in ``content``.
   * - Streaming
     - Buffered: Reef emits the complete response as valid OpenAI SSE events,
       without token-by-token latency. A streamed answer carries only
       ``content``, so a streaming client gets no reasoning.

Match ``training.options.chat_template_kwargs`` to serving, such as
``{enable_thinking: false}``. Unrenderable or over-context samples do not train;
``unrenderable_samples`` counts them. Task jobs retain their own sampled tokens
and log probabilities for training.

Losses
~~~~~~

Select a recipe with a supported Tinker objective:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Loss family
     - Behavior
   * - ``importance_sampling``
     - Built in. Places the trajectory advantage on every response token that
       the loss mask selects.
   * - ``sft``
     - Built in. Trains Tinker's ``cross_entropy`` on the same tokens, so the
       `SFT recipe <recipes/sft.rst>`__ runs on Tinker. Its samples need no
       rollout log probabilities.
   * - TTTD (``recipes/tttd/tinker.py``)
     - Keeps its unmasked policy term and its masked, centered frozen-base KL
       on ``importance_sampling``, as its Slime objective does.
       ``training.options.kl-coef`` sets the penalty. The base sampler scores
       the same token IDs; its probabilities never replace the rollout log
       probabilities.
   * - SAO, OpenClaw-RL
     - Not implemented. An unregistered loss family fails before training.

Policy objectives need one advantage per trajectory. SFT needs no advantages
or rollout log probabilities. An unknown loss family fails before training.
Set ``kl-coef: 0`` for losses without a KL penalty, including ``sft`` and
``importance_sampling``. See `Training service
<../developer-guide/recipe-training-service.rst#job-parameters>`__ for objective extensions.

Job settings
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Setting
     - Meaning
   * - ``training.options.batch-size``
     - Samples in one optimizer step. Default 1.
   * - ``pipeline.sample2job.config.job_samples``
     - The stored samples required to submit a job. Whole record sets and
       trajectory rendering can produce a larger optimizer batch.
   * - ``lora-rank``, ``seed``
     - Adapter initialization.
   * - ``learning-rate``
     - Adam learning rate, until a recipe selects a learning-rate schedule.
       Other Adam settings use the pinned SDK's defaults (beta1 0.9, beta2
       0.95, epsilon 1e-12, no weight decay, no gradient clipping).
   * - ``training.timeout-s``
     - Bounds SDK futures and sets transport timeouts.
   * - ``inference.timeout-s``
     - Bounds sampling.

Job data takes whole record sets, so a job can hold more samples than
``job_samples``. The component cuts it into optimizer steps of the batch size
(`Job size <../developer-guide/recipe-control-service.rst#job-size>`__). A step keeps each
rollout of the objective's ``rollout_unit`` whole: one sample, or one
comparison set. A job's optional ``scheduling`` parameter sets the epochs, the
shuffle, and the ``partial``, ``drop`` or ``error`` policy for the last step.
The timeouts do not cancel a remote update that the service already accepted.

Task jobs
~~~~~~~~~

The `generated-task recipe
<../reference/configuration-generated-tasks.rst#generated-tasks>`__ can submit Harbor tasks.
Each task runs ``rollout.rollouts_per_task`` times at the job's start weights.
The verifier scores each rollout, then the objective trains its task group.

The ``rollout`` job parameter selects the agent and sandbox. A resumed job
reuses saved rollouts. These rollouts use Tinker's sampler directly, so serving
can use a separate runtime.

Jobs, checkpoints, and recovery
-------------------------------

A new scenario initially serves the base model. Its first job creates a LoRA
from ``seed``. With ``seed: null``, Reef saves one untrained adapter under
``<state-dir>/base/`` and reuses it for starts from the base.

Each job restores its start's weights and optimizer state in a new training
session. It saves training state and sampler weights at job end.
Set ``recipe.config.job_parameters.save_interval`` for intermediate checkpoints.
The next job starts from the last checkpoint even if evaluation rejected it.

Published artifacts contain a small ``tinker-checkpoint.json`` manifest, not
weight tensors. Restore, activation and train-from need the remote checkpoints
and access to their originating account, project and service.
Local retention does not delete remote checkpoints. Keep every remote checkpoint
that a Reef artifact references.

- A restart resumes after the last saved checkpoint.
- An uncertain API result can consume credits or leave an unreferenced checkpoint.
  A retry uses a fresh session from the same start, avoiding a repeated uncertain update.
- Requests keep the sampler they selected when admitted.
- Train-from cancels the job continuing the previous start. The next default
  job restores the selected artifact's weights and optimizer state.

Serve on inference engines
--------------------------

Use Reef's SGLang engines to serve hosted Tinker adapters. This requires GPUs
for inference. It does not use local training GPUs.

Set the same base model on both sides:

.. code:: yaml

   inference:
     model-path: Qwen/Qwen3-8B
     backend: sglang
     num-gpus: 2                 # tensor-parallel-size for tensor-parallel engines
     max-loaded-adapters: 2
   training:
     backend: tinker
     options:
       state-dir: /var/lib/reef/tinker
       lora-rank: 32

An engine configuration selects ``inference.serving: engines``.
``reef serve`` starts a local inference component unless ``inference.remote``
names one. ``training.colocate`` is refused because Tinker has no local training GPUs.

**Adapter slots.** The engines start with LoRA serving enabled:
``max-lora-rank`` is ``lora-rank`` and ``lora-target-modules`` is ``all``.
``inference.max-loaded-adapters`` sets ``max-loaded-loras`` and
``max-loras-per-batch``. Each scenario keeps its served adapter resident, so
this value must be at least the number of training scenarios. A scenario's new
adapter replaces its own served adapter when no slot is free.

**Loading.** Reef downloads each selected sampler checkpoint and converts it
to PEFT under ``<state-dir>/sampler-weights/``. It caches each sampler path.
The converter needs torch and Transformers, supplied by the ``tinker`` extra.
A service without checkpoint archives, such as Spindle, cannot use this path.
A restart reloads each served adapter before the scenario serves.

**Compatibility.** Match the base model, tokenizer and chat-template arguments.
Tinker adapters include ``lm_head``; the installed SGLang version must support it.
This topology has only stub-engine tests. A real SGLang run has not validated it.

Serve from a remote GPU host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep Reef and training on a CPU host, and run the inference component on a GPU
host. Selected adapters upload to that component.

On the GPU host, install Reef with SGLang. Set two different tokens and start
the component:

.. code:: bash

   export REEF_REMOTE_CONTROL_TOKEN=...    # starts engines and replaces weights
   export REEF_REMOTE_INFERENCE_TOKEN=...  # sends inference requests
   python -m reef.inference.remote --port 8000

The component starts Ray but waits for engine startup instructions.
Its control routes return 409 until engines start. ``--engine`` defaults to
``sglang`` and limits which backend can start. Use ``--help`` for host and
upload settings.

On the Reef host, set the same two variables and add an ``inference.remote``
section with the component's URL:

.. code:: yaml

   inference:
     model-path: Qwen/Qwen3-8B
     backend: sglang
     num-gpus: 2
     remote:
       url: https://gpu-host.example.com
   training:
     backend: tinker
     options:
       state-dir: /var/lib/reef/tinker
       lora-rank: 32

``reef serve`` then starts no local component. The GPU host downloads
``model-path``; the Reef host keeps it as a remote identifier. The `settings
reference <../reference/configuration-settings.rst#inference>`__ lists the
``inference.remote`` settings.

Keep the remote control and inference tokens different. The control token
can start model code and replace weights: protect it like a host login.
Use HTTPS and keep credentials out of logs and configuration files.

The component serves one controlling runtime at a time. When that runtime
stops, the engines stop but the component remains available. After a component
restart, the runtime detects it within 30 seconds and reloads the served adapters.
Upload time depends on adapter size and network speed.

Train all the parameters on Spindle
-----------------------------------

The ``spindle_full`` backend trains all the parameters of a model on a
`Spindle <https://github.com/modal-projects/spindle>`__ deployment's
full-parameter engine, through the same job loop as ``tinker``. Use it to keep
training a checkpoint whose weights are on Hugging Face, such as one that was
trained outside Reef.

Prerequisites:

- The checkpoint is a Hugging Face revision, ``hf://<repo>@<commit>``, with a
  full 40-digit commit SHA.
- The Spindle deployment has a full-parameter engine whose ``model`` is
  ``<repo>`` and whose weights are that commit. Reef does not start engines;
  deploy the engine with Spindle first.
- The service environment has the Spindle API key, in ``TINKER_API_KEY`` or
  the variable that ``training.options.api-key-env`` names, and a Hugging Face
  token that can read the repo, such as ``HF_TOKEN``.
- The service has Modal credentials (``uv pip install 'reef-infra[modal]'``,
  then ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` or a profile). In the
  Spindle deployment's Modal environment, the Secret that ``hf-secret`` names
  (default ``huggingface-secret``) holds an ``HF_TOKEN`` that can write
  ``release-repo`` and read the base.

.. code:: yaml

   inference:
     model-path: org/model                 # the base's repo
   training:
     backend: spindle_full
     options:
       state-dir: /var/lib/reef/spindle-full
       base-url: https://<workspace>--spindle-server.modal.run
       base: hf://org/model@0123456789abcdef0123456789abcdef01234567
       release-repo: org/model-releases     # one commit per saved checkpoint
       modal-environment: spindle            # where the deployment runs
       learning-rate: 1.0e-6

**Checkpoints.** The starting artifact references the base revision.
Each checkpoint exports full weights, tokenizer and template files to a new
commit in ``release-repo``. The artifact holds an ``hf-reference.json`` and a
full-parameter ``tinker-checkpoint.json`` manifest.

Reef's engines load the Hugging Face revision. Default trainer serving uses
Spindle's sampler. Later training restores the weights and optimizer state.

**Imports.** An import into a ``spindle_full`` scenario must be one of:

- a full-parameter manifest of the base model on the same service, beside a
  reference to full weights;
- the base's own files, by reference or as a copy.

The engine trains only its base, so Reef refuses other full weights with HTTP
400. A component trains one base. To train another checkpoint, deploy a Spindle
engine for it and a scenario with it as ``base``.

**Cost and validation.** Each job reserves training GPUs, starts a model and
exports its weights. Include startup and export time in the budget.
Only one live 4B smoke run has validated this backend; it measured no learning quality.

Score centering
~~~~~~~~~~~~~~~

A policy gradient whose token weight is not the full importance ratio
``p / q`` drifts the trained policy ``p`` toward the sampler ``q``. Score
centering (`arXiv:2609.20807 <https://arxiv.org/abs/2609.20807>`__) removes
that drift. With ``training.options.score-centering: true``, a ``spindle_full``
component trains each ``importance_sampling`` job, such as a generated-task or
task job, as Spindle's ``cispo`` loss with the centering term:

- ``cispo`` weights each sampled token's score by
  ``clamp(p / q, score-centering-weight-low, score-centering-weight-high)``.
  The default bounds, 1 and 1, are REINFORCE. Wider bounds, such as 0 and 2,
  are truncated importance sampling. The term is zero under the full ratio,
  so the weight cannot be the full ratio.
- A task job's rollouts record the sampler's top ``score-centering-top-k``
  log probabilities (default 128) at each sampled token. Those must come from
  the distribution that the trainer reads, so the rollouts sample at
  temperature 1 without truncation. A model call that sets another
  ``temperature``, or a ``top_p`` or ``top_k``, gets HTTP 400. Harbor's
  ``terminus-2`` agent sets none of them by default.
- Only a task job's rollouts record the sampler's top-k. A job of samples that
  serving recorded, or of uploaded trajectories, fails with score centering on.
- ``sft`` jobs train as they are. A job of another policy-gradient loss family
  is refused, since the term must match the gradient it corrects. So is a
  nonzero ``kl-coef``.

.. code:: yaml

   training:
     backend: spindle_full
     options:
       score-centering: true
       score-centering-top-k: 128           # sampler log probabilities per sampled token
       score-centering-weight-low: 1.0      # 1 and 1: REINFORCE
       score-centering-weight-high: 1.0
       score-centering-min-tail-mass: 1.0e-6

The Spindle deployment must implement the sampler's ``topk_sample_logprobs``
and score centering on ``cispo`` (``loss_fn_config.score_centering`` with the
``sampler_topk_token_ids`` and ``sampler_topk_logprobs`` inputs). A deployment
without them refuses the rollouts' sample requests or the training call, so the
job fails. Spindle reports the term in each job's metrics, as
``score_centering:sum``, ``score_centering_sampler_head_mass:mean`` and
``score_centering_trainer_head_mass:mean``.
