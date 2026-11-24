Train with Tinker
=================

The optional ``tinker`` backend trains LoRA adapters and samples through
`Tinker's SDK <https://tinker-docs.thinkingmachines.ai/>`__. Reef runs on a CPU
host and keeps the requests, the feedback, candidate selection and the snapshot
chain. By default it starts no Ray cluster, no Slime and no inference engine.
The ``tinker`` extra pins the SDK (``tinker==0.39.0``).

This page covers:

- `Install and run`_: the hosted LoRA path.
- `What the backend supports`_: requests, losses and job settings.
- `Jobs, checkpoints, and recovery`_.
- `Serve on inference engines`_: optional, Reef's SGLang engines serve the adapters.
- `Train all the parameters on Spindle`_: optional, the ``spindle_full`` backend.

Install and run
---------------

Use Python 3.12 or newer on Linux or macOS.

.. code:: bash

   uv pip install -e '.[tinker]' -e ./third_party/reef-client

Then configure the deployment:

- Set ``training.backend: tinker``.
- Set ``inference.model-path`` to a Tinker base model. It stays a remote
  identifier: Reef downloads no weights, and the SDK fetches the tokenizer.
- Set ``training.options.state-dir`` to a persistent directory. One runtime
  owns it exclusively.
- Set ``TINKER_API_KEY`` in the service environment.
  ``training.options.api-key-env`` names another variable, and the optional
  ``project-id`` selects the Tinker project. Reef reads the key when the
  runtime starts. The resolved configuration and the checkpoint manifests do
  not contain it.

The `four-rollout smoke <../../tutorials/tinker/README.md>`__ has a complete
config and runner. It creates one adapter, generates four short responses and
runs one optimizer step with synthetic rewards. It uses Tinker credits. It
checks the mechanism; it is not a benchmark.

**Another Tinker-compatible service.** ``training.options.base-url`` selects
one, such as a `Spindle <https://modal.com/docs/guide/spindle>`__ deployment on
Modal. Unset, Reef uses Thinking Machines' hosted Tinker. The trainer, its
sampler and ``inference.serving: trainer`` all use that service.

- A checkpoint path such as ``tinker://<run>/weights/<name>`` resolves only on
  the service that made it. Each checkpoint manifest records the service, and
  the component refuses a checkpoint that another service holds.
- A service may lack some calls. Spindle serves no checkpoint archives, so
  Reef's own engines can serve only hosted Tinker's checkpoints.
- Reef makes a base-model sampling client at its first use (a KL term, a task
  rollout or a request). On Spindle such a client starts a sampler pool.

**Scenarios.** Several scenarios can share the training component, because each
job carries its own start. When Tinker samples (``inference.serving: trainer``,
the default), a deployment still trains one scenario. When Reef's engines serve
the adapters, each scenario gets its own adapter, so several scenarios can
train on one deployment.

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

Serving records no training tensors. The training component renders each
recorded call with the base model's tokenizer and
``training.options.chat_template_kwargs``. This value must equal the chat
template arguments of serving, for example ``{enable_thinking: false}``. The
component computes the rollout log probabilities with a forward pass at the
job's start weights. A sample does not train when it cannot render, or when its
tokens after the first exceed the context window. The job metric
``unrenderable_samples`` counts such samples. A task job's own rollouts keep
the token IDs and log probabilities that Tinker sampled.

Losses
~~~~~~

Tinker computes the loss from one of its built-in loss functions. Each job
trains the Tinker loss of the objective that its recipe selected (`Train each
kind of sample with its own method
<../developer-guide/recipe-training-service.rst#job-parameters>`__).

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

Constraints:

- Tinker preparation needs one advantage per trajectory. A supervised or
  distillation objective fails before training.
- ``kl-coef`` applies to every job. When it is not zero, a job whose loss has
  no KL penalty, such as ``importance_sampling``, fails before training.

To add a loss, a method gives its ``TinkerLoss`` a ``loss_family`` name and
decorates the class with ``@register_loss_family``
(``reef.train.tinker_backend.loss_families``), in the module of its
``TinkerObjective``. Its ``loss_fn`` names the Tinker built-in and ``inputs``
shapes that function's ``loss_fn_inputs``. A ``TinkerCustomLoss`` instead
computes the loss on the Reef host from the token log probabilities that Tinker
returns (``forward_backward_custom``). That path needs torch locally, and no
shipped recipe uses it.

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
     - Job size: the pipeline submits a job when it holds this many samples.
       Set it to the batch size, so that one job is one optimizer step.
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

A job's data can be Harbor tasks, as the `generated-task recipe
<../reference/configuration-generated-tasks.rst#generated-tasks>`__ makes. Before it trains,
the component runs ``rollout.rollouts_per_task`` rollouts of each task on
Tinker's sampler at the job's start weights. The agent's model calls go to a
sampler endpoint on this host, and the task's verifier scores each rollout.
The component trains on each task's group of samples and reports each task's
rewards in the job's metrics. It keeps the rollouts in its job directory, so a
resumed job trains the same samples. The ``rollout`` job parameter sets the
rollouts, the agent and the sandbox. Reef's serving takes no part, so another
runtime can serve the snapshots.

Jobs, checkpoints, and recovery
-------------------------------

**Start.** A new scenario starts from a base root: a ``base.json`` file that
names the model, the LoRA rank and the service. Control writes it from the
configuration without a Tinker call, and serving it samples the base model. The
first job makes a new LoRA from ``seed``. With ``seed: null``, the training
service saves one untrained adapter when the first such job runs, keeps its
manifest in ``<state-dir>/base/``, and starts every such job from it.

**A job.** A job restores the weights and the optimizer state of its start
checkpoint in a new Tinker training session and runs the optimizer steps of its
data. It then saves the state (``save_state``) and the sampler weights
(``save_weights_for_sampler``), each under a unique name without expiration,
and closes the session. The job parameter ``save_interval``
(``recipe.config.job_parameters``) gives the optimizer steps between two saved
checkpoints. Without it, a job saves one checkpoint when it ends.

**Candidates.** Each saved checkpoint is a candidate. Candidate evaluation
decides whether Reef publishes it; a rejected candidate changes no served
sampler. The next job starts from the last saved checkpoint, whether or not
Reef selected it.

**The artifact.** Reef publishes a small ``tinker-checkpoint.json`` manifest:
the schema version, base model, LoRA rank, training-state URI and sampler URI,
and the progress of a learning-rate schedule when a recipe selects one. It
holds references, not weight tensors. Restore, activation and train-from need
continued access to both remote checkpoints in the originating account and
project. Local retention does not remove remote files. Use Tinker's checkpoint
management for remote retention, and keep every checkpoint that Reef
references.

**Failures and restarts.**

- An uncertain API result may leave an unreferenced remote checkpoint and
  consume credits. A retry starts from the same checkpoint in a fresh session,
  so it cannot apply the uncertain gradient twice.
- A job that a restart interrupts continues after its last saved checkpoint.
- Admitted requests keep the sampler of the snapshot they started with. A
  restart serves the active selection under a new runtime load.
- Train-from works. Reef imports the older artifact's manifest into the
  component and cancels the running job that continues the previous start. The
  next job that names no start begins from that artifact's weights and
  optimizer state.

Serve on inference engines
--------------------------

Reef's SGLang engines can serve the adapters that Tinker trains, in place of
Tinker's sampler. The engines run on an inference component
(``python -m reef.inference.remote``), separate from the trainer. The control
plane starts the engines on the component and loads each saved adapter into
them. Tinker still trains in the HTTP service's process and never talks to the
engines.

Select ``inference.backend: sglang`` with the base model that Tinker trains.
Any engine setting selects ``inference.serving: engines``:

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

``reef serve`` then starts an inference component on this host before the HTTP
service:

- The component listens on a free loopback port and starts its own Ray session.
  With ``training.ray-address`` it joins that cluster.
- ``reef serve`` generates the two component tokens
  (``REEF_REMOTE_CONTROL_TOKEN`` and ``REEF_REMOTE_INFERENCE_TOKEN``) unless
  they are set.
- The HTTP service starts the engines before it serves, so its readiness
  deadline is at least ``training.ready-timeout``.
- ``reef serve`` refuses ``training.colocate``: there are no training GPUs to
  share.

**Adapter slots.** The engines start with LoRA serving enabled:
``max-lora-rank`` is ``lora-rank`` and ``lora-target-modules`` is ``all``.
``inference.max-loaded-adapters`` sets ``max-loaded-loras`` and
``max-loras-per-batch``. Each scenario keeps its served adapter resident, so
this value must be at least the number of training scenarios. A scenario's new
adapter replaces its own served adapter when no slot is free.

**Loading a snapshot.** A snapshot still holds only its manifest. When the
control plane loads one, it downloads the adapter that the manifest's
``sampler_path`` names and converts it into a PEFT adapter. It keeps the result
under ``<state-dir>/sampler-weights/``, one directory per sampler path, so a
snapshot downloads once. It then loads the adapter into every engine under a
new adapter name, and requests reach it through the scenario's
``served_models`` route. A rejected candidate loads nothing. A restart starts
the engines again and loads each scenario's committed adapter before the
scenario serves. A service that serves no checkpoint archives, such as Spindle,
fails the load. The converter is pinned in the ``tinker`` extra; it brings
torch and transformers.

**Matching both sides.** The trajectories come from Reef's engines while Tinker
computes the loss, so the base model and tokenizer must match. Set
``training.options.chat_template_kwargs`` to the engines' chat template
arguments. Tinker's adapters also carry a LoRA on ``lm_head``. Whether the
installed SGLang accepts that module is checked only when the engines load the
adapter. This topology is tested with stub engines only; real SGLang engines
have not been run with it.

Serve from a remote GPU host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The component can run on another host, such as a rented GPU machine, while Reef
and the Tinker trainer stay on a CPU host. The control plane uploads each
selected adapter directory to the component, and the component loads it into
its engines.

On the GPU host, install Reef with SGLang. Set two different tokens and start
the component:

.. code:: bash

   export REEF_REMOTE_CONTROL_TOKEN=...    # starts engines and replaces weights
   export REEF_REMOTE_INFERENCE_TOKEN=...  # sends inference requests
   python -m reef.inference.remote --port 8000

The component starts its own Ray session and no engines. Its control routes
other than ``health`` answer ``409`` until a control plane starts the engines.
It starts only the engine backend that ``--engine`` names (default ``sglang``)
and refuses any other. ``python -m reef.inference.remote --help`` lists the
bind address, the upload directory and the Ray options.

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
``model-path``; the Reef host keeps it as a remote identifier. `The inference
section <../reference/configuration-inference.rst#the-inference-section>`__ lists the
``inference.remote`` settings.

Behavior to rely on:

- The HTTP service sends inference requests to the component, which forwards
  each request and its answer unchanged. Version stamps come from the engines.
- When the control plane stops, the component stops its engines and keeps
  running.
- When the component restarts, it runs no engines. The control plane notices
  within 30 seconds, starts the engines again and loads each scenario's served
  adapter.
- The component serves one control plane at a time: a second start replaces
  the first one's engines.
- Each selected candidate uploads one adapter directory, so the upload time
  grows with the adapter size and the network path.

.. warning::

   The control token gives its holder control of the GPU host. Its holder
   chooses the model, the engine options and the served weights, and the
   engines run a model's own code (``trust_remote_code``). Protect it like a
   login to that host: serve the component over HTTPS, keep the token out of
   logs and deployment files, and keep it separate from the inference token.

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

**Start.** A new scenario's root snapshot holds a reference to the base
revision (`Import an artifact <operate.rst#import-an-artifact>`__). Control
writes it from the configuration. A job from it starts a new model with the
base's own weights.

**Snapshots.** When Spindle saves a checkpoint, it writes the Hugging Face
weights to its checkpoint Volume (``checkpoint-volume``, default
``spindle-checkpoints``). A Modal function then commits those weights, with the
base's tokenizer, chat template and processor files, to ``release-repo``. Each
saved checkpoint is one commit that holds exactly that checkpoint's files. The
snapshot holds two files:

- ``hf-reference.json``, the commit with the weights. Any serving runtime can
  load it: Reef's engines (``inference.serving: engines``) download it, and a
  runtime that loads from the Hub reads the reference.
- ``tinker-checkpoint.json``, a manifest with ``lora_rank: null``. A later job
  restores its weights and optimizer state. Spindle's sampler, the default
  serving (``inference.serving: trainer``), serves its sampler path.

**Imports.** An import into a ``spindle_full`` scenario must be one of:

- a full-parameter manifest of the base model on the same service, beside a
  reference to full weights;
- the base's own files, by reference or as a copy.

The engine trains only its base, so Reef refuses other full weights with HTTP
400. A component trains one base. To train another checkpoint, deploy a Spindle
engine for it and a scenario with it as ``base``.

**Cost.** A full-parameter run is single tenant. Each job holds the engine's
training GPUs while it runs, for example 8 H200 GPUs to train and 4 to sample
for a 27B model. Each job starts a new model on the engine and unloads it when
it ends, so each job pays the engine's start time. On one scoped Spindle run of
a 4B model, the first job took about 6 minutes, a job from a saved checkpoint
about 4, and the export of the weights (9 GB) about a minute. That run is the
only live ``spindle_full`` run; it did not measure training quality.

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
