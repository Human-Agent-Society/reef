Train with Tinker
=================

The optional ``tinker`` backend runs LoRA training and sampling through
`Tinker's SDK <https://tinker-docs.thinkingmachines.ai/>`__. Reef runs on a CPU
host and retains ownership of requests, feedback, candidate selection, and the
versioned release chain. By default it launches no Ray cluster, no Slime and
no inference engine; `Serve on inference engines`_ describes the option to
serve the trained adapters on SGLang engines. The tested SDK interface is
pinned to ``tinker==0.28.1``.

Install and run
---------------

Use Python 3.12 or newer on Linux or macOS, the version Reef itself requires.

.. code:: bash

   uv pip install -e '.[tinker]' -e ./third_party/reef-client

Set ``TINKER_API_KEY`` in the service environment. Credentials are read only
when the selected runtime starts; neither resolved configuration nor checkpoint
manifests contain the key. ``training.options.api-key-env`` selects a different
environment variable. An optional ``project-id`` selects the Tinker project.

The `four-rollout smoke <../../tutorials/tinker/README.md>`__ includes a complete
config and runner. It creates one adapter, generates four short responses, and
performs one optimizer step with synthetic rewards. Running it uses Tinker
credits; it is a mechanism check, not a benchmark or a model-quality claim.

Choose ``training.backend: tinker``, an available Tinker base model in
``inference.model-path``, and a persistent ``training.options.state-dir``.
The model remains a remote identifier: deployment does not download its weight
snapshot. The SDK obtains tokenizer resources for prompt rendering. The runtime
requires exclusive ownership of its state directory. When Tinker samples, a
deployment trains one scenario.

Supported inference and training
---------------------------------

The backend supports ``/v1/chat/completions`` with textual system, user, and
assistant messages, including a final assistant prefill. It accepts one output
per request, ``max_tokens`` or ``max_completion_tokens``, ``temperature``,
``top_p``, ``top_k``, ``seed``, and ``stop``. The supported template option is
``chat_template_kwargs.enable_thinking``. Unknown options, tool calls,
multimodal messages, and Anthropic routes fail explicitly.

Streaming is buffered: the complete sampled response is emitted as valid
OpenAI SSE events, with the exact private training capture retained separately.
It does not provide token-by-token latency. Prompt IDs come from the tokenizer's
chat template. Sampled IDs and log probabilities come directly from Tinker;
the backend never recovers training tokens by tokenizing decoded output.

Reef's normal ``WeightTrainingRecipe`` and ``RuntimeCandidateBackend`` drive the
integration. The backend is one training runtime (``reef.train.tinker_backend``)
and one inference runtime (``reef.inference.tinker``), like every other pair
Reef schedules. They share nothing in process: each candidate's artifact
carries a ``tinker-checkpoint.json`` manifest naming its remote training
state and sampler, the inference side serves the sampler an artifact names,
and the training side branches from the checkpoint Reef last committed,
which ``state-dir`` remembers across restarts.

``training.options.batch-size`` (default 1) is the batch size: the samples in
one optimizer step. The training pipeline makes a batch when it holds this
many samples. A batch takes whole record sets, so it can hold more; Reef then
cuts it into optimizer steps of the batch size. Backend-neutral step
scheduling retains comparison sets, shuffling, epochs, and
partial/drop/error remainder policies for the last step.

Tinker computes the loss on its side from a built-in loss function and the
per-sample inputs that function reads. The built-in ``importance_sampling``
family places the trajectory advantage on every response token the loss mask
selects. A method registers its own ``TinkerLoss`` with
``register_loss_family_ref(name, "package.module:CLASS", backend="tinker")``,
next to its Slime reference; the class is imported when the family is first
resolved. Its ``loss_fn`` names the Tinker built-in it trains with and
``inputs`` shapes that function's ``loss_fn_inputs``. A ``TinkerCustomLoss``
instead computes the loss on the Reef host from the token log probabilities
Tinker returns, through ``forward_backward_custom``; that path needs torch
locally and no shipped recipe uses it. The TTTD recipe's
``recipes/tttd/tinker.py`` keeps its unmasked policy term and masked,
centered frozen-base KL on ``importance_sampling``, matching its Slime
objective. ``training.options.kl-coef`` controls that penalty. The base
sampler scores the original token IDs, and those probabilities never replace
the captured behavior-policy probabilities. SAO and OpenClaw-RL losses are
not implemented by this integration; selecting an unregistered loss family
fails before training.

Each job trains the Tinker loss of the objective that its recipe selected
(`Switch methods within a run
<../developer-guide/write-a-recipe.rst#switch-methods-within-a-run>`__).
Tinker preparation needs one advantage per trajectory, so a supervised or
distillation objective fails before training.
``kl-coef`` applies to every job. When it is not zero, a job whose loss has no
KL penalty, such as ``importance_sampling``, also fails before training.

``lora-rank``, ``seed``, and ``learning-rate`` configure adapter initialization
and Adam updates. Other Adam settings currently use the pinned SDK's defaults
(beta1 0.9, beta2 0.95, epsilon 1e-12, zero weight decay and gradient clipping).
``learning-rate`` applies until a recipe selects a learning-rate schedule.
``training.timeout-s`` bounds SDK futures and configures
transport timeouts; ``inference.timeout-s`` bounds sampling. These timeouts do
not cancel a remote update that the service has already accepted.

Candidates, checkpoints, and recovery
--------------------------------------

Each candidate starts a separate Tinker training session from the incumbent's
saved weights **and optimizer state**. An accepted optimizer update exports both
``save_state`` and ``save_weights_for_sampler`` with explicit unique names and
no expiration. The training session then closes. Candidates can therefore be
evaluated without changing the incumbent sampler, and rejected candidates do
not require undoing an in-place model mutation.

An uncertain API result may leave an unreferenced remote checkpoint and consume
credits. A retry starts from the same incumbent checkpoint in a fresh session;
it cannot accidentally apply the uncertain gradient a second time to that model.
This favors recovery correctness over session reuse and training throughput.

Reef publishes a small local ``tinker-checkpoint.json`` artifact containing the
schema version, base model, LoRA rank, training-state URI, and sampler URI.
After a recipe selects a learning-rate schedule, it also records the schedule's
progress. It
contains references, not portable weight tensors. Restore and rollback require
continued access to both remote checkpoints in the originating account/project.
Local disk retention does not remove remote files. Use Tinker's checkpoint
management for remote retention, preserving all checkpoints referenced by Reef.

Selection closes admission until the matching Reef training job is committed.
Already admitted requests retain their immutable sampler snapshot. The weight
loader binds the materialized, recovered artifact before serving a scenario;
restart creates a fresh runtime-load incarnation and starts subsequent training
from that artifact's optimizer state. Rollback binds the republished older
snapshot, including its optimizer. Each Tinker version is a durable
checkpoint, so a restart does not lose a version.

Serve on inference engines
--------------------------

Reef's own SGLang engines can serve the adapters that Tinker trains, in place
of Tinker's sampler. The engines run on an inference component
(``python -m reef.inference.remote``), a service separate from the trainer.
The Reef control plane starts the engines on the component and loads each
saved adapter into them. Tinker still trains in the HTTP service's process,
and the trainer never talks to the engines.

Select ``inference.backend: sglang`` with the base model that Tinker trains in
``inference.model-path``, ``inference.num-gpus`` and, for tensor-parallel
engines, ``inference.tensor-parallel-size``:

.. code:: yaml

   inference:
     model-path: Qwen/Qwen3-8B
     backend: sglang
     num-gpus: 2
   training:
     backend: tinker
     options:
       state-dir: /var/lib/reef/tinker
       lora-rank: 32
       max-loaded-adapters: 2

``reef serve`` then starts an inference component on this host before the
HTTP service. The component listens on a free loopback port and starts its own
Ray session; with ``training.ray-address`` it joins that cluster instead.
``reef serve`` generates the two component tokens
(``REEF_REMOTE_CONTROL_TOKEN`` and ``REEF_REMOTE_INFERENCE_TOKEN``) unless
they are set, and both children inherit them. The HTTP service starts the
engines before it serves, so its readiness deadline is at least
``training.ready-timeout``.

The engines start with LoRA serving enabled: ``max-lora-rank`` comes from
``lora-rank``, ``max-loaded-loras`` and ``max-loras-per-batch`` from
``max-loaded-adapters``, and ``lora-target-modules`` defaults to ``all``. The
engines keep at most ``max-loaded-adapters`` adapters resident. Each scenario
keeps its served adapter resident, so this value must be at least the number
of training scenarios; a scenario's new adapter replaces its own served
adapter when no slot is free.

Each scenario trains its own adapter. A job runs the batch's optimizer steps
on Tinker from the scenario's committed checkpoint. It then downloads the
result and converts it with ``tinker-cookbook`` into a PEFT adapter in the
``adapter/`` directory of the job's checkpoint. When Reef selects the
candidate, the control plane loads that directory into every engine under a
new adapter name for the scenario. Requests reach that adapter through the
weight surface. The engines capture the sampled tokens and log probabilities
as they do for Slime. A rejected candidate loads nothing. A restart starts the
engines again and loads each scenario's committed adapter before the scenario
serves. The converter is pinned in the ``tinker`` extra; it brings torch and
transformers.

The trajectories then come from Reef's engines while Tinker computes the loss,
so the base model and tokenizer on both sides must match, and the
importance-sampling ratio carries the numerical difference between the two
engines' log probabilities. ``training.colocate`` does not apply: there are no
training GPUs to share. Tinker's adapters also carry a LoRA on ``lm_head``,
which the Slime configuration never targets; whether the installed SGLang
accepts that module is checked only when the engines load the adapter.

Serve from a remote GPU host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The component can also run on another host, such as a rented GPU machine or
a managed GPU platform, while Reef and the Tinker trainer stay on a CPU host.
No tensors cross the network: the control plane uploads each selected
adapter directory to the component, and the component loads it into its
engines.

On the GPU host, install Reef with SGLang. Set two different tokens and start
the component:

.. code:: bash

   export REEF_REMOTE_CONTROL_TOKEN=...    # starts engines and replaces weights
   export REEF_REMOTE_INFERENCE_TOKEN=...  # sends inference requests
   python -m reef.inference.remote --port 8000

The component starts its own Ray session and no engines. Its control routes
answer ``409`` until a control plane starts the engines. It starts only the
engine backend that ``--engine`` names (default ``sglang``) and refuses any
other backend that a control plane asks for.
``python -m reef.inference.remote --help`` lists the bind address, the upload
directory and the Ray options.

On the Reef host, set the same two variables. Keep ``inference.backend:
sglang`` and add an ``inference.remote`` section with the component's URL:

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

``reef serve`` then starts no local component. The control plane asks the
remote component to start the engines with the settings described above. The
GPU host downloads ``model-path``; the Reef host keeps it as a remote
identifier.

The HTTP service sends inference requests to the component, and the
component forwards each request and its answer unchanged. Token capture and
version stamps therefore come from the engines. When the control plane stops,
the component stops its engines and keeps running.

The control token gives its holder control of the GPU host. Its holder
chooses the model, the engine options and the served weights, and the engines
run a model's own code (``trust_remote_code``). Protect it like a login to
that host: serve the component over HTTPS, keep the token out of logs and
deployment files, and keep it separate from the inference token.

Each selected candidate uploads one adapter directory, so the upload time
grows with the adapter size and the network path. The component serves one
control plane at a time: a second start replaces the first one's engines.
`The inference section <../reference/configuration.rst#the-inference-section>`__
lists the ``inference.remote`` settings.

``tests/tinker_backend/test_tinker_local_engine_ray.py`` runs this topology on
a CPU. The component runs a stub engine in its own process and Ray session, and
the test process is the control plane with the Tinker SDK faked. It trains one
TTTD step, checks that the component serves the adapter, restarts the control
plane, and checks that the component serves the committed adapter again. With
``REEF_TEST_TINKER_REAL=1`` and ``TINKER_API_KEY`` set, it trains, downloads
and converts a real checkpoint.

Verification scope
-------------------

CPU tests exercise data alignment and loss math, SDK call ordering, rejected and
uncertain candidates, commit admission, request capture and SSE, deployment,
scenario commit/restart/rollback, and serving on engines (per-scenario
adapters, rejection, restart reload) against engine and SDK doubles. The
component topology is tested with stub engines only; real SGLang engines have
not been run with it. A live Tinker run is additionally needed
to verify account model availability, remote persistence, and training quality.
No live training or performance result is claimed by this implementation.
