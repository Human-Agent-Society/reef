Train with Tinker
=================

The optional ``tinker`` backend runs LoRA training and sampling through
`Tinker's SDK <https://tinker-docs.thinkingmachines.ai/>`__. Reef runs on a CPU
host and retains ownership of requests, feedback, candidate selection, and the
versioned release chain. It does not launch Ray, Slime, or a local inference
engine. The tested SDK interface is pinned to ``tinker==0.28.1``.

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
requires exclusive ownership of its state directory and one training scenario.

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
which ``state-dir`` remembers across restarts. Backend-neutral step scheduling retains comparison sets, explicit
batch sizes, shuffling, epochs, and partial/drop/error remainder policies.
``training.options.batch-size`` supplies the configured optimizer batch size.
Only exact-version samples are admitted: ``max_staleness`` must be zero.

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

``lora-rank``, ``seed``, and ``learning-rate`` configure adapter initialization
and Adam updates. Other Adam settings currently use the pinned SDK's defaults
(beta1 0.9, beta2 0.95, epsilon 1e-12, zero weight decay and gradient clipping).
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
schema version, base model, LoRA rank, training-state URI, and sampler URI. It
contains references, not portable weight tensors. Restore and rollback require
continued access to both remote checkpoints in the originating account/project.
Local disk retention does not remove remote files. Use Tinker's checkpoint
management for remote retention, preserving all checkpoints referenced by Reef.

Selection closes admission until the matching Reef training job is committed.
Already admitted requests retain their immutable sampler snapshot. The weight
loader binds the materialized, recovered artifact before serving a scenario;
restart creates a fresh runtime-load incarnation and starts subsequent training
from that artifact's optimizer state. Rollback binds the republished older
snapshot, including its optimizer. The default checkpoint cadence of one is
recommended: non-checkpoint live releases can only be served during their
original runtime incarnation and fall back to the last durable checkpoint on
restart, following Reef's normal weight recovery contract.

Serve with a local SGLang engine
--------------------------------

Tinker can also be the trainer behind Reef's own inference engines. Select
``inference.backend: sglang`` with a local ``inference.model-path`` (the same
base model Tinker trains), ``inference.num-gpus`` and, for tensor-parallel
engines, ``inference.tensor-parallel-size``:

.. code:: yaml

   schema-version: 2
   inference:
     model-path: /models/Qwen3-8B
     backend: sglang
     num-gpus: 2
   training:
     backend: tinker
     options:
       state-dir: /var/lib/reef/tinker
       lora-rank: 32
       max-loaded-adapters: 2

Reef then runs its model driver: it reserves the inference GPUs in Ray,
starts the SGLang engines with LoRA serving enabled (``max-lora-rank`` from
``lora-rank``, ``max-loaded-loras`` and ``max-loras-per-batch`` from
``max-loaded-adapters``, ``lora-target-modules`` defaulting to ``all``), and
runs Reef's training coordinator with Tinker as its training backend. The
trainer reserves no GPU. Each scenario trains its own adapter: a training job
runs one optimizer step on Tinker from the scenario's published checkpoint,
downloads the result and converts it with ``tinker-cookbook`` into a PEFT
adapter directory under the job's checkpoint, and publication has Reef ask
every engine to load that directory under the adapter name it records for
the scenario; the trainer itself never talks to the engines. Requests are addressed to that adapter by the weight surface, and
the engines capture the sampled tokens and log probabilities exactly as they
do for Slime. Rejected candidates load nothing; a restart reloads each
scenario's committed adapter from disk before serving. The converter is
``tinker-cookbook``, pinned in the ``tinker`` extra; it brings torch and
transformers, which the driver host running the engines already has.

The trajectories then come from the local engine while Tinker computes the
loss, so the base model and tokenizer on both sides must match, and the
importance-sampling ratio carries the numerical difference between the two
engines' log probabilities. ``training.colocate`` does not apply: there are
no training GPUs to share. Tinker's adapters also carry a LoRA on ``lm_head``,
which the Slime configuration never targets; whether the installed SGLang
accepts that module is checked only when the engine loads the adapter.

The SDK session opens inside Reef's coordinator process, where the trainer
runs, and the coordinator learns the recipe's Tinker loss reference from the
driver, so the same recipe package is imported on both sides.

``tests/reef_service/test_tinker_local_engine_ray.py`` runs this topology on a
CPU: a local Ray cluster with pretend GPUs, the real model driver, a stub
engine that accepts Reef's adapter-file transfer in place of SGLang, and the
Tinker SDK faked on the driver's path. With ``REEF_TEST_TINKER_REAL=1`` and
``TINKER_API_KEY`` set it trains, downloads and converts a real checkpoint.

Verification scope
-------------------

CPU tests exercise data alignment and loss math, SDK call ordering, rejected and
uncertain candidates, commit admission, request capture and SSE, deployment,
scenario commit/restart/rollback, and the coordinator path with a local engine
(publication, rejection, restart recovery, adapter reload) against engine and
SDK doubles. A live Tinker run is additionally needed
to verify account model availability, remote persistence, and training quality.
No live training or performance result is claimed by this implementation.
