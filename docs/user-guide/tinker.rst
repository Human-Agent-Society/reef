Train with Tinker
=================

The optional ``tinker`` backend runs LoRA training and sampling through
`Tinker's SDK <https://tinker-docs.thinkingmachines.ai/>`__. Reef runs on a CPU
host and retains ownership of requests, feedback, candidate selection, and the
versioned release chain. It does not launch Ray, Slime, or a local inference
engine. The tested SDK interface is pinned to ``tinker==0.28.1``.

Install and run
---------------

Use Python 3.11 or newer on Linux or macOS. The base Reef distribution continues
to support Python 3.10; installing the optional extra there does not install
Tinker, and constructing its runtime raises a version error.

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

Reef's normal ``WeightTrainingRecipe`` and ``RuntimeTrainingBackend`` drive the
integration. Backend-neutral step scheduling retains comparison sets, explicit
batch sizes, shuffling, epochs, and partial/drop/error remainder policies.
``training.options.batch-size`` supplies the configured optimizer batch size.
Only exact-version samples are admitted: ``max_staleness`` must be zero.

The built-in ``importance_sampling`` loss accepts one trajectory advantage per
sample and applies the response loss mask. A method may register a
``TinkerLoss`` using ``register_tinker_loss``. The TTTD recipe registers its own
adapter in ``recipes/tttd/tinker.py``: its unmasked policy term and masked,
centered frozen-base KL match the definitions in its existing Slime objective.
``training.options.kl-coef`` controls that penalty. The base sampler scores the
original token IDs, and those probabilities never replace the captured
behavior-policy probabilities. SAO and OpenClaw-RL losses are not implemented
by this integration; selecting an unregistered loss family fails before training.

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

Verification scope
-------------------

CPU tests exercise data alignment and loss math, SDK call ordering, rejected and
uncertain candidates, commit admission, request capture and SSE, deployment,
and scenario commit/restart/rollback. A live Tinker run is additionally needed
to verify account model availability, remote persistence, and training quality.
No live training or performance result is claimed by this implementation.
