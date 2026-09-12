"""The single-process MLX model: generation, optimization, and adapter I/O.

One engine owns one base model, one LoRA adapter and one optimizer for the
life of the service. Serving generates through the adapter Reef has published;
training mutates the same parameters, so the two never run at once — the
runtime closes inference admission around every optimizer step.

Keeping the optimizer here rather than rebuilding it per step is what makes a
Reef training step continue the previous one: the Adam moments and the step
counter survive across the whole run, exactly as they do inside a long-lived
Megatron actor.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner import linear_to_lora_layers

from reef.core.errors import ReefError
from reef.train.mlx_backend.gated_delta import chunked_gated_delta
from reef.train.mlx_backend.messages import prepare_messages
from reef.train.mlx_backend.rows import DistillationRow, GenerationListener, TeacherCandidate, TrainingRow

ADAPTER_WEIGHTS = "adapters.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
ORIGIN = "reef_origin.json"
ORIGIN_SCHEMA = "reef.mlx.adapter/1"

DEFAULT_LORA_KEYS = ("self_attn.q_proj", "self_attn.v_proj")


class MLXEngineError(ReefError):
    """The MLX engine was asked for something it cannot do correctly."""


@dataclass(frozen=True)
class MLXEngineConfig:
    """Everything about the model that a deployment chooses.

    Generation length, group size and adapted layer count are the knobs that
    bound unified-memory use, so they are all explicit rather than implied.
    """

    model_path: str
    lora_layers: int = 8
    lora_rank: int = 8
    lora_scale: float = 2.0
    lora_dropout: float = 0.0
    lora_keys: tuple[str, ...] = DEFAULT_LORA_KEYS
    learning_rate: float = 1e-5
    #: AdamW's decoupled decay. MLX defaults to 0.01; slime's OpenClaw-RL run
    #: sets 0.1, and this stays at MLX's default until a deployment asks for
    #: the reference's.
    weight_decay: float = 0.01
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    #: Maximum sequences held in one backward pass. Lower it when a step
    #: exhausts unified memory; the step then accumulates over several
    #: micro-batches instead of failing.
    micro_batch_size: int = 8
    #: Candidate vocabulary entries recorded per generated token. Distillation
    #: objectives train on the distribution the policy actually considered, so
    #: it has to be captured while generating — it cannot be recovered later.
    #: Zero records nothing, which is what a purely on-policy objective wants.
    capture_topk: int = 0
    #: Upper bound, in GiB, on MLX's reusable buffer cache. MLX pools freed
    #: buffers to avoid re-allocating; across a long run of generate+backward
    #: cycles that pool grows until, on a memory-tight box, it crowds out the
    #: model and the machine swaps — generation then crawls and agent turns hit
    #: their timeout. Capping the pool returns buffers to the OS past the bound.
    #: Zero leaves MLX's default (effectively unbounded). Pair with the
    #: per-step ``clear_cache`` the training methods already call.
    cache_limit_gib: float = 0.0
    #: Prompt tokens fed per pass when filling the attention cache. A bare
    #: forward over the whole sequence builds a ``[heads, length, length]``
    #: attention tensor — 206 GB at 65k tokens with 24 heads — which no
    #: chunking of the output head can avoid. Prefilling in slices keeps that
    #: to ``[heads, slice, length]``. Zero forwards the sequence whole, which
    #: is faster for the short rows where it fits.
    prefill_step_size: int = 0
    #: Target positions scored per pass over the output head. The logits
    #: tensor is ``[micro_batch, chunk, vocab]``, and at a 250k vocabulary
    #: that single tensor is what decides whether a long sequence trains at
    #: all: 16k positions in one pass is 16 GB, in 1k chunks it is 1 GB. Zero
    #: scores the whole sequence at once, which is fastest for short rows.
    log_probs_chunk_size: int = 0
    #: Tokens per chunk of a GatedDeltaNet recurrence during training. mlx-lm
    #: runs that recurrence one token at a time, and differentiating its loop
    #: keeps several state matrices per token per crossed layer — about
    #: 10.5 MB a token on Qwen3.8-27B, which is what decides whether a hybrid
    #: model trains past its last full-attention block, and a chain of tiny
    #: kernels that is slow in both directions.
    #: :mod:`reef.train.mlx_backend.gated_delta` runs the chunkwise form
    #: instead — matrix products within a chunk, one state passed between
    #: chunks — so both the memory and the sequential steps are per chunk.
    #: 64 is the reference implementation's choice. Zero leaves mlx-lm's loop
    #: as it is. Irrelevant to a model without such a layer.
    recurrence_chunk_size: int = 64
    #: Recompute each adapted layer's forward during the backward instead of
    #: keeping its activations. Trades about a third more time per step for
    #: activation memory that no longer grows with the number of adapted
    #: layers, which is what lets every layer of a large model train at
    #: once: without it, all 64 layers of Qwen3.8-27B on a 700-token row
    #: hold 42 GB, most of it activations. Off by default; a handful of
    #: layers does not need it and is faster without.
    checkpoint_layers: bool = False
    #: Extra values handed to the chat template, the deployment's default for
    #: every request. A reasoning model's template is the usual reason to set
    #: one: Qwen3 opens a ``<think>`` block in the generation prompt unless
    #: ``enable_thinking`` is false, and those tokens are then response tokens
    #: like any other: they train, and the served reply carries them as
    #: ``reasoning_content`` rather than as the answer. A request may override
    #: this per call.
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must be non-empty")
        _reject_reserved_template_kwargs(self.chat_template_kwargs)
        if isinstance(self.capture_topk, bool) or not isinstance(self.capture_topk, int) or self.capture_topk < 0:
            raise ValueError("capture_topk must be a non-negative integer")
        for name in ("lora_layers", "lora_rank", "max_tokens", "micro_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.cache_limit_gib < 0:
            raise ValueError("cache_limit_gib must be non-negative")
        if not self.lora_keys:
            raise ValueError("lora_keys must name at least one projection")
        if self.prefill_step_size < 0 or isinstance(self.prefill_step_size, bool):
            raise ValueError("prefill_step_size must be a non-negative integer")
        if self.recurrence_chunk_size < 0 or isinstance(self.recurrence_chunk_size, bool):
            raise ValueError("recurrence_chunk_size must be a non-negative integer")
        if self.prefill_step_size and _lora_reaches_keys_or_values(self.lora_keys):
            # Prefilling is exact only while the cached tensors are constant
            # with respect to the trained weights. An adapted key or value
            # projection makes the prompt's cache a function of those weights,
            # so freezing it would drop that part of the gradient silently.
            # Queries are safe: a prompt position's query never reaches a
            # response position's output.
            raise ValueError(
                "prefill_step_size caches the prompt, which is exact only when the adapter leaves keys "
                f"and values alone; lora_keys={list(self.lora_keys)} adapts them. Adapt q_proj (and o_proj "
                "or the MLP) for long prompts, or set prefill_step_size to 0."
            )


@dataclass(frozen=True)
class Rollout:
    """One generated completion, kept token-native from sampling to training."""

    prompt_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    text: str
    finish_reason: str
    #: Per response token, the ids of the highest-probability candidates and
    #: their log-probs. Empty unless ``capture_topk`` asked for them.
    topk_indices: tuple[tuple[int, ...], ...] = ()
    topk_log_probs: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class _ObjectiveWeights:
    """How the two OpenClaw-RL terms combine, and how each is bounded."""

    w_rl: float
    w_opd: float
    eps_lo: float
    eps_hi: float
    diff_clip: float | None
    #: Penalty on divergence from the frozen base, the term slime exposes as
    #: ``--use-kl-loss`` with ``kl_loss_coef``. Zero leaves it out entirely,
    #: which is what the reference config does. It earns its place when the
    #: policy's whole output distribution sinks rather than only the tokens
    #: the distillation term targets: over 33 steps at lr 3e-5 the style
    #: markers fell 0.88 nats — the intended effect — but everything else
    #: fell 0.21, and the policy collapsed into emitting one tool call
    #: unconditionally. This holds the rest of the distribution in place.
    kl_coef: float = 0.0


@dataclass(frozen=True)
class _StepTensors:
    """One micro-batch, padded to a common width.

    Every array except ``sequences`` is laid out on target positions
    (``width - 1`` columns), so no member needs shifting at use.
    """

    sequences: mx.array
    rollout_log_probs: mx.array
    advantages: mx.array
    mask: mx.array


def _as_float(value: mx.array) -> float:
    """One scalar MLX array as a float.

    ``mx.array.item()`` is typed ``int | float | complex`` because MLX
    supports complex dtypes. Every array narrowed here is a real loss,
    log-probability or norm, so the conversion is total — but it has to be
    stated rather than assumed.
    """
    scalar = value.item()
    if isinstance(scalar, complex):
        raise MLXEngineError("expected a real scalar, got a complex one")
    return float(scalar)


def _flat(tree: Any) -> list[tuple[str, Any]]:
    """``tree_flatten`` as the list of pairs it returns without a destination.

    Its declared return type is ``list | dict`` because passing ``destination``
    changes the shape; nothing here passes one.
    """
    flattened = tree_flatten(tree)
    if not isinstance(flattened, list):
        raise MLXEngineError("tree_flatten returned a mapping where a list of pairs was expected")
    return flattened


def _as_int(value: Any) -> int:
    """One token id, whether MLX hands back a 0-d array or a plain int."""
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, complex):
            raise MLXEngineError("expected an integral token id, got a complex one")
        return int(scalar)
    return int(value)


def _token_log_probs(model: nn.Module, sequences: mx.array, mask: mx.array) -> mx.array:
    """Per-response-token log-probs, shaped like ``mask``.

    ``mask`` is 1 exactly on the target positions that correspond to response
    tokens, so the prompt contributes no gradient: the policy is only ever
    credited for tokens it chose.
    """
    logits = model(sequences[:, :-1])
    targets = sequences[:, 1:]
    # -cross_entropy is log p(target | context) without materializing a
    # [batch, length, vocab] log-softmax, which matters at 150k vocabularies.
    return -nn.losses.cross_entropy(logits, targets, reduction="none") * mask


def _head_holder(model: nn.Module) -> Any:
    """The module that owns the body and the output head, or None.

    mlx-lm models are consistently a body producing hidden states plus a head
    projecting them to the vocabulary, but the owner is the model itself for a
    plain text model and a nested ``language_model`` for one converted from a
    VLM checkpoint. Chunked scoring needs the two halves separately; when the
    shape is not recognised the caller scores the whole sequence instead.
    """
    node: Any = model
    for _ in range(4):
        if node is None:
            return None
        if hasattr(node, "model") and hasattr(node.model, "layers"):
            return node
        node = getattr(node, "language_model", None)
    return None


def _gradient_path_layers(model: nn.Module) -> list[nn.Module]:
    """The decoder layers a gradient crosses on its way to a trainable weight.

    The loss sits after the last layer, so the path runs from the lowest
    layer holding a trainable parameter up to the top; everything below it
    is a frozen prefix that no cotangent ever enters. When the model is not
    a recognisable body-plus-head, the whole model is the path.
    """
    holder = _head_holder(model)
    if holder is None:
        return [model]
    layers = list(holder.model.layers)
    trainable = [index for index, layer in enumerate(layers) if _flat(layer.trainable_parameters())]
    if not trainable:
        return []
    return layers[min(trainable) :]


@contextmanager
def _checkpointed_layers(layers: Sequence[nn.Module]) -> Iterator[None]:
    """Run the given layers under ``mx.checkpoint`` for the span of a backward.

    A checkpointed layer keeps its inputs and recomputes its forward when the
    backward needs the activations, so those stop accumulating layer by
    layer. Python resolves ``layer(x)`` on the class, so this is done the way
    mlx-lm's trainer does it, by wrapping the class's ``__call__`` — but for
    the span of one backward and put back after, so serving and every
    forward-only pass run the unwrapped layer.
    """
    classes = {type(layer) for layer in layers}
    originals = {cls: cls.__call__ for cls in classes}
    for cls, call in originals.items():

        def checkpointed(module, *args, call=call, **kwargs):
            def inner(params, *args, **kwargs):
                module.update(params)
                return call(module, *args, **kwargs)

            return mx.checkpoint(inner)(module.trainable_parameters(), *args, **kwargs)

        cls.__call__ = checkpointed
    try:
        yield
    finally:
        for cls, call in originals.items():
            cls.__call__ = call


def _lora_reaches_keys_or_values(lora_keys: Sequence[str]) -> bool:
    """Whether the adapter changes what earlier positions contribute.

    Caching a prompt is exact only when the cached tensors do not depend on
    the trained parameters. Queries are safe — a prompt position's query never
    reaches a response position's output — but adapted keys or values make the
    cache a function of the weights being trained, and freezing it then drops
    a real part of the gradient.
    """
    return any("k_proj" in key or "v_proj" in key for key in lora_keys)


#: Template arguments the engine owns. ``tokenize`` and ``add_generation_prompt``
#: decide what a prompt *is*, so letting a caller set them would break the
#: promise that the tokens which train are the tokens that were served.
_RESERVED_TEMPLATE_KWARGS = (
    "tokenize",
    "add_generation_prompt",
    "conversation",
    "messages",
    # `tools` has its own request field; accepting it here too would let one
    # request declare two different toolsets.
    "tools",
)


def _reject_reserved_template_kwargs(template_kwargs: Mapping[str, Any]) -> None:
    reserved = sorted(name for name in _RESERVED_TEMPLATE_KWARGS if name in template_kwargs)
    if reserved:
        raise ValueError(
            f"chat_template_kwargs may not set {reserved}: the engine owns how a prompt is rendered, "
            "so that the tokens which train are the tokens that were served"
        )


def _head_logits(holder: Any, hidden: mx.array) -> mx.array:
    """Project hidden states to vocabulary logits, tied or untied."""
    if hasattr(holder, "lm_head"):
        return holder.lm_head(hidden)
    return holder.model.embed_tokens.as_linear(hidden)


def _response_mask(prompt_lengths: mx.array, sequence_lengths: mx.array, width: int) -> mx.array:
    """1 on target positions holding a response token, 0 elsewhere.

    Target position ``i`` predicts ``sequences[i + 1]``, so the response spans
    ``prompt_length - 1 <= i < sequence_length - 1``.
    """
    positions = mx.arange(width - 1)[None, :]
    inside = (positions >= (prompt_lengths[:, None] - 1)) & (positions < (sequence_lengths[:, None] - 1))
    return inside.astype(mx.float32)


class MLXEngine:
    """Own the model, the adapter and the optimizer for one Reef deployment.

    Every MLX operation runs on one dedicated thread. MLX streams are
    per-thread — ``mlx_lm`` generation uses a thread-local stream, and a
    stream created on one thread does not exist on another — so a model
    loaded on the service's main thread cannot be generated from an arbitrary
    asyncio worker. Owning a single engine thread fixes that and, as a
    welcome consequence, serializes serving against training by construction:
    the colocated model is never read and written at once.
    """

    def __init__(self, config: MLXEngineConfig) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reef-mlx")
        self._publication = 0
        self._run(self._bootstrap)

    def _bootstrap(self) -> None:
        """Load the model on the engine thread that will also generate on it."""
        mx.random.seed(self._config.seed)
        if self._config.cache_limit_gib > 0:
            # Bound MLX's reusable buffer pool so a long serve+train run cannot
            # let it grow into swap. clear_cache after each step handles the
            # training buffers; this caps the generation side between steps.
            mx.set_cache_limit(int(self._config.cache_limit_gib * (2**30)))
        # `load` returns a 3-tuple only with return_config=True.
        self._model, self._tokenizer = load(self._config.model_path)[:2]
        # Eval mode is the engine's resting state, not something inherited
        # from ``load``: it is what selects the fast kernels for serving, and
        # ``_differentiable`` departs from it only for the span of a backward.
        self._model.eval()
        self._model.freeze()
        linear_to_lora_layers(self._model, self._config.lora_layers, self._lora_parameters())
        self._gradient_path = _gradient_path_layers(self._model)
        self._optimizer = optim.AdamW(
            learning_rate=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )
        self._holder = _head_holder(self._model)

    def _run(self, work: Callable[[], Any]) -> Any:
        """Execute ``work`` on the engine thread and re-raise what it raises."""
        return self._executor.submit(work).result()

    def _release_step_memory(self) -> float:
        """Return the backward's buffers to the OS; report live memory in GiB.

        A full-layer backward allocates large intermediates that MLX would
        otherwise keep pooled for reuse. Over a long serve+train run that pool
        grows until it crowds the model into swap and generation stalls, so a
        step drops it here. Runs on the engine thread (callers are already on
        it), and reports active (non-cache) bytes so a run can watch the true
        working set hold flat instead of climbing.
        """
        mx.clear_cache()
        return mx.get_active_memory() / (2**30)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    @property
    def config(self) -> MLXEngineConfig:
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    def _lora_parameters(self) -> dict[str, Any]:
        return {
            "rank": self._config.lora_rank,
            "scale": self._config.lora_scale,
            "dropout": self._config.lora_dropout,
            "keys": list(self._config.lora_keys),
        }

    @contextmanager
    def _differentiable(self) -> Iterator[None]:
        """Training mode on the gradient's path, for the span of one backward.

        mlx-lm's GatedDeltaNet carries two implementations of its recurrence
        and picks by ``training``: in eval mode a Metal kernel that has no
        vjp, in training mode a loop of plain ops that MLX can differentiate.
        ``load`` leaves the model in eval mode, where a gradient stops dead at
        the first GatedDeltaNet it meets — on a hybrid such as Qwen3.8 that
        confines LoRA to the last full-attention block and the MLP beside it,
        and ``[Primitive::vjp] Not implemented`` greets a third layer.

        The ops loop runs one token at a time and keeps every token's
        recurrent state for the backward pass, so it is switched on only
        where the gradient actually travels, and the loop itself is routed
        through :mod:`reef.train.mlx_backend.gated_delta`, which checkpoints
        it so that a long row crossing several such layers fits in unified
        memory. The frozen prefix keeps the kernel and its forward costs what
        it always did; serving and the forward-only scoring passes never
        enter here at all. On a model without such a layer, training mode
        changes nothing (LoRA dropout is the only other mode-dependent
        module, and it is off by default).
        """
        for layer in self._gradient_path:
            layer.train()
        try:
            with (
                chunked_gated_delta(self._config.recurrence_chunk_size),
                _checkpointed_layers(self._gradient_path if self._config.checkpoint_layers else []),
            ):
                yield
        finally:
            for layer in self._gradient_path:
                layer.eval()

    # ---------------------------------------------------------------- serving

    def render_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> list[int]:
        return self._run(lambda: self._render_prompt(messages, tools, template_kwargs))

    def _render_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> list[int]:
        """Tokenize a chat request exactly as generation will see it.

        The chat template is applied here and nowhere else, so the tokens that
        train are the tokens that were served. Skipping it makes a base model
        run to the token ceiling instead of emitting its end-of-turn marker.

        ``tools`` is rendered by the template, not by the caller: a template
        states the schemas *and* the one call syntax it will parse back. A
        request whose tools are dropped leaves the model guessing both, and it
        guesses differently every time — the same model invented six call
        syntaxes and a tool name that did not exist. Serving them is also what
        makes the turn trainable on what it actually saw.

        A request's ``template_kwargs`` override the deployment's defaults key
        by key, which is how one caller turns a reasoning model's ``<think>``
        block off without changing what the rest of the deployment serves.
        """
        extra = dict(self._config.chat_template_kwargs)
        if template_kwargs:
            _reject_reserved_template_kwargs(template_kwargs)
            extra.update(template_kwargs)
        if tools:
            extra["tools"] = [dict(tool) for tool in tools]
        prompt = self._tokenizer.apply_chat_template(
            prepare_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
            **extra,
        )
        return list(self._tokenizer.encode(prompt))

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Rollout:
        return self._run(lambda: self._generate(prompt_tokens, max_tokens, temperature))

    def generate_stream(
        self,
        prompt_tokens: Sequence[int],
        *,
        listener: GenerationListener,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Rollout:
        """``generate``, reporting each readable piece of text to ``listener`` as it is sampled.

        A listener that answers ``cancelled`` stops the generation at that
        token, and the partial rollout comes back with ``finish_reason``
        ``"cancelled"`` so nothing records it.
        """
        return self._run(lambda: self._generate(prompt_tokens, max_tokens, temperature, listener))

    def generate_batch(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Rollout]:
        """Sample one completion per prompt, sharing a single continuous batch.

        A thin driver over :class:`ContinuousBatcher`: submit every prompt and
        drain. All prompts are queued before the first ``step``, so they enter
        one batch together and decode in parallel, a finished sequence evicted
        rather than padded. To feed concurrent requests into a shared batch,
        drive :class:`ContinuousBatcher` directly (``submit`` then ``step``);
        single-sequence streaming stays on :meth:`generate_stream`.
        """
        batcher = ContinuousBatcher(self)
        tickets = [batcher.submit(prompt, max_tokens=max_tokens, temperature=temperature) for prompt in prompts]
        try:
            resolved = self._run(lambda: batcher.drain_on_engine_thread())
        finally:
            batcher.close()
        return [resolved[ticket] for ticket in tickets]

    # ------------------------------------------------ continuous serving batch

    def _serving_batcher(self) -> ContinuousBatcher:
        """The long-lived batcher the serving pump submits into.

        One per engine, created on first use, so tickets and the queue persist
        across pumps. It admits work in batch windows: prompts queued while the
        current batch decodes wait, and the next idle pump opens a fresh batch
        for all of them together (see :meth:`ContinuousBatcher._insert_queued`
        for why splicing into a live batch is unsafe on the hybrid cache).
        Distinct from :meth:`generate_batch`'s ephemeral batcher, which drains a
        fixed set of prompts on the engine thread in one call.
        """
        batcher = getattr(self, "_serving_batch", None)
        if batcher is None:
            batcher = ContinuousBatcher(self)
            self._serving_batch = batcher
        return batcher

    def submit_rollout(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> object:
        """Queue one completion for the serving batch; the ticket claims it.

        Thread-safe and non-blocking: the prompt enters the next batch window,
        which opens on the first :meth:`pump_rollouts` after the current batch
        drains (immediately if none is decoding). A backend fans concurrent
        requests in through this and pumps them together, so rollouts that
        arrive within one window share a single decode instead of serializing.
        """
        return self._serving_batcher().submit(prompt_tokens, max_tokens=max_tokens, temperature=temperature)

    def rollouts_pending(self) -> bool:
        """Whether the serving batch has queued or in-flight completions."""
        return self._serving_batcher().has_work()

    def pump_rollouts(self) -> list[tuple[object, Rollout]]:
        """Advance the serving batch one round; the rollouts that finished it.

        Runs on the engine thread. Each round decodes one token for every live
        sequence; when the batch is empty it admits the queued prompts as the
        next batch. Calling it in a loop while :meth:`rollouts_pending` holds
        drains the current batch and then opens the next — a submit between
        rounds joins the same decode only while the batch is still idle.
        """
        return self._serving_batcher().step()

    def _generate(
        self,
        prompt_tokens: Sequence[int],
        max_tokens: int | None,
        temperature: float | None,
        listener: GenerationListener | None = None,
    ) -> Rollout:
        """Sample one completion, recording the log-prob of every chosen token.

        The recorded value is the model's own log-softmax at the sampled
        token. It is the behaviour proxy the importance ratio divides by, so
        it must come from the engine that generated — Reef never reconstructs
        it by re-scoring decoded text.
        """
        temperature = self._config.temperature if temperature is None else temperature
        limit = self._config.max_tokens if max_tokens is None else max_tokens
        sampler = make_sampler(temp=temperature, top_p=self._config.top_p)
        prompt = mx.array(list(prompt_tokens))
        stop_ids = set(self._tokenizer.eos_token_ids)

        capture = self._config.capture_topk
        output: list[int] = []
        log_probs: list[float] = []
        topk_indices: list[tuple[int, ...]] = []
        topk_log_probs: list[tuple[float, ...]] = []
        finish_reason = "length"
        # mlx-lm's streaming detokenizer releases text only once a token
        # boundary has settled it, so a multi-byte character never reaches
        # the caller half-decoded. The rollout's own text is still decoded
        # whole below; the pieces are for showing, not for training.
        detokenizer = self._tokenizer.detokenizer if listener is not None else None
        if detokenizer is not None:
            detokenizer.reset()
        for token, step_log_probs in generate_step(prompt, self._model, max_tokens=limit, sampler=sampler):
            # mlx-lm yields a plain int here; older versions yield a 0-d array.
            token_id = _as_int(token)
            if token_id in stop_ids:
                finish_reason = "stop"
                break
            if listener is not None and listener.cancelled():
                finish_reason = "cancelled"
                break
            chosen = step_log_probs[token_id]
            if capture:
                # argpartition is enough: the objective works on a candidate
                # set, and never needs it ordered.
                candidates = mx.argpartition(-step_log_probs, kth=capture - 1)[:capture]
                values = step_log_probs[candidates]
                mx.eval(chosen, candidates, values)
                topk_indices.append(tuple(_as_int(value) for value in candidates))
                topk_log_probs.append(tuple(_as_float(value) for value in values))
            else:
                mx.eval(chosen)
            output.append(token_id)
            log_probs.append(_as_float(chosen))
            if detokenizer is not None and listener is not None:
                detokenizer.add_token(token_id)
                piece = detokenizer.last_segment
                if piece:
                    listener.emit(piece)
        if detokenizer is not None and listener is not None:
            detokenizer.finalize()
            piece = detokenizer.last_segment
            if piece:
                listener.emit(piece)
        return Rollout(
            prompt_tokens=tuple(int(value) for value in prompt_tokens),
            output_tokens=tuple(output),
            rollout_log_probs=tuple(log_probs),
            text=self._tokenizer.decode(output),
            finish_reason=finish_reason,
            topk_indices=tuple(topk_indices),
            topk_log_probs=tuple(topk_log_probs),
        )

    # --------------------------------------------------------------- training

    def base_log_probs(self, rows: Sequence[TrainingRow]) -> list[list[float]]:
        return self._run(lambda: self._base_log_probs(rows))

    def _base_log_probs(self, rows: Sequence[TrainingRow]) -> list[list[float]]:
        """Response log-probs under the frozen base, with the adapter disabled.

        LoRA computes ``x @ A @ B * scale``, so zeroing every ``lora_b`` turns
        the adapted model back into its base without a second copy of the
        weights in unified memory.
        """
        snapshot = self._adapter_snapshot()
        zeroed = {
            name: (mx.zeros_like(value) if name.endswith("lora_b") else value) for name, value in snapshot.items()
        }
        self._apply_adapter(zeroed)
        try:
            return self._response_log_probs(rows)
        finally:
            self._apply_adapter(snapshot)

    def _response_log_probs(self, rows: Sequence[TrainingRow]) -> list[list[float]]:
        collected: list[list[float]] = []
        for start in range(0, len(rows), self._config.micro_batch_size):
            chunk = rows[start : start + self._config.micro_batch_size]
            tensors = self._pack(chunk)
            values = _token_log_probs(self._model, tensors.sequences, tensors.mask)
            mx.eval(values)
            for index, row in enumerate(chunk):
                response_length = len(row.loss_mask)
                prompt_length = len(row.tokens) - response_length
                window = values[index, prompt_length - 1 : prompt_length - 1 + response_length]
                collected.append([_as_float(value) for value in window])
        return collected

    @staticmethod
    def _on_targets(row: TrainingRow, values: Sequence[float], width: int) -> list[float]:
        """Lay a per-response-token vector onto target positions.

        Response token ``j`` is predicted by target position
        ``prompt_length - 1 + j``, and there are ``width - 1`` target
        positions, so every packed vector lines up with the model's output
        without a shift at use.
        """
        prompt_length = len(row.tokens) - len(row.loss_mask)
        return [0.0] * (prompt_length - 1) + list(values) + [0.0] * (width - len(row.tokens))

    def _pack(self, rows: Sequence[TrainingRow]) -> _StepTensors:
        pad_id = self._tokenizer.pad_token_id
        pad_id = 0 if pad_id is None else int(pad_id)
        width = max(len(row.tokens) for row in rows)
        sequences = mx.array([list(row.tokens) + [pad_id] * (width - len(row.tokens)) for row in rows])
        sequence_lengths = mx.array([len(row.tokens) for row in rows])
        prompt_lengths = mx.array([len(row.tokens) - len(row.loss_mask) for row in rows])
        rollout = mx.array([self._on_targets(row, row.rollout_log_probs, width) for row in rows])
        advantages = mx.array([self._on_targets(row, row.advantages, width) for row in rows])
        mask = _response_mask(prompt_lengths, sequence_lengths, width)
        return _StepTensors(sequences, rollout, advantages, mask)

    def train_step(self, rows: Sequence[TrainingRow]) -> dict[str, Any]:
        return self._run(lambda: self._train_step(rows))

    def _train_step(self, rows: Sequence[TrainingRow]) -> dict[str, Any]:
        """Apply one TTT-Discover importance-sampling update over ``rows``.

        The objective is the reference's un-clipped surrogate,
        ``-exp(logπθ - logπrollout) · advantage`` summed over response tokens.
        Gradients accumulate across micro-batches so the update is the same
        full-batch sum however the batch is split for memory.
        """
        if not rows:
            raise MLXEngineError("a training step requires at least one row")
        # Policy log-probs before the update, computed once. They are the
        # step's staleness diagnostic: on a fresh rollout the ratio against
        # the recorded behaviour proxy is 1, and drift away from 1 is exactly
        # how far the batch has aged behind the serving weights.
        before = self._response_log_probs(rows)
        total_ratio = 0.0
        total_kl = 0.0
        counted = 0
        for row, current in zip(before, rows, strict=True):
            for policy, rollout in zip(row, current.rollout_log_probs, strict=True):
                total_ratio += math.exp(policy - rollout)
                total_kl += rollout - policy
                counted += 1

        total_loss = 0.0
        accumulated: dict[str, mx.array] | None = None
        for start in range(0, len(rows), self._config.micro_batch_size):
            chunk = rows[start : start + self._config.micro_batch_size]
            loss, flat = self._micro_batch_gradients(self._pack(chunk))
            accumulated = flat if accumulated is None else {n: accumulated[n] + g for n, g in flat.items()}
            total_loss += loss

        if accumulated is None:
            raise MLXEngineError("training step produced no gradients")
        self._optimizer.update(self._model, tree_unflatten(list(accumulated.items())))
        mx.eval(self._model.parameters(), self._optimizer.state)
        active_gib = self._release_step_memory()
        return {
            "loss": total_loss,
            "importance_ratio": total_ratio / counted if counted else 0.0,
            "ppo_kl": total_kl / counted if counted else 0.0,
            "response_tokens": counted,
            "rows": len(rows),
            "optimizer_step": _as_int(self._optimizer.step),
            "active_gib": active_gib,
        }

    # ---------------------------------------------------- distillation (OPD)

    def _response_logits(self, tokens: Sequence[int], response_length: int) -> mx.array:
        """Logits on the target positions that predict the response tokens."""
        return self._response_logits_of(self._model, tokens, response_length)

    def _teacher_candidate(
        self,
        candidate: TeacherCandidate,
        response_length: int,
        student_indices: mx.array,
        native_k: int,
    ) -> tuple[mx.array, mx.array]:
        """Frozen-base log-probs at the student's candidates, plus the base's own.

        The second return is the selection signal: which hint to believe is
        decided by how far the teacher's own preferred tokens overlap the
        policy's, so both have to come out of the same forward.
        """
        # The teacher runs forward only, so a cached prompt costs it nothing
        # in fidelity: there is no gradient to lose.
        cache = self._prefill(self._model, candidate.tokens, response_length)
        logits = self._response_logits_of(self._model, candidate.tokens, response_length, cache)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        at_student = mx.take_along_axis(log_probs, student_indices, axis=-1)
        native = mx.argpartition(-log_probs, kth=native_k - 1, axis=-1)[:, :native_k]
        mx.eval(at_student, native)
        return at_student, native

    def _teacher_rows(
        self, row: DistillationRow, native_k: int, *, with_reference: bool = False
    ) -> tuple[list[mx.array], list[mx.array], mx.array | None]:
        """Score every hint candidate under the frozen base.

        ``lora_b`` is zeroed for the whole sweep rather than per candidate:
        the base is the same model for all of them, and restoring between
        passes would only buy repeated work.

        ``with_reference`` also scores the row's *own* prompt — no hint — at
        the tokens the policy sampled. That is the reference distribution a KL
        penalty needs, and it costs one more forward inside a window where the
        adapter is already zeroed, rather than a second zero-and-restore.
        """
        snapshot = self._adapter_snapshot()
        zeroed = {
            name: (mx.zeros_like(value) if name.endswith("lora_b") else value) for name, value in snapshot.items()
        }
        self._apply_adapter(zeroed)
        try:
            student_indices = mx.array([list(candidates) for candidates in row.topk_indices])
            response_length = len(row.loss_mask)
            gathered = []
            native = []
            for candidate in row.candidates:
                at_student, own = self._teacher_candidate(candidate, response_length, student_indices, native_k)
                gathered.append(at_student)
                native.append(own)
            reference = None
            if with_reference:
                cache = self._prefill(self._model, row.tokens, response_length)
                logits = self._response_logits_of(self._model, row.tokens, response_length, cache)
                reference = mx.stop_gradient(self._sampled_log_probs(logits, row, response_length))
                mx.eval(reference)
            return gathered, native, reference
        finally:
            self._apply_adapter(snapshot)

    @staticmethod
    def _select_hint(native_topk: Sequence[mx.array], student_indices: mx.array, selection: str) -> mx.array:
        """Which hint to believe, per response token.

        ``shortest`` always takes candidate 0 — the rollout module orders them
        shortest first. ``sequence_optimal`` scores each candidate by how many
        of its own preferred tokens the policy also considered, summed over
        the response, and takes the best for every token in it.
        """
        rows = student_indices.shape[0]
        if selection == "shortest" or len(native_topk) == 1:
            return mx.zeros((rows,), dtype=mx.int32)
        if selection != "sequence_optimal":
            raise MLXEngineError(
                f"the mlx runtime implements hint selection 'shortest' and 'sequence_optimal', got {selection!r}"
            )
        overlaps = []
        for own in native_topk:
            equal = student_indices[:, :, None] == own[:, None, :]
            overlaps.append(mx.sum(mx.any(equal, axis=-1).astype(mx.int32), axis=-1))
        totals = mx.stack([mx.sum(overlap) for overlap in overlaps])
        mx.eval(totals)
        best = _as_int(mx.argmax(totals))
        return mx.full((rows,), best, dtype=mx.int32)

    def openclawrl_step(
        self,
        rows: Sequence[DistillationRow],
        *,
        w_rl: float,
        w_opd: float,
        eps_lo: float,
        eps_hi: float,
        diff_clip: float | None,
        hint_selection: str,
        native_k: int,
        kl_coef: float = 0.0,
    ) -> dict[str, Any]:
        """One OpenClaw-RL step: the reward term plus the distillation term.

        The reward term is a clipped surrogate on the tokens the policy
        sampled, carrying the turn's single accept/reject bit. The
        distillation term puts a teacher-weighted advantage on every candidate
        the policy considered. Both reduce as the reference does — the mean
        over a sample's response tokens, summed across samples — and combine
        as ``w_rl * reward + w_opd * distillation``.
        """
        return self._run(
            lambda: self._openclawrl_step(
                rows,
                w_rl=w_rl,
                w_opd=w_opd,
                eps_lo=eps_lo,
                eps_hi=eps_hi,
                diff_clip=diff_clip,
                hint_selection=hint_selection,
                native_k=native_k,
                kl_coef=kl_coef,
            )
        )

    def _openclawrl_step(
        self,
        rows: Sequence[DistillationRow],
        *,
        w_rl: float,
        w_opd: float,
        eps_lo: float,
        eps_hi: float,
        diff_clip: float | None,
        hint_selection: str,
        native_k: int,
        kl_coef: float = 0.0,
    ) -> dict[str, Any]:
        if not rows:
            raise MLXEngineError("a training step requires at least one row")
        if w_rl == 0.0 and w_opd == 0.0:
            # Both terms off is not a no-op: the optimizer still runs, and
            # AdamW decays the adapter on a zero gradient. A step that only
            # shrinks the weights it was asked not to train is worth refusing.
            raise MLXEngineError("openclawrl needs a non-zero w_rl or w_opd; both zero only decays the adapter")

        weights = _ObjectiveWeights(w_rl, w_opd, eps_lo, eps_hi, diff_clip, kl_coef)
        accumulated: dict[str, mx.array] | None = None
        total_loss = 0.0
        response_tokens = 0

        for row in rows:
            student_indices = mx.array([list(candidates) for candidates in row.topk_indices])
            teacher_gathered, teacher_native, reference = self._teacher_rows(
                row, native_k, with_reference=weights.kl_coef != 0.0
            )
            chosen = self._select_hint(teacher_native, student_indices, hint_selection)
            # The selection is constant over a response today, so one index
            # picks the hint rather than a per-token gather.
            teacher_log_probs = teacher_gathered[_as_int(chosen[0])]

            loss, flat = self._row_gradients(row, student_indices, teacher_log_probs, weights, reference)
            accumulated = flat if accumulated is None else {n: accumulated[n] + g for n, g in flat.items()}
            total_loss += loss
            response_tokens += len(row.loss_mask)

        if accumulated is None:
            raise MLXEngineError("training step produced no gradients")
        self._optimizer.update(self._model, tree_unflatten(list(accumulated.items())))
        mx.eval(self._model.parameters(), self._optimizer.state)
        active_gib = self._release_step_memory()
        return {
            "loss": total_loss,
            "rows": len(rows),
            "response_tokens": response_tokens,
            "optimizer_step": _as_int(self._optimizer.step),
            "active_gib": active_gib,
            "w_rl": w_rl,
            "w_opd": w_opd,
            "kl_coef": kl_coef,
        }

    def _row_gradients(
        self,
        row: DistillationRow,
        student_indices: mx.array,
        teacher_log_probs: mx.array,
        weights: _ObjectiveWeights,
        reference_log_probs: mx.array | None = None,
    ) -> tuple[float, dict[str, mx.array]]:
        """Loss and gradients for one judged turn.

        The reward term carries the turn's single accept/reject bit on the
        tokens the policy sampled; the distillation term carries the teacher's
        preference on every candidate it considered. A KL term, when a
        reference is supplied, keeps the rest of the distribution from
        drifting while those two reshape the part they target.
        """
        from reef.train.mlx_backend.objective import candidate_log_probs, opd_one_sample, policy_loss

        response_length = len(row.loss_mask)
        student_captured = mx.array([list(values) for values in row.topk_log_probs])
        mask = mx.array([float(value) for value in row.loss_mask])
        reward = float(row.reward)

        # The prompt is cached outside the traced function, so the gradient
        # covers the response positions only. That is the whole gradient here:
        # the config refuses a cached prefill unless the adapter leaves keys
        # and values alone, and an unadapted prompt cache is constant with
        # respect to the weights being trained.
        cache = self._prefill(self._model, row.tokens, response_length)

        def loss_fn(model: nn.Module) -> mx.array:
            logits = self._response_logits_of(model, row.tokens, response_length, cache)
            total = mx.zeros((), dtype=mx.float32)
            if weights.w_rl != 0.0:
                sampled = self._sampled_log_probs(logits, row, response_length)
                # The old actor is this actor: one optimizer step per rollout,
                # so the ratio starts at 1 and the surrogate sits at its
                # unclipped linear point.
                ppo_kl = mx.clip(mx.stop_gradient(sampled) - sampled, -20.0, 20.0)
                advantages = mx.full(sampled.shape, reward, dtype=mx.float32)
                per_token, _ = policy_loss(ppo_kl, advantages, weights.eps_lo, weights.eps_hi)
                total = total + weights.w_rl * self._sample_mean(per_token, mask)
            if weights.w_opd != 0.0:
                result = opd_one_sample(
                    candidate_log_probs(logits, student_indices),
                    student_indices=student_indices,
                    student_captured_log_probs=student_captured,
                    # subset_mode="student": the teacher was gathered at the
                    # student's own candidates, so the sets coincide.
                    teacher_indices=student_indices,
                    teacher_log_probs=teacher_log_probs,
                    eps_lo=weights.eps_lo,
                    eps_hi=weights.eps_hi,
                    diff_clip=weights.diff_clip,
                )
                total = total + weights.w_opd * self._sample_mean(result.per_token_pg, mask)
            if weights.kl_coef != 0.0 and reference_log_probs is not None:
                # k3: exp(d) - d - 1 with d = ref - current. Non-negative,
                # zero only where the policy still matches the base, and
                # unlike a plain difference it cannot be driven negative to
                # buy reward elsewhere.
                current = self._sampled_log_probs(logits, row, response_length)
                difference = mx.clip(reference_log_probs - current, -20.0, 20.0)
                per_token_kl = mx.exp(difference) - difference - 1.0
                total = total + weights.kl_coef * self._sample_mean(per_token_kl, mask)
            return total

        chunk = self._config.log_probs_chunk_size
        with self._differentiable():
            # Chunk the output head when the response is long enough to make the
            # [response, vocab] logits the memory ceiling — the same trade the
            # TTT-Discover path makes, so a long response trains without ever
            # materialising the whole logits tensor or its gradient. Only when
            # the prompt is not cached, since chunking runs the body whole.
            if chunk and self._holder is not None and cache is None and response_length > chunk:
                return self._chunked_row_gradients(
                    row, student_indices, teacher_log_probs, weights, reference_log_probs, chunk
                )
            loss, grads = nn.value_and_grad(self._model, loss_fn)(self._model)
            mx.eval(loss, grads)
        return _as_float(loss), dict(_flat(grads))

    def _chunked_row_gradients(
        self,
        row: DistillationRow,
        student_indices: mx.array,
        teacher_log_probs: mx.array,
        weights: _ObjectiveWeights,
        reference_log_probs: mx.array | None,
        chunk: int,
    ) -> tuple[float, dict[str, mx.array]]:
        """``_row_gradients`` without ever holding the whole ``[response, vocab]``.

        Same split as :meth:`_chunked_gradients`: run the body once for the
        response hidden states (``vocab / hidden`` times smaller than logits),
        score the objective over slices of those states so nothing larger than
        ``[chunk, vocab]`` exists at a time, then carry the assembled cotangent
        back through the body in one vjp. Every OPD/RL/KL term is a per-position
        sum reduced by the sample's constant token count, so summing the slices
        and dividing once reproduces the whole-response gradient exactly.
        """
        from reef.train.mlx_backend.objective import candidate_log_probs, opd_one_sample, policy_loss

        holder = self._holder
        response_length = len(row.loss_mask)
        start = len(row.tokens) - response_length - 1
        inputs = mx.array([list(row.tokens)])[:, :-1]

        mask = mx.array([float(value) for value in row.loss_mask])
        sampled_tokens = mx.array(list(row.tokens[-response_length:]))
        student_captured = mx.array([list(values) for values in row.topk_log_probs])
        reward = float(row.reward)
        # The _sample_mean denominator, shared by every term and constant with
        # respect to the weights, so it factors out of the per-slice sums.
        denom = mx.maximum(mx.sum(mask), mx.array(1.0))

        hidden = holder.model(inputs)
        mx.eval(hidden)
        response_hidden = hidden[:, start : start + response_length]

        total_loss = mx.zeros((), dtype=mx.float32)
        cotangents = []
        for begin in range(0, response_length, chunk):
            stop = min(begin + chunk, response_length)
            mask_s = mask[begin:stop]
            sampled_s = sampled_tokens[begin:stop]
            indices_s = student_indices[begin:stop]
            captured_s = student_captured[begin:stop]
            teacher_s = teacher_log_probs[begin:stop]
            reference_s = None if reference_log_probs is None else reference_log_probs[begin:stop]

            def slice_loss(
                hidden_slice: mx.array,
                mask_s: mx.array = mask_s,
                sampled_s: mx.array = sampled_s,
                indices_s: mx.array = indices_s,
                captured_s: mx.array = captured_s,
                teacher_s: mx.array = teacher_s,
                reference_s: mx.array | None = reference_s,
            ) -> mx.array:
                logits = _head_logits(holder, hidden_slice)[0]

                def sampled_log_probs() -> mx.array:
                    gathered = mx.take_along_axis(logits, sampled_s[:, None], axis=-1)[:, 0]
                    return gathered - mx.logsumexp(logits, axis=-1)

                total = mx.zeros((), dtype=mx.float32)
                if weights.w_rl != 0.0:
                    sampled = sampled_log_probs()
                    ppo_kl = mx.clip(mx.stop_gradient(sampled) - sampled, -20.0, 20.0)
                    advantages = mx.full(sampled.shape, reward, dtype=mx.float32)
                    per_token, _ = policy_loss(ppo_kl, advantages, weights.eps_lo, weights.eps_hi)
                    total = total + weights.w_rl * mx.sum(per_token * mask_s)
                if weights.w_opd != 0.0:
                    result = opd_one_sample(
                        candidate_log_probs(logits, indices_s),
                        student_indices=indices_s,
                        student_captured_log_probs=captured_s,
                        teacher_indices=indices_s,
                        teacher_log_probs=teacher_s,
                        eps_lo=weights.eps_lo,
                        eps_hi=weights.eps_hi,
                        diff_clip=weights.diff_clip,
                    )
                    total = total + weights.w_opd * mx.sum(result.per_token_pg * mask_s)
                if weights.kl_coef != 0.0 and reference_s is not None:
                    difference = mx.clip(reference_s - sampled_log_probs(), -20.0, 20.0)
                    per_token_kl = mx.exp(difference) - difference - 1.0
                    total = total + weights.kl_coef * mx.sum(per_token_kl * mask_s)
                return total / denom

            loss_slice, (cotangent,) = mx.vjp(slice_loss, [response_hidden[:, begin:stop]], [mx.array(1.0)])
            mx.eval(loss_slice, cotangent)
            total_loss = total_loss + loss_slice[0].astype(mx.float32)
            cotangents.append(cotangent)

        # The head reads only the response positions, so the body's cotangent is
        # the assembled response cotangent there and zero over the prompt.
        response_cotangent = mx.concatenate(cotangents, axis=1)
        prefix = mx.zeros((hidden.shape[0], start, hidden.shape[2]), dtype=response_cotangent.dtype)
        seed = mx.concatenate([prefix, response_cotangent], axis=1)

        trainable = _flat(self._model.trainable_parameters())
        names = [name for name, _ in trainable]

        def body(*params: mx.array) -> mx.array:
            self._model.update(tree_unflatten(list(zip(names, params, strict=True))))
            return holder.model(inputs)

        _, gradients = mx.vjp(body, [value for _, value in trainable], [seed])
        mx.eval(gradients)
        return _as_float(total_loss), dict(zip(names, gradients, strict=True))

    @staticmethod
    def _sample_mean(per_token: mx.array, mask: mx.array) -> mx.array:
        """The reference's reduction: a sample's mean over its response tokens."""
        return mx.sum(per_token * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))

    def _prefill(self, model: nn.Module, tokens: Sequence[int], response_length: int) -> Any:
        """Fill an attention cache with the prompt, a slice at a time.

        Returns ``None`` when prefilling is switched off or the prompt is
        shorter than one slice, in which case the caller forwards the sequence
        whole. Each slice is evaluated before the next, so the intermediate
        graph is released rather than accumulated.
        """
        step = self._config.prefill_step_size
        prompt_length = len(tokens) - response_length - 1
        if not step or prompt_length <= step:
            return None
        holder = _head_holder(model)
        if holder is None:
            return None
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(holder)
        for offset in range(0, prompt_length, step):
            chunk = mx.array([list(tokens[offset : min(offset + step, prompt_length)])])
            holder.model(chunk, cache=cache)
            mx.eval([entry.state for entry in cache])
        return cache

    def _response_logits_of(
        self,
        model: nn.Module,
        tokens: Sequence[int],
        response_length: int,
        cache: Any = None,
    ) -> mx.array:
        """Logits for the target positions that predict the response tokens.

        Response token ``j`` is predicted from target position
        ``prompt_length - 1 + j``, so this slice lines up with the captured
        candidates whatever the prompt length is — which matters because the
        teacher's prompt is longer than the student's by its hint.

        The hidden states are sliced *before* the output head. That is what
        makes a long prompt affordable: the head is the only part of the model
        whose activation is vocabulary-sized, so projecting the whole context
        would cost ``[context, vocab]`` — 130 GB at a 130k prompt and a 250k
        vocabulary — for a result of which only the response rows are ever
        read. The body still runs over the whole context, because the response
        attends to all of it.
        """
        start = len(tokens) - response_length - 1
        holder = _head_holder(model)
        if holder is None:
            # An architecture whose body and head cannot be told apart: fall
            # back to projecting everything, which is correct but costly.
            sequence = mx.array([list(tokens)])
            return model(sequence[:, :-1])[0, start : start + response_length]
        if cache is not None:
            # The prompt is already in the cache; only the response positions
            # still have to run, and their attention reaches back through it.
            tail = mx.array([list(tokens[start : len(tokens) - 1])])
            return _head_logits(holder, holder.model(tail, cache=cache))[0]
        sequence = mx.array([list(tokens)])
        hidden = holder.model(sequence[:, :-1])
        return _head_logits(holder, hidden[:, start : start + response_length])[0]

    @staticmethod
    def _sampled_log_probs(logits: mx.array, row: DistillationRow, response_length: int) -> mx.array:
        """Log-probs of the tokens the policy actually emitted."""
        sampled = mx.array(list(row.tokens[-response_length:]))[:, None]
        gathered = mx.take_along_axis(logits, sampled, axis=-1)[:, 0]
        return gathered - mx.logsumexp(logits, axis=-1)

    # ---------------------------------------------------------------- adapter

    def adapter_snapshot(self) -> dict[str, mx.array]:
        return self._run(self._adapter_snapshot)

    @staticmethod
    def _surrogate(
        log_probs: mx.array,
        rollout_log_probs: mx.array,
        advantages: mx.array,
        mask: mx.array,
    ) -> mx.array:
        """TTT-Discover's un-clipped importance-sampling surrogate.

        ``-exp(logpi_theta - logpi_rollout) * advantage``, summed over response
        tokens. Both gradient paths call this, so a chunked step and a
        whole-sequence step optimise the same objective by construction.

        The reduction is float32 even though the model runs in float16. This
        is a sum of thousands of same-magnitude terms whose signs follow the
        advantages, so it cancels heavily; accumulating it in float16 loses
        precision the gradient then inherits.
        """
        ratio = mx.exp(log_probs.astype(mx.float32) - rollout_log_probs.astype(mx.float32)) * mask
        return (-ratio * advantages * mask).astype(mx.float32).sum()

    def _micro_batch_gradients(self, tensors: _StepTensors) -> tuple[float, dict[str, mx.array]]:
        """Loss and gradients for one micro-batch of the TTT-Discover objective."""
        chunk = self._config.log_probs_chunk_size
        with self._differentiable():
            if chunk and self._holder is not None and tensors.mask.shape[1] > chunk:
                return self._chunked_gradients(tensors, chunk)

            def loss_fn(model: nn.Module) -> mx.array:
                log_probs = _token_log_probs(model, tensors.sequences, tensors.mask)
                # rollout_log_probs is already laid out on target positions, so
                # no shift is needed here.
                return self._surrogate(log_probs, tensors.rollout_log_probs, tensors.advantages, tensors.mask)

            loss, grads = nn.value_and_grad(self._model, loss_fn)(self._model)
            mx.eval(loss, grads)
        return _as_float(loss), dict(_flat(grads))

    def _chunked_gradients(self, tensors: _StepTensors, chunk: int) -> tuple[float, dict[str, mx.array]]:
        """The same gradients, without ever holding the full logits tensor.

        Three passes instead of one. The body runs first to produce hidden
        states, which are ``vocab / hidden`` times smaller than logits — about
        50x here. The loss is then scored over slices of those hidden states,
        each slice yielding its own gradient with respect to the hidden states
        and nothing larger than ``[batch, chunk, vocab]`` existing at a time.
        Because the objective is a plain sum over tokens, concatenating those
        slice gradients gives exactly the gradient the whole-sequence path
        would produce. A final vector-Jacobian product carries that cotangent
        back through the body into the adapter parameters.
        """
        holder = self._holder
        inputs = tensors.sequences[:, :-1]
        targets = tensors.sequences[:, 1:]

        hidden = holder.model(inputs)
        mx.eval(hidden)

        total_loss = mx.zeros((), dtype=mx.float32)
        cotangents = []
        for start in range(0, hidden.shape[1], chunk):
            stop = min(start + chunk, hidden.shape[1])
            target_slice = targets[:, start:stop]
            mask_slice = tensors.mask[:, start:stop]
            rollout_slice = tensors.rollout_log_probs[:, start:stop]
            advantage_slice = tensors.advantages[:, start:stop]

            def slice_loss(
                hidden_slice: mx.array,
                target_slice: mx.array = target_slice,
                mask_slice: mx.array = mask_slice,
                rollout_slice: mx.array = rollout_slice,
                advantage_slice: mx.array = advantage_slice,
            ) -> mx.array:
                logits = _head_logits(holder, hidden_slice)
                log_probs = -nn.losses.cross_entropy(logits, target_slice, reduction="none") * mask_slice
                return self._surrogate(log_probs, rollout_slice, advantage_slice, mask_slice)

            loss_slice, (cotangent,) = mx.vjp(slice_loss, [hidden[:, start:stop]], [mx.array(1.0)])
            mx.eval(loss_slice, cotangent)
            total_loss = total_loss + loss_slice[0].astype(mx.float32)
            cotangents.append(cotangent)

        # One backward through the body, seeded with the assembled cotangent.
        # Only parameters on a path to a trainable weight receive a gradient,
        # so a frozen prefix costs nothing here.
        seed = mx.concatenate(cotangents, axis=1)

        # mx.vjp differentiates a function of a flat list of arrays, so the
        # parameter tree is flattened here and rebuilt inside.
        trainable = _flat(self._model.trainable_parameters())
        names = [name for name, _ in trainable]

        def body(*params: mx.array) -> mx.array:
            self._model.update(tree_unflatten(list(zip(names, params, strict=True))))
            return holder.model(inputs)

        _, gradients = mx.vjp(body, [value for _, value in trainable], [seed])
        mx.eval(gradients)
        return _as_float(total_loss), dict(zip(names, gradients, strict=True))

    def _adapter_snapshot(self) -> dict[str, mx.array]:
        return {name: mx.array(value) for name, value in _flat(self._model.trainable_parameters())}

    def apply_adapter(self, snapshot: Mapping[str, mx.array]) -> None:
        self._run(lambda: self._apply_adapter(snapshot))

    def _apply_adapter(self, snapshot: Mapping[str, mx.array]) -> None:
        self._model.update(tree_unflatten(list(snapshot.items())))
        mx.eval(self._model.parameters())

    def adapter_delta(self, before: Mapping[str, mx.array], after: Mapping[str, mx.array]) -> tuple[float, int]:
        return self._run(lambda: self._adapter_delta(before, after))

    def _adapter_delta(self, before: Mapping[str, mx.array], after: Mapping[str, mx.array]) -> tuple[float, int]:
        """L2 distance and the count of tensors that actually moved.

        Reef publishes a candidate only when this is non-zero: a trainer that
        reports success while every weight is unchanged must not reach the
        artifact stack as a trained update.
        """
        total = mx.zeros((), dtype=mx.float32)
        changed = 0
        for name, old in before.items():
            new = after.get(name)
            if new is None or new.shape != old.shape:
                continue
            squared = mx.sum((new.astype(mx.float32) - old.astype(mx.float32)) ** 2)
            mx.eval(squared)
            total = total + squared
            if _as_float(squared) > 0.0:
                changed += 1
        mx.eval(total)
        return _as_float(total) ** 0.5, changed

    def next_runtime_load_id(self) -> str:
        """Mint the token that identifies these serving weights.

        Namespaced by process incarnation so a restart can never hand out a
        token an earlier incarnation already used for different weights.
        """
        self._publication += 1
        return f"mlx-{os.getpid()}-{self._publication}"

    def origin(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        import importlib.metadata

        def installed(package: str) -> str:
            try:
                return importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                return "unknown"

        versions = {package: installed(package) for package in ("mlx", "mlx-lm")}
        record: dict[str, Any] = {
            "schema": ORIGIN_SCHEMA,
            "base_model": self._config.model_path,
            "tokenizer": self._config.model_path,
            "lora_parameters": self._lora_parameters(),
            "libraries": versions,
            "rollout": "in-process mlx-lm generate_step",
        }
        if extra:
            record.update(extra)
        return record

    def save_adapter(self, destination: Path, *, origin_extra: Mapping[str, Any] | None = None) -> Path:
        return self._run(lambda: self._save_adapter(destination, origin_extra))

    def _save_adapter(self, destination: Path, origin_extra: Mapping[str, Any] | None = None) -> Path:
        """Write the adapter atomically: readers see the old one or the new one.

        Everything is written into a sibling directory, fsynced, and renamed
        into place, so an interrupted publication can never leave a partially
        written adapter where an activated one is expected.
        """
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            weights = dict(_flat(self._model.trainable_parameters()))
            if not weights:
                raise MLXEngineError("the model exposes no trainable parameters to publish")
            mx.save_safetensors(str(staging / ADAPTER_WEIGHTS), weights)
            # mlx-lm's own adapter shape: tuner.utils.load_adapters reads
            # exactly these keys, so the stock loader can serve the artifact.
            (staging / ADAPTER_CONFIG).write_text(
                json.dumps(
                    {
                        "fine_tune_type": "lora",
                        "num_layers": self._config.lora_layers,
                        "lora_parameters": self._lora_parameters(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (staging / ORIGIN).write_text(json.dumps(self.origin(extra=origin_extra), indent=2), encoding="utf-8")
            for name in (ADAPTER_WEIGHTS, ADAPTER_CONFIG, ORIGIN):
                handle = os.open(staging / name, os.O_RDONLY)
                try:
                    os.fsync(handle)
                finally:
                    os.close(handle)
            directory = os.open(staging, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination

    def load_adapter(self, source: Path) -> None:
        self._run(lambda: self._load_adapter(source))

    def _load_adapter(self, source: Path) -> None:
        """Load a published adapter's weights into the live model."""
        weights_path = Path(source) / ADAPTER_WEIGHTS
        if not weights_path.is_file():
            raise MLXEngineError(f"adapter at {source} has no {ADAPTER_WEIGHTS}")
        loaded = mx.load(str(weights_path))
        # mx.load returns an array for a .npy file and a mapping for
        # safetensors; only the mapping shape is a servable adapter.
        if not isinstance(loaded, dict) or not loaded:
            raise MLXEngineError(f"adapter at {source} carries no weight mapping")
        self._apply_adapter(loaded)


@dataclass
class _BatchSubmission:
    """A prompt queued for the batch, and the ticket its rollout comes back on."""

    ticket: object
    prompt: tuple[int, ...]
    max_tokens: int
    temperature: float
    capture_topk: int


@dataclass
class _BatchLive:
    """One sequence being decoded in the live batch, accumulating its rollout."""

    ticket: object
    prompt: tuple[int, ...]
    capture_topk: int
    tokens: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    topk_indices: list[tuple[int, ...]] = field(default_factory=list)
    topk_log_probs: list[tuple[float, ...]] = field(default_factory=list)
    finish_reason: str = "length"


class ContinuousBatcher:
    """Continuous batching over mlx-lm's ``BatchGenerator``.

    ``BatchGenerator`` is the continuous-batching engine: it inserts prompts
    into a running batch and evicts sequences as they finish, so a completed
    one is never padded through to the batch's end. What it does not surface
    through the ``batch_generate`` convenience is the per-token log-prob the
    OpenClaw-RL objective's diagnostic reads and the top-k the OPD candidate set
    needs — but each ``Response`` it yields carries the step's full log-softmax
    (``response.logprobs``), so this reads them straight off it.

    ``submit`` queues a prompt from any thread and returns a ticket. ``step``
    runs one generation round on the engine's MLX thread — draining the queue
    into the live batch and returning the rollouts that finished that round.
    Concurrent callers therefore share one decode. All MLX work stays on the
    engine's single thread; only the submission queue is touched from outside
    it, under a lock.
    """

    def __init__(self, engine: MLXEngine) -> None:
        self._engine = engine
        self._generator: Any = None
        self._queue: list[_BatchSubmission] = []
        self._queue_lock = threading.Lock()
        self._live: dict[int, _BatchLive] = {}

    def submit(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        capture_topk: int | None = None,
    ) -> object:
        """Queue a prompt; the returned ticket claims its rollout. Callable from
        any thread — the prompt enters the batch on the next :meth:`step`.
        """
        config = self._engine._config
        submission = _BatchSubmission(
            ticket=object(),
            prompt=tuple(prompt),
            max_tokens=config.max_tokens if max_tokens is None else max_tokens,
            temperature=config.temperature if temperature is None else temperature,
            capture_topk=config.capture_topk if capture_topk is None else capture_topk,
        )
        with self._queue_lock:
            self._queue.append(submission)
        return submission.ticket

    def has_work(self) -> bool:
        """Whether anything is queued or still decoding."""
        with self._queue_lock:
            return bool(self._queue) or bool(self._live)

    def step(self) -> list[tuple[object, Rollout]]:
        """One generation round on the engine thread; its finished rollouts as
        ``(ticket, rollout)``. This is the hook a service pumps between training
        steps so generation and training interleave on the one MLX thread.
        """
        return self._engine._run(self._step)

    def drain_on_engine_thread(self) -> dict[object, Rollout]:
        """Run everything submitted so far to completion, returning a rollout per
        ticket. Must run on the engine thread — call it through ``engine._run``
        (:meth:`MLXEngine.generate_batch` does).
        """
        resolved: dict[object, Rollout] = {}
        while self._live or self._has_queued():
            resolved.update(self._step())
        return resolved

    def close(self) -> None:
        """Release the generator's wired-memory limit. Idempotent."""
        if self._generator is not None:
            generator, self._generator = self._generator, None
            self._engine._run(generator.close)

    # ------------------------------------------------------ engine-thread only

    def _has_queued(self) -> bool:
        with self._queue_lock:
            return bool(self._queue)

    def _step(self) -> list[tuple[object, Rollout]]:
        self._insert_queued()
        if not self._live:
            return []
        finished: list[tuple[object, Rollout]] = []
        for response in self._generator.next_generated():
            live = self._live.get(response.uid)
            if live is None:
                continue
            # The stop token is not part of the reply — the single path breaks
            # without recording it. A length finish keeps its last token.
            if response.finish_reason != "stop":
                token_id = int(response.token)
                live.tokens.append(token_id)
                live.log_probs.append(float(response.logprobs[token_id]))
                if live.capture_topk:
                    candidates = mx.argpartition(-response.logprobs, kth=live.capture_topk - 1)[: live.capture_topk]
                    values = response.logprobs[candidates]
                    mx.eval(candidates, values)
                    live.topk_indices.append(tuple(_as_int(value) for value in candidates))
                    live.topk_log_probs.append(tuple(_as_float(value) for value in values))
            if response.finish_reason is not None:
                live.finish_reason = response.finish_reason
                finished.append((live.ticket, self._rollout(live)))
                self._live.pop(response.uid)
        if not self._live and not self._has_queued():
            # The batch drained. Return the decode's pooled buffers to the OS
            # now, rather than letting the generation-side pool grow between
            # the increasingly-spaced training steps (whose clear_cache is
            # otherwise the only reclaim) until it crowds a training backward.
            mx.clear_cache()
        return finished

    def _insert_queued(self) -> None:
        # Static batching: never splice a queued prompt into a live generation
        # batch. mlx-lm's continuous BatchGenerator mis-merges the Qwen3.5
        # hybrid cache when a fresh prompt (a GatedDeltaNet conv-state, sequence
        # length ~= the conv kernel) is inserted into a decoding batch
        # (attention KV, sequence length in the hundreds): the per-layer caches
        # concatenate on mismatched shapes and the decode Metal-OOMs. Draining
        # the current batch before admitting the next keeps every sequence in a
        # batch at one generation offset, so only the shape-safe filter() runs.
        # Sequences within a batch still decode in parallel; only whole batches
        # serialise.
        if self._live:
            return
        with self._queue_lock:
            pending, self._queue = self._queue, []
        if not pending:
            return
        if self._generator is None:
            self._generator = BatchGenerator(
                self._engine._model,
                stop_tokens=[[token] for token in self._engine._tokenizer.eos_token_ids],
            )
        top_p = self._engine._config.top_p
        uids = self._generator.insert(
            [list(submission.prompt) for submission in pending],
            [submission.max_tokens for submission in pending],
            samplers=[make_sampler(temp=submission.temperature, top_p=top_p) for submission in pending],
        )
        for submission, uid in zip(pending, uids, strict=True):
            self._live[uid] = _BatchLive(
                ticket=submission.ticket,
                prompt=submission.prompt,
                capture_topk=submission.capture_topk,
            )

    def _rollout(self, live: _BatchLive) -> Rollout:
        return Rollout(
            prompt_tokens=live.prompt,
            output_tokens=tuple(live.tokens),
            rollout_log_probs=tuple(live.log_probs),
            text=self._engine._tokenizer.decode(live.tokens),
            finish_reason=live.finish_reason,
            topk_indices=tuple(live.topk_indices),
            topk_log_probs=tuple(live.topk_log_probs),
        )


__all__ = [
    "ADAPTER_CONFIG",
    "ADAPTER_WEIGHTS",
    "ORIGIN",
    "ContinuousBatcher",
    "DistillationRow",
    "MLXEngine",
    "MLXEngineConfig",
    "MLXEngineError",
    "Rollout",
    "TeacherCandidate",
    "TrainingRow",
]
