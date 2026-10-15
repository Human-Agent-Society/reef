"""The training mechanisms of the MLX engine: rows, scoring, gradients and the optimizer step.

A loss family drives :class:`MLXTrainer` on the engine thread. The trainer
knows no objective: it packs rows, scores them, takes the gradient of a
:class:`TokenObjective` and applies the optimizer. All functions here run
on the engine thread.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.cache import make_prompt_cache

from reef.core.batches import TrajectoryItem
from reef.core.errors import ReefError
from reef.train.mlx_backend.config import MLXEngineConfig
from reef.train.mlx_backend.gated_delta import use_chunkwise_gated_delta
from reef.train.types.rows import policy_row_violation

BYTES_PER_GIB = 2**30

#: A prompt cache: one entry per layer, typed as ``make_prompt_cache`` types it.
PromptCache = list[Any]


class MLXEngineError(ReefError):
    """The MLX engine was asked for something it cannot do correctly."""


def real_scalar(value: mx.array) -> float:
    """A scalar MLX array as a float; MLX also has complex dtypes, which no loss uses."""
    scalar = value.item()
    if isinstance(scalar, complex):
        raise MLXEngineError("expected a real scalar, got a complex one")
    return float(scalar)


def flat_parameters(tree: object) -> list[tuple[str, mx.array]]:
    """A parameter tree as ``(dotted name, array)`` pairs."""
    flattened = tree_flatten(tree)
    if not isinstance(flattened, list):
        raise MLXEngineError("tree_flatten returned a mapping instead of a list of pairs")
    return flattened


@dataclass(frozen=True)
class ModelParts:
    """An mlx-lm model split into the body that makes hidden states and the head that makes logits.

    ``language_model`` owns the body and the head: the model itself, or its
    ``language_model`` child for a model converted from a vision-language
    checkpoint. ``lm_head`` is ``None`` when the head shares the input
    embedding.
    """

    model: nn.Module
    language_model: nn.Module
    body: nn.Module
    lm_head: nn.Module | None

    @classmethod
    def of(cls, model: nn.Module) -> ModelParts:
        """Find the parts; raise :class:`MLXEngineError` for a layout that is not an mlx-lm text model."""
        language_model = model.children().get("language_model", model)
        if not isinstance(language_model, nn.Module):
            raise MLXEngineError("the model's language_model is not a module")
        body = language_model.children().get("model")
        if not isinstance(body, nn.Module) or not isinstance(body.children().get("layers"), list):
            raise MLXEngineError("the model has no body with decoder layers; it is not an mlx-lm text model")
        lm_head = language_model.children().get("lm_head")
        if lm_head is None and not isinstance(body.children().get("embed_tokens"), nn.Module):
            raise MLXEngineError("the model has neither an output head nor input embeddings to share")
        if lm_head is not None and not isinstance(lm_head, nn.Module):
            raise MLXEngineError("the model's lm_head is not a module")
        return cls(model, language_model, body, lm_head)

    @property
    def layers(self) -> list[nn.Module]:
        return list(self.body.children()["layers"])

    def logits(self, hidden: mx.array) -> mx.array:
        """Project hidden states to vocabulary logits."""
        if self.lm_head is None:
            return self.body.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)


def gradient_path_layers(parts: ModelParts) -> list[nn.Module]:
    """The decoder layers from the lowest one with a trainable parameter up to the last.

    The loss is computed after the last layer, so a gradient reaches a
    trainable weight only through these layers. The layers below them are a
    frozen prefix.
    """
    layers = parts.layers
    trainable = [index for index, layer in enumerate(layers) if flat_parameters(layer.trainable_parameters())]
    return layers[min(trainable) :] if trainable else []


@contextmanager
def checkpointed_layers(layers: Sequence[nn.Module]) -> Iterator[None]:
    """Run ``layers`` under ``mx.checkpoint`` for the block.

    A checkpointed layer keeps its inputs and computes its forward again in the
    backward pass, so its activations are not kept. Python looks ``layer(x)``
    up on the class, so the block wraps each class's ``__call__``, as mlx-lm's
    trainer does, and restores it on exit.
    """
    classes = {type(layer) for layer in layers}
    originals = {cls: cls.__call__ for cls in classes}
    for cls, call in originals.items():

        def checkpointed(module, *args, call=call, **kwargs):
            def inner(params, *args, **kwargs):
                module.update(params)
                return call(module, *args, **kwargs)

            return mx.checkpoint(inner)(module.trainable_parameters(), *args, **kwargs)

        cls.__call__ = checkpointed  # type: ignore[method-assign]
    try:
        yield
    finally:
        for cls, call in originals.items():
            cls.__call__ = call  # type: ignore[method-assign]


@dataclass(frozen=True)
class ResponseTokens:
    """A prompt followed by a response: the unit the trainer scores."""

    tokens: tuple[int, ...]
    response_length: int

    def __post_init__(self) -> None:
        if self.response_length <= 0 or len(self.tokens) <= self.response_length:
            raise ValueError("a scored sequence needs at least one prompt token and one response token")

    @property
    def prompt_length(self) -> int:
        return len(self.tokens) - self.response_length


@dataclass(frozen=True)
class TrainingRow(ResponseTokens):
    """One reserved sample with one value per response token for the mask, the rollout log probability and the advantage."""

    loss_mask: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    advantages: tuple[float, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        lengths = {len(self.loss_mask), len(self.rollout_log_probs), len(self.advantages)}
        if lengths != {self.response_length}:
            raise ValueError("loss_mask, rollout_log_probs and advantages must cover exactly the response tokens")

    @classmethod
    def from_item(cls, item: TrajectoryItem, advantage: float) -> TrainingRow:
        """The row of one trajectory, with its trajectory advantage on every response token."""
        training = item.training
        tokens = tuple(training.get("tokens", ()))
        loss_mask = tuple(training.get("loss_mask", ()))
        rollout_log_probs = tuple(training.get("rollout_log_probs", ()))
        violation = policy_row_violation(tokens, loss_mask, rollout_log_probs)
        if violation:
            raise ValueError(f"invalid MLX training row: {violation}")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
            raise ValueError("MLX training tokens must be non-negative integers")
        if not math.isfinite(advantage):
            raise ValueError("MLX advantages must be finite")
        return cls(
            tokens=tokens,
            response_length=len(loss_mask),
            loss_mask=tuple(int(value) for value in loss_mask),
            rollout_log_probs=tuple(float(value) for value in rollout_log_probs),
            advantages=(float(advantage),) * len(loss_mask),
        )


@dataclass(frozen=True)
class PackedRows:
    """Rows padded to one width and scored from target position ``start``.

    Target position ``i`` predicts token ``i + 1``. The scored positions are
    ``[start, width - 1)``, where ``start`` is the shortest prompt length
    minus one. ``targets`` and ``response_mask`` have one column per scored
    position; the mask is 1 on response tokens and 0 on prompt and padding.
    """

    rows: tuple[ResponseTokens, ...]
    sequences: mx.array
    start: int

    @classmethod
    def of(cls, rows: Sequence[ResponseTokens], pad_token_id: int) -> PackedRows:
        if not rows:
            raise MLXEngineError("packing needs at least one row")
        width = max(len(row.tokens) for row in rows)
        sequences = mx.array([[*row.tokens, *([pad_token_id] * (width - len(row.tokens)))] for row in rows])
        return cls(tuple(rows), sequences, min(row.prompt_length for row in rows) - 1)

    @property
    def targets(self) -> mx.array:
        return self.sequences[:, self.start + 1 :]

    @property
    def scored_count(self) -> int:
        return int(self.sequences.shape[1]) - 1 - self.start

    @property
    def response_mask(self) -> mx.array:
        return self.on_scored([[1.0] * row.response_length for row in self.rows])

    def on_scored(self, values: Sequence[Sequence[float]]) -> mx.array:
        """Place one response-length vector per row on the scored positions; zero elsewhere."""
        laid_out = []
        for row, row_values in zip(self.rows, values, strict=True):
            offset = row.prompt_length - 1 - self.start
            laid_out.append(
                [0.0] * offset + list(row_values) + [0.0] * (self.scored_count - offset - row.response_length)
            )
        return mx.array(laid_out)

    def response_values(self, values: mx.array) -> list[list[float]]:
        """The response window of each row of a ``[rows, scored]`` array, as floats."""
        windows = []
        for index, row in enumerate(self.rows):
            offset = row.prompt_length - 1 - self.start
            windows.append([real_scalar(value) for value in values[index, offset : offset + row.response_length]])
        return windows


class TokenObjective(ABC):
    """A training loss over the scored positions of one packed batch.

    The trainer may score the positions in slices and add the slice losses,
    so the loss must be a sum over positions. A constant normaliser belongs
    in each position's term.
    """

    @abstractmethod
    def loss(self, logits: mx.array, begin: int, stop: int) -> mx.array:
        """The scalar loss of scored positions ``[begin, stop)``; ``logits`` is ``[rows, stop - begin, vocab]``."""


class MLXTrainer:
    """Score, differentiate and optimize one adapted model, without any objective of its own.

    The AdamW state lives as long as the trainer, so each training step
    continues the moments and the step count of the previous one.
    """

    def __init__(self, parts: ModelParts, config: MLXEngineConfig, pad_token_id: int) -> None:
        self._parts = parts
        self._config = config
        self._pad_token_id = pad_token_id
        self._gradient_path = gradient_path_layers(parts)
        self._optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)

    @property
    def micro_batch_size(self) -> int:
        return self._config.micro_batch_size

    @property
    def gradient_path(self) -> list[nn.Module]:
        return list(self._gradient_path)

    def pack(self, rows: Sequence[ResponseTokens]) -> PackedRows:
        return PackedRows.of(rows, self._pad_token_id)

    # -- Adapter tensors

    def adapter_tensors(self) -> dict[str, mx.array]:
        """A copy of every trainable tensor."""
        return {name: mx.array(value) for name, value in flat_parameters(self._parts.model.trainable_parameters())}

    def load_adapter_tensors(self, tensors: Mapping[str, mx.array]) -> None:
        self._parts.model.update(tree_unflatten(list(tensors.items())))
        mx.eval(self._parts.model.parameters())

    @contextmanager
    def base_model(self) -> Iterator[None]:
        """Score the frozen base model for the block.

        LoRA adds ``x @ A @ B * scale``, so a zero ``lora_b`` turns the adapted
        model into its base without a second copy of the weights.
        """
        snapshot = self.adapter_tensors()
        self.load_adapter_tensors(
            {name: mx.zeros_like(value) if name.endswith("lora_b") else value for name, value in snapshot.items()}
        )
        try:
            yield
        finally:
            self.load_adapter_tensors(snapshot)

    # -- Forward passes

    def response_logits(self, packed: PackedRows) -> mx.array:
        """Logits of the scored positions, ``[rows, scored, vocab]``, without a gradient."""
        logits = self._scored_logits(packed, self._prefill(packed))
        mx.eval(logits)
        return logits

    def response_log_probs(self, packed: PackedRows) -> mx.array:
        """Log probabilities of the target tokens on the scored positions, zero outside the responses."""
        logits = self._scored_logits(packed, self._prefill(packed))
        log_probs = -nn.losses.cross_entropy(logits, packed.targets, reduction="none") * packed.response_mask
        mx.eval(log_probs)
        return log_probs

    def _prefill(self, packed: PackedRows) -> PromptCache | None:
        """Fill a prompt cache with the positions before ``start``, ``prefill_step_size`` tokens per pass.

        Returns ``None`` when prefilling is off or the prompt is not longer
        than one pass; the caller then runs the whole sequence.
        """
        step = self._config.prefill_step_size
        if not step or packed.start <= step:
            return None
        cache = make_prompt_cache(self._parts.language_model)
        for offset in range(0, packed.start, step):
            self._parts.body(packed.sequences[:, offset : min(offset + step, packed.start)], cache=cache)
            mx.eval([entry.state for entry in cache])
        return cache

    def _scored_logits(self, packed: PackedRows, cache: PromptCache | None) -> mx.array:
        """Run the body and project only the scored positions; the head is the one vocabulary-sized activation."""
        inputs = packed.sequences[:, :-1]
        if cache is None:
            hidden = self._parts.body(inputs)[:, packed.start :]
        else:
            hidden = self._parts.body(inputs[:, packed.start :], cache=cache)
        return self._parts.logits(hidden)

    # -- Gradients and the optimizer

    @contextmanager
    def _differentiable(self) -> Iterator[None]:
        """Training mode on the gradient path for the block.

        mlx-lm's GatedDeltaNet uses a Metal kernel without a vjp in eval mode
        and differentiable operations in training mode. Only the gradient path
        switches; the frozen prefix and every forward-only pass keep the kernel.
        """
        for layer in self._gradient_path:
            layer.train()
        try:
            with (
                use_chunkwise_gated_delta(self._config.recurrence_chunk_size),
                checkpointed_layers(self._gradient_path if self._config.checkpoint_layers else []),
            ):
                yield
        finally:
            for layer in self._gradient_path:
                layer.eval()

    def gradients(self, packed: PackedRows, objective: TokenObjective) -> tuple[float, dict[str, mx.array]]:
        """The loss of ``objective`` on ``packed`` and its gradient for every trainable tensor."""
        cache = self._prefill(packed)
        chunk_size = self._config.log_probs_chunk_size
        with self._differentiable():
            if chunk_size and cache is None and packed.scored_count > chunk_size:
                return self._chunked_gradients(packed, objective, chunk_size)

            def loss_of(model: nn.Module) -> mx.array:
                return objective.loss(self._scored_logits(packed, cache), 0, packed.scored_count)

            loss, gradients = nn.value_and_grad(self._parts.model, loss_of)(self._parts.model)
            mx.eval(loss, gradients)
        return real_scalar(loss), dict(flat_parameters(gradients))

    def _chunked_gradients(
        self, packed: PackedRows, objective: TokenObjective, chunk_size: int
    ) -> tuple[float, dict[str, mx.array]]:
        """:meth:`gradients` without the whole ``[rows, scored, vocab]`` logits.

        The body runs once without a gradient. Each slice of the scored hidden
        states then passes through the head and the objective with its own
        vjp, so at most ``[rows, chunk_size, vocab]`` logits exist at a time.
        The objective is a sum over positions, so the slice cotangents together
        are the whole one. A last vjp carries them through the body.
        """
        inputs = packed.sequences[:, :-1]
        hidden = self._parts.body(inputs)
        mx.eval(hidden)
        scored = hidden[:, packed.start :]
        total_loss = mx.zeros((), dtype=mx.float32)
        cotangents = [mx.zeros((hidden.shape[0], packed.start, hidden.shape[2]), dtype=hidden.dtype)]
        for begin in range(0, packed.scored_count, chunk_size):
            stop = min(begin + chunk_size, packed.scored_count)

            def slice_loss(hidden_slice: mx.array, begin: int = begin, stop: int = stop) -> mx.array:
                return objective.loss(self._parts.logits(hidden_slice), begin, stop)

            outputs, (cotangent,) = mx.vjp(slice_loss, [scored[:, begin:stop]], [mx.array(1.0)])
            mx.eval(outputs, cotangent)
            total_loss = total_loss + outputs[0].astype(mx.float32)
            cotangents.append(cotangent)
        seed = mx.concatenate(cotangents, axis=1)

        # mx.vjp differentiates a function of a flat list of arrays, so the
        # trainable tree is flattened here and rebuilt inside.
        trainable = flat_parameters(self._parts.model.trainable_parameters())
        names = [name for name, _ in trainable]

        def body(*parameters: mx.array) -> mx.array:
            self._parts.model.update(tree_unflatten(list(zip(names, parameters, strict=True))))
            return self._parts.body(inputs)

        _, gradients = mx.vjp(body, [value for _, value in trainable], [seed])
        mx.eval(gradients)
        return real_scalar(total_loss), dict(zip(names, gradients, strict=True))

    def optimizer_step(self, passes: Iterable[tuple[PackedRows, TokenObjective]]) -> dict[str, float | int]:
        """Sum the gradients of every pass, apply one AdamW update, and release the step's buffers.

        Reports the summed loss, the optimizer's step count and the active
        (non-cache) memory in GiB after the step.
        """
        total_loss = 0.0
        accumulated: dict[str, mx.array] = {}
        for packed, objective in passes:
            loss, gradients = self.gradients(packed, objective)
            accumulated = (
                {name: accumulated[name] + gradient for name, gradient in gradients.items()}
                if accumulated
                else gradients
            )
            total_loss += loss
        self._optimizer.update(self._parts.model, tree_unflatten(list(accumulated.items())))
        mx.eval(self._parts.model.parameters(), self._optimizer.state)
        mx.clear_cache()
        return {
            "loss": total_loss,
            "optimizer_step": int(real_scalar(self._optimizer.step)),
            "active_gib": mx.get_active_memory() / BYTES_PER_GIB,
        }


__all__ = [
    "BYTES_PER_GIB",
    "MLXEngineError",
    "MLXTrainer",
    "ModelParts",
    "PackedRows",
    "ResponseTokens",
    "TokenObjective",
    "TrainingRow",
    "flat_parameters",
    "gradient_path_layers",
    "real_scalar",
]
