"""The MLX engine: one base model, one LoRA adapter and one optimizer for the life of the service.

Every MLX operation runs on one engine thread. MLX streams are per thread,
so arrays made on one thread cannot be used on another, and the one thread
also serializes generation against training on the shared weights.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner import linear_to_lora_layers

from reef.artifact.peft import ADAPTER_CONFIG, ADAPTER_METADATA, METADATA_SCHEMA, file_sha256
from reef.core.batches import TrajectoryItem
from reef.core.chat_protocol import ParsedToolCall, template_messages
from reef.runtime.recovery import fsync_dir, mkdir_durable, write_json
from reef.train.mlx_backend.config import MLXEngineConfig, check_template_kwargs
from reef.train.mlx_backend.contracts import AdapterSnapshot, GenerationListener, LossOptions, ModelEngine, Rollout
from reef.train.mlx_backend.losses import resolve_mlx_loss
from reef.train.mlx_backend.training import (
    BYTES_PER_GIB,
    MLXEngineError,
    MLXTrainer,
    ModelParts,
    flat_parameters,
    real_scalar,
)

#: mlx-lm's adapter weights file; ``mlx_lm.tuner.utils.load_adapters`` reads it beside ``adapter_config.json``.
MLX_ADAPTER_WEIGHTS = "adapters.safetensors"

#: The chat template writes the opening thinking tag last, so the last few prompt tokens show it.
REASONING_TAIL_TOKENS = 8


@dataclass(frozen=True)
class MLXAdapterSnapshot(AdapterSnapshot):
    """Copies of the adapter's trainable tensors, made on the engine thread."""

    tensors: Mapping[str, mx.array]


@dataclass(frozen=True)
class LoadedModel:
    """What the engine thread loads once: the tokenizer, the model parts and the trainer."""

    tokenizer: TokenizerWrapper
    parts: ModelParts
    trainer: MLXTrainer


class MLXEngine(ModelEngine):
    """Generate, train and store one LoRA adapter on an mlx-lm model."""

    def __init__(self, model_path: str, config: MLXEngineConfig) -> None:
        if not model_path:
            raise ValueError("the mlx engine requires a model path")
        self._model_path = model_path
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reef-mlx")
        try:
            self._loaded = self.run(self._load)
        except BaseException:
            self._executor.shutdown(wait=True)
            raise

    def _load(self) -> LoadedModel:
        mx.random.seed(self._config.seed)
        if self._config.cache_limit_gib:
            mx.set_cache_limit(int(self._config.cache_limit_gib * BYTES_PER_GIB))
        model, tokenizer = load(self._model_path)[:2]
        # Eval mode selects the fast kernels for serving; the trainer leaves
        # it only on the gradient path and only for a backward pass.
        model.eval()
        model.freeze()
        linear_to_lora_layers(model, self._config.lora_layers, self._lora_parameters())
        if not flat_parameters(model.trainable_parameters()):
            raise MLXEngineError(f"lora_keys {list(self._config.lora_keys)} match no module of the adapted layers")
        parts = ModelParts.of(model)
        pad_token_id = tokenizer.pad_token_id
        trainer = MLXTrainer(parts, self._config, 0 if pad_token_id is None else int(pad_token_id))
        return LoadedModel(tokenizer, parts, trainer)

    def run[T](self, work: Callable[[], T]) -> T:
        """Run ``work`` on the engine thread and return its result or raise its exception."""
        return self._executor.submit(work).result()

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def config(self) -> MLXEngineConfig:
        return self._config

    @property
    def trainer(self) -> MLXTrainer:
        """The training mechanisms; call them only inside :meth:`run`."""
        return self._loaded.trainer

    def _lora_parameters(self) -> dict[str, Any]:
        return {
            "rank": self._config.lora_rank,
            "scale": self._config.lora_scale,
            "dropout": self._config.lora_dropout,
            "keys": list(self._config.lora_keys),
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    # -- Prompts and replies

    def render_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> list[int]:
        """Apply the chat template and tokenize; generation and training both read these tokens.

        The template also renders ``tools``: it states their schemas and the
        call syntax it parses back. A request's ``template_kwargs`` override
        the deployment's ``chat_template_kwargs`` key by key.
        """
        render_kwargs = dict(self._config.chat_template_kwargs)
        if template_kwargs:
            check_template_kwargs(template_kwargs)
            render_kwargs.update(template_kwargs)
        if tools:
            render_kwargs["tools"] = [dict(tool) for tool in tools]
        rendered_messages = template_messages(list(messages))
        tokenizer = self._loaded.tokenizer

        def render() -> list[int]:
            prompt = tokenizer.apply_chat_template(
                rendered_messages, tokenize=False, add_generation_prompt=True, **render_kwargs
            )
            return list(tokenizer.encode(prompt))

        return self.run(render)

    def opens_reasoning(self, prompt_tokens: Sequence[int]) -> bool:
        tail = self._loaded.tokenizer.decode(list(prompt_tokens[-REASONING_TAIL_TOKENS:]))
        return str(tail).rstrip().endswith("<think>")

    @property
    def tool_call_markers(self) -> tuple[str, str] | None:
        tokenizer = self._loaded.tokenizer
        if not tokenizer.has_tool_calling:
            return None
        return tokenizer.tool_call_start, tokenizer.tool_call_end

    def parse_tool_call(self, body: str, tools: Sequence[Mapping[str, Any]]) -> ParsedToolCall:
        """Parse one call with the parser mlx-lm chose for the chat template; arguments follow their schema types."""
        parsed = self._loaded.tokenizer.tool_parser(body, [dict(tool) for tool in tools])
        name = parsed.get("name") if isinstance(parsed, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValueError("a tool call names no function")
        arguments = parsed.get("arguments")
        if arguments is None:
            arguments = {}
        return ParsedToolCall(
            name, arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        )

    # -- Generation

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        listener: GenerationListener | None = None,
    ) -> Rollout:
        """Sample one completion and record the log probability of each sampled token.

        The recorded value is the engine's own log-softmax at the sampled
        token: the behaviour log probability of the importance ratio. A
        ``listener`` receives each piece of text that the detokenizer settles;
        when it cancels, the rollout ends with ``finish_reason`` ``cancelled``.
        """
        return self.run(
            lambda: self._generate(
                prompt_tokens,
                self._config.max_tokens if max_tokens is None else max_tokens,
                self._config.temperature if temperature is None else temperature,
                listener,
            )
        )

    def _generate(
        self,
        prompt_tokens: Sequence[int],
        max_tokens: int,
        temperature: float,
        listener: GenerationListener | None,
    ) -> Rollout:
        tokenizer = self._loaded.tokenizer
        stop_ids = set(tokenizer.eos_token_ids)
        capture_topk = self._config.capture_topk
        sampler = make_sampler(temp=temperature, top_p=self._config.top_p)
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        output: list[int] = []
        log_probs: list[float] = []
        topk_indices: list[tuple[int, ...]] = []
        topk_log_probs: list[tuple[float, ...]] = []
        finish_reason = "length"
        prompt = mx.array(list(prompt_tokens))
        for sampled, step_log_probs in generate_step(
            prompt, self._loaded.parts.model, max_tokens=max_tokens, sampler=sampler
        ):
            token = int(sampled)
            if token in stop_ids:
                finish_reason = "stop"
                break
            if listener is not None and listener.cancelled():
                finish_reason = "cancelled"
                break
            chosen = step_log_probs[token]
            if capture_topk:
                # The loss reads the candidates as a set, so their order does not matter.
                candidates = mx.argpartition(-step_log_probs, kth=capture_topk - 1)[:capture_topk]
                values = step_log_probs[candidates]
                mx.eval(chosen, candidates, values)
                topk_indices.append(tuple(int(real_scalar(index)) for index in candidates))
                topk_log_probs.append(tuple(real_scalar(value) for value in values))
            else:
                mx.eval(chosen)
            output.append(token)
            log_probs.append(real_scalar(chosen))
            if listener is not None:
                detokenizer.add_token(token)
                if piece := detokenizer.last_segment:
                    listener.emit(piece)
        if listener is not None:
            detokenizer.finalize()
            if piece := detokenizer.last_segment:
                listener.emit(piece)
        return Rollout(
            prompt_tokens=tuple(int(token) for token in prompt_tokens),
            output_tokens=tuple(output),
            rollout_log_probs=tuple(log_probs),
            text=tokenizer.decode(output),
            finish_reason=finish_reason,
            topk_indices=tuple(topk_indices),
            topk_log_probs=tuple(topk_log_probs),
        )

    # -- Training

    def check_loss(self, loss_family: str, options: LossOptions) -> None:
        resolve_mlx_loss(loss_family).settings(options)

    def train(
        self,
        loss_family: str,
        samples: Sequence[TrajectoryItem],
        advantages: Sequence[float],
        options: LossOptions,
    ) -> dict[str, float | int]:
        step = resolve_mlx_loss(loss_family).step(samples, advantages, options)
        return self.run(lambda: step.train(self._loaded.trainer))

    # -- The adapter

    def adapter_snapshot(self) -> MLXAdapterSnapshot:
        return MLXAdapterSnapshot(self.run(self._loaded.trainer.adapter_tensors))

    def apply_adapter(self, snapshot: AdapterSnapshot) -> None:
        tensors = self._tensors_of(snapshot)
        self.run(lambda: self._loaded.trainer.load_adapter_tensors(tensors))

    def adapter_delta(self, before: AdapterSnapshot, after: AdapterSnapshot) -> tuple[float, int]:
        old, new = self._tensors_of(before), self._tensors_of(after)
        if old.keys() != new.keys():
            raise MLXEngineError("the two adapter snapshots hold different tensors")

        def delta() -> tuple[float, int]:
            squared = [
                real_scalar(mx.sum((new[name].astype(mx.float32) - value.astype(mx.float32)) ** 2))
                for name, value in old.items()
            ]
            return sum(squared) ** 0.5, sum(1 for value in squared if value > 0.0)

        return self.run(delta)

    @staticmethod
    def _tensors_of(snapshot: AdapterSnapshot) -> Mapping[str, mx.array]:
        if not isinstance(snapshot, MLXAdapterSnapshot):
            raise MLXEngineError(f"the mlx engine cannot read a {type(snapshot).__name__}")
        return snapshot.tensors

    def save_adapter(self, destination: Path, *, source: Mapping[str, Any]) -> None:
        """Write the live adapter in mlx-lm's format, with Reef's adapter metadata, to a new directory.

        The files are written and synced in a sibling directory, which is then
        renamed to ``destination``. A reader sees either no directory or the
        complete adapter.
        """
        destination = destination.resolve()
        if destination.exists():
            raise MLXEngineError(f"adapter destination {destination} already exists")
        mkdir_durable(destination.parent)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            tensors = self.run(self._loaded.trainer.adapter_tensors)
            weights = staging / MLX_ADAPTER_WEIGHTS
            self.run(lambda: mx.save_safetensors(str(weights), dict(tensors)))
            with weights.open("rb") as handle:
                os.fsync(handle.fileno())
            lora_parameters = self._lora_parameters()
            write_json(
                staging / ADAPTER_CONFIG,
                {"fine_tune_type": "lora", "num_layers": self._config.lora_layers, "lora_parameters": lora_parameters},
            )
            write_json(
                staging / ADAPTER_METADATA,
                {
                    "schema": METADATA_SCHEMA,
                    "base_model": {"name_or_path": self._model_path},
                    "lora": lora_parameters,
                    "libraries": {package: importlib.metadata.version(package) for package in ("mlx", "mlx-lm")},
                    "source": dict(source),
                    "files": {name: file_sha256(staging / name) for name in (ADAPTER_CONFIG, MLX_ADAPTER_WEIGHTS)},
                },
            )
            os.replace(staging, destination)
            fsync_dir(destination.parent)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def load_adapter(self, source: Path) -> None:
        weights = source / MLX_ADAPTER_WEIGHTS
        if not weights.is_file():
            raise MLXEngineError(f"adapter at {source} has no {MLX_ADAPTER_WEIGHTS}")

        def load_tensors() -> None:
            loaded = mx.load(str(weights))
            if not isinstance(loaded, dict) or not loaded:
                raise MLXEngineError(f"adapter at {source} carries no tensors")
            self._loaded.trainer.load_adapter_tensors(loaded)

        self.run(load_tensors)


__all__ = ["MLX_ADAPTER_WEIGHTS", "MLXAdapterSnapshot", "MLXEngine"]
