"""A fake MLX engine for the MLX runtime, serving and inference contracts; needs no MLX."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from reef.core.chat_protocol import ParsedToolCall, template_messages
from reef.train.mlx_backend.config import MLXEngineConfig
from reef.train.mlx_backend.contracts import (
    AdapterSnapshot,
    GenerationListener,
    LossOptions,
    ModelEngine,
    OptimizerSnapshot,
    Rollout,
)

#: The rendered prompt of every request.
PROMPT_TOKENS = (11, 12, 13)

#: The rollout of every generation unless a test sets another one.
DEFAULT_ROLLOUT = Rollout(
    prompt_tokens=PROMPT_TOKENS,
    output_tokens=(21, 22),
    rollout_log_probs=(-0.5, -0.25),
    text="an answer",
    finish_reason="stop",
    topk_indices=((21, 99), (22, 98)),
    topk_log_probs=((-0.5, -3.0), (-0.25, -4.0)),
)


@dataclass(frozen=True)
class FakeAdapterSnapshot(AdapterSnapshot):
    weights: Mapping[str, float]


@dataclass(frozen=True)
class FakeOptimizerSnapshot(OptimizerSnapshot):
    steps: int


@dataclass(frozen=True)
class TrainCall:
    loss_family: str
    samples: tuple[Any, ...]
    advantages: tuple[float, ...]
    options: LossOptions


@dataclass
class FakeMLXEngine(ModelEngine):
    """Plain Python state in place of a model.

    ``train`` adds 1.0 to every weight when ``moves_weights`` is set.
    ``generate`` returns ``rollout``; with a listener it emits ``pieces``
    (the text split in two by default) and stops when the listener cancels.
    ``render_prompt`` normalizes messages as the engine does and records
    them. ``check_loss`` accepts the families in ``loss_families``. Tool calls use
    Qwen's XML syntax between ``<tool_call>`` markers when ``tool_calling``
    is set.
    """

    rollout: Rollout = DEFAULT_ROLLOUT
    pieces: Sequence[str] | None = None
    prompt_tail: str = "<|im_start|>assistant\n"
    tool_calling: bool = True
    moves_weights: bool = True
    loss_families: frozenset[str] = frozenset({"tttd", "openclawrl"})
    engine_config: MLXEngineConfig = field(default_factory=MLXEngineConfig)
    weights: dict[str, float] = field(default_factory=lambda: {"layer.lora_a": 0.0, "layer.lora_b": 0.0})
    #: The optimizer state: the number of steps that ``train`` has taken.
    optimizer_steps: int = 0
    rendered: list[dict[str, Any]] = field(default_factory=list)
    generated: list[dict[str, Any]] = field(default_factory=list)
    streamed: list[str] = field(default_factory=list)
    trained: list[TrainCall] = field(default_factory=list)
    saved: list[tuple[Path, Mapping[str, Any]]] = field(default_factory=list)
    loaded: list[Path] = field(default_factory=list)
    closed: bool = False

    @property
    def model_path(self) -> str:
        return "fake/model"

    @property
    def config(self) -> MLXEngineConfig:
        return self.engine_config

    def render_prompt(self, messages, *, tools=None, template_kwargs=None) -> list[int]:
        self.rendered.append(
            {"messages": template_messages(list(messages)), "tools": tools, "template_kwargs": template_kwargs}
        )
        return list(PROMPT_TOKENS)

    def opens_reasoning(self, prompt_tokens: Sequence[int]) -> bool:
        return self.prompt_tail.rstrip().endswith("<think>")

    @property
    def tool_call_markers(self) -> tuple[str, str] | None:
        return ("<tool_call>", "</tool_call>") if self.tool_calling else None

    def parse_tool_call(self, body: str, tools: Sequence[Mapping[str, Any]]) -> ParsedToolCall:
        _, _, call = body.partition("<function=")
        name, _, parameters = call.partition(">")
        if not name.strip():
            raise ValueError("a tool call names no function")
        arguments = {}
        for chunk in parameters.split("<parameter=")[1:]:
            key, _, value = chunk.partition(">")
            arguments[key] = value.split("</parameter>")[0].strip()
        return ParsedToolCall(name.strip(), json.dumps(arguments))

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        listener: GenerationListener | None = None,
    ) -> Rollout:
        self.generated.append(
            {
                "prompt_tokens": tuple(prompt_tokens),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "weights": dict(self.weights),
            }
        )
        if listener is None:
            return self.rollout
        text = self.rollout.text
        pieces = [text[: len(text) // 2], text[len(text) // 2 :]] if self.pieces is None else list(self.pieces)
        for piece in pieces:
            if listener.cancelled():
                return replace(self.rollout, finish_reason="cancelled")
            self.streamed.append(piece)
            listener.emit(piece)
        return self.rollout

    def adapter_snapshot(self) -> FakeAdapterSnapshot:
        return FakeAdapterSnapshot(dict(self.weights))

    def apply_adapter(self, snapshot: AdapterSnapshot) -> None:
        if not isinstance(snapshot, FakeAdapterSnapshot):
            raise TypeError("the fake engine reads only its own snapshots")
        self.weights = dict(snapshot.weights)

    def optimizer_snapshot(self) -> FakeOptimizerSnapshot:
        return FakeOptimizerSnapshot(self.optimizer_steps)

    def restore_optimizer(self, snapshot: OptimizerSnapshot) -> None:
        if not isinstance(snapshot, FakeOptimizerSnapshot):
            raise TypeError("the fake engine reads only its own snapshots")
        self.optimizer_steps = snapshot.steps

    def adapter_delta(self, before: AdapterSnapshot, after: AdapterSnapshot) -> tuple[float, int]:
        if not isinstance(before, FakeAdapterSnapshot) or not isinstance(after, FakeAdapterSnapshot):
            raise TypeError("the fake engine reads only its own snapshots")
        changed = sum(1 for name, value in after.weights.items() if before.weights[name] != value)
        return float(changed), changed

    def check_loss(self, loss_family: str, options: LossOptions) -> None:
        if loss_family not in self.loss_families:
            raise ValueError(f"unsupported MLX loss family {loss_family!r}")

    def train(self, loss_family, samples, advantages, options: LossOptions) -> dict[str, float | int]:
        self.trained.append(TrainCall(loss_family, tuple(samples), tuple(advantages), options))
        self.optimizer_steps += 1
        if self.moves_weights:
            self.weights = {name: value + 1.0 for name, value in self.weights.items()}
        return {"loss": -1.0, "rows": len(samples)}

    def save_adapter(self, destination: Path, *, source: Mapping[str, Any]) -> None:
        if destination.exists():
            raise ValueError(f"adapter destination {destination} already exists")
        destination.mkdir(parents=True)
        (destination / "adapters.safetensors").write_text(json.dumps(self.weights))
        self.saved.append((destination, dict(source)))

    def load_adapter(self, source: Path) -> None:
        self.weights = json.loads((source / "adapters.safetensors").read_text())
        self.loaded.append(source)

    def close(self) -> None:
        self.closed = True


__all__ = ["DEFAULT_ROLLOUT", "PROMPT_TOKENS", "FakeAdapterSnapshot", "FakeMLXEngine", "TrainCall"]
