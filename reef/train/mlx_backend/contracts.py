"""The values and the engine contract that the MLX runtime, serving and inference share; no MLX import.

The runtime halves and the request handler use the engine only through
:class:`ModelEngine`. Tests run them against a fake engine on any machine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.core.batches import TrajectoryItem
from reef.core.chat_protocol import ParsedToolCall
from reef.train.mlx_backend.config import MLXEngineConfig


class GenerationListener(ABC):
    """Receives the text of a streaming generation while it runs.

    The engine calls ``emit`` on its thread with each piece of text that the
    detokenizer settles, so ``emit`` must only hand the piece off. The engine
    polls ``cancelled`` once per token; ``True`` ends the generation.
    """

    @abstractmethod
    def emit(self, piece: str) -> None: ...

    @abstractmethod
    def cancelled(self) -> bool: ...


@dataclass(frozen=True)
class Rollout:
    """One generated completion, with the tokens and log probabilities that training reads.

    ``finish_reason`` is ``stop``, ``length`` or ``cancelled``. The top-K
    fields hold, per response token, the candidate ids and their log
    probabilities; they are empty unless the engine captures them.
    """

    prompt_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    text: str
    finish_reason: str
    topk_indices: tuple[tuple[int, ...], ...] = ()
    topk_log_probs: tuple[tuple[float, ...], ...] = ()


class AdapterSnapshot(ABC):  # noqa: B024 - a typed handle; each engine defines its own contents
    """Adapter weights copied out of an engine. Only the engine that made a snapshot reads it."""


class OptimizerSnapshot(ABC):  # noqa: B024 - a typed handle; each engine defines its own contents
    """The optimizer state of an engine at one point. Only the engine that made a snapshot reads it."""


@dataclass(frozen=True)
class LossOptions:
    """The loss settings of a deployment: the KL coefficient and the family's own options."""

    kl_coef: float = 0.0
    options: Mapping[str, Any] = field(default_factory=dict)


class ModelEngine(ABC):
    """One base model with one LoRA adapter and its optimizer: generation, training and adapter files."""

    @property
    @abstractmethod
    def model_path(self) -> str: ...

    @property
    @abstractmethod
    def config(self) -> MLXEngineConfig: ...

    @abstractmethod
    def render_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> list[int]:
        """Apply the chat template and tokenize; raise ``ValueError`` for messages it cannot render."""

    @abstractmethod
    def opens_reasoning(self, prompt_tokens: Sequence[int]) -> bool:
        """Whether this rendered prompt ends by opening the model's thinking block."""

    @property
    @abstractmethod
    def tool_call_markers(self) -> tuple[str, str] | None:
        """The start and end markers of a tool call, or ``None`` when the template states no call syntax."""

    @abstractmethod
    def parse_tool_call(self, body: str, tools: Sequence[Mapping[str, Any]]) -> ParsedToolCall:
        """Parse the text between the tool-call markers; raise ``ValueError`` for a call that names no function."""

    @abstractmethod
    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        listener: GenerationListener | None = None,
    ) -> Rollout:
        """Sample one completion from the current adapter; ``listener`` receives the text as it is sampled."""

    @abstractmethod
    def adapter_snapshot(self) -> AdapterSnapshot: ...

    @abstractmethod
    def apply_adapter(self, snapshot: AdapterSnapshot) -> None: ...

    @abstractmethod
    def optimizer_snapshot(self) -> OptimizerSnapshot: ...

    @abstractmethod
    def restore_optimizer(self, snapshot: OptimizerSnapshot) -> None:
        """Return the optimizer to a snapshot: its moments and its step count."""

    @abstractmethod
    def adapter_delta(self, before: AdapterSnapshot, after: AdapterSnapshot) -> tuple[float, int]:
        """The L2 distance between two snapshots and the number of tensors that differ."""

    @abstractmethod
    def check_loss(self, loss_family: str, options: LossOptions) -> None:
        """Raise ``ValueError`` if this engine cannot train ``loss_family`` with ``options``."""

    @abstractmethod
    def train(
        self,
        loss_family: str,
        samples: Sequence[TrajectoryItem],
        advantages: Sequence[float],
        options: LossOptions,
    ) -> dict[str, float | int]:
        """Run one optimizer step of ``loss_family`` on the live adapter and return its metrics."""

    @abstractmethod
    def save_adapter(self, destination: Path, *, source: Mapping[str, Any]) -> None:
        """Write the live adapter to the new directory ``destination``; ``source`` records what trained it."""

    @abstractmethod
    def load_adapter(self, source: Path) -> None:
        """Make the adapter saved at ``source`` the live adapter."""

    @abstractmethod
    def close(self) -> None:
        """Stop the engine thread."""


__all__ = ["AdapterSnapshot", "GenerationListener", "LossOptions", "ModelEngine", "OptimizerSnapshot", "Rollout"]
