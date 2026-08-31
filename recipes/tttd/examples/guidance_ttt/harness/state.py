from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


def _jsonable(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return _jsonable(OmegaConf.to_container(value, resolve=True))
    except Exception:
        pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted((_jsonable(v) for v in value), key=repr)
    return value


@dataclass
class LibraryEntry:
    id: str
    parent_id: str | None
    problem_id: str
    timestep: int
    guidance: str
    execution_thinking: str
    solution: str
    verifier_reward: float
    verifier_raw_score: float | None
    verifier_status: str
    verifier_message: str
    summary: str
    reusable_idea: str
    failure_mode: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.verifier_reward = float(self.verifier_reward)
        self.verifier_raw_score = None if self.verifier_raw_score is None else float(self.verifier_raw_score)
        self.metadata = _jsonable(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LibraryEntry:
        return cls(**data)


@dataclass
class LibraryNode:
    id: str
    problem_id: str
    timestep: int
    entry_id: str | None
    value: float
    raw_score: float | None
    visits: int
    parent_id: str | None
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.value = float(self.value)
        self.raw_score = None if self.raw_score is None else float(self.raw_score)
        self.visits = int(self.visits)
        self.children = list(self.children)
        self.metadata = _jsonable(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LibraryNode:
        return cls(**data)


@dataclass
class LLMRequest:
    system: str
    user: str
    model: str
    temperature: float
    max_tokens: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: str
    reasoning: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    reward: float
    raw_score: float | None
    valid: bool
    status: str
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def execution_error(cls, message: str) -> VerificationResult:
        return cls(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="execution_error",
            message=message,
            artifacts={},
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def make_root_node(*, problem_id: str, raw_score: float | None, reward: float) -> LibraryNode:
    return LibraryNode(
        id=str(uuid4()),
        problem_id=problem_id,
        timestep=0,
        entry_id=None,
        value=float(reward),
        raw_score=raw_score,
        visits=0,
        parent_id=None,
        children=[],
        metadata={"kind": "root"},
    )
