"""Content-addressed declarative Reef composition candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.harness import AdapterDescriptor, render_composition
from reef.harness.nodes import NODE_KINDS

MAX_NODES = 32
ALLOWED_NODE_KINDS = frozenset({"agent_command", "rules", "skill"})


@dataclass(frozen=True)
class CompositionCandidate:
    """Immutable candidate identity derived only from normalized tree content."""

    canonical_json: str
    parent_hashes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any],
        *,
        parent_hashes: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> CompositionCandidate:
        normalized = normalize_composition(value)
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        parents = tuple(dict.fromkeys(str(parent) for parent in parent_hashes if parent))
        if any(len(parent) != 64 for parent in parents):
            raise ValueError("candidate parent hashes must be full SHA-256 values")
        return cls(canonical, parents, dict(metadata or {}))

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        parent_hashes: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> CompositionCandidate:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("composition file must contain a JSON object")
        return cls.from_value(raw, parent_hashes=parent_hashes, metadata=metadata)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def composition(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)

    @property
    def nodes(self) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        return tuple((node["kind"], node["config"]) for node in self.composition["nodes"])

    def render(self, descriptor: AdapterDescriptor) -> dict[str, str]:
        return render_composition(self.nodes, descriptor)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "parent_hashes": list(self.parent_hashes),
            "metadata": dict(self.metadata),
            "composition": self.composition,
        }


def normalize_composition(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the public Reef node vocabulary and return stable JSON data."""
    if set(value) - {"schema_version", "nodes"}:
        unknown = sorted(set(value) - {"schema_version", "nodes"})
        raise ValueError(f"composition contains unknown top-level fields: {unknown!r}")
    schema_version = value.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError("composition schema_version must be 1")
    raw_nodes = value.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("composition requires a non-empty nodes list")
    if len(raw_nodes) > MAX_NODES:
        raise ValueError(f"composition may contain at most {MAX_NODES} nodes")

    normalized_nodes: list[dict[str, Any]] = []
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, Mapping) or set(raw_node) != {"kind", "config"}:
            raise ValueError(f"composition node {index} must contain exactly kind and config")
        kind = raw_node.get("kind")
        config = raw_node.get("config")
        if kind not in NODE_KINDS:
            raise ValueError(f"composition node {index} has unknown kind {kind!r}")
        if kind not in ALLOWED_NODE_KINDS:
            raise ValueError(
                f"composition node {index} uses unsupported kind {kind!r}; "
                "the sealed experiment permits only declarative nodes"
            )
        NODE_KINDS[str(kind)](None, config)
        try:
            stable_config = json.loads(json.dumps(config, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"composition node {index} config must be JSON-serializable") from exc
        normalized_nodes.append({"kind": str(kind), "config": stable_config})
    return {"schema_version": 1, "nodes": normalized_nodes}


def genesis_composition() -> CompositionCandidate:
    return CompositionCandidate.from_value(
        {
            "schema_version": 1,
            "nodes": [
                {
                    "kind": "rules",
                    "config": {
                        "text": (
                            "Work carefully in the terminal until the task is complete. "
                            "Inspect the environment, verify changes, and do not stop at a plan."
                        )
                    },
                },
                {
                    "kind": "skill",
                    "config": {
                        "name": "terminal-task",
                        "text": (
                            "# Terminal task\n\nUse the bash tool to inspect and modify the task environment. "
                            "Run the available tests or verifier-facing checks before finishing."
                        ),
                    },
                },
            ],
        },
        metadata={"role": "genesis"},
    )
