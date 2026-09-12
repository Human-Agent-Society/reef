"""Backend-neutral component contracts for model deployment ownership.

These contracts describe startup and cleanup only. Training steps, weight
transport and commit-gated activation keep their existing runtime contracts.
Concrete integrations keep framework arguments and allocation handles private.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from reef.runtime.executor import Executor


class DeploymentResources(Protocol):
    """A coordinated allocation and its runtime connection, owned by Reef."""

    def start(self) -> None:
        """Acquire resources; close must also handle a partially failed start."""

    def close(self) -> None:
        """Release owned reservations and connections, idempotently."""


@dataclass(frozen=True)
class InferenceConnection:
    """Borrowed control transport with an explicitly versioned adapter protocol.

    The protocol identifies the RPC vocabulary, including direct weight-update
    attachment. An HTTP endpoint alone does not satisfy this connection.
    """

    protocol: str
    control: Executor


class InferenceService(Protocol):
    """An inference component that owns its engines but borrows reservations."""

    @property
    def connection_protocol(self) -> str:
        """Control protocol provided by the selected engine integration."""

    def start(self, resources: DeploymentResources) -> InferenceConnection:
        """Start engines in supplied resources and return a borrowed connection."""

    def check_health(self) -> None:
        """Raise when the component is not ready."""

    def close(self) -> None:
        """Release owned engines, including partial starts, idempotently."""


class TrainingService(Protocol):
    """Training workers that attach to resources and an inference connection."""

    @property
    def inference_protocol(self) -> str | None:
        """Required protocol, or None for an explicit combined compatibility path."""

    def start(self, resources: DeploymentResources, inference: InferenceConnection | None) -> None:
        """Start training workers without acquiring ownership of supplied objects."""

    def check_health(self) -> None:
        """Raise when the component is not ready."""

    def close(self) -> None:
        """Release owned training objects, including partial starts, idempotently."""


@dataclass(frozen=True)
class ModelDeploymentPlan:
    """Configured components; constructing a plan must not allocate resources.

    A missing inference component explicitly selects a combined compatibility
    lifecycle. It is never a fallback after a separate component fails to start.
    """

    resources: DeploymentResources
    inference: InferenceService | None
    training: TrainingService

    def validate(self) -> None:
        required = self.training.inference_protocol
        provided = self.inference.connection_protocol if self.inference is not None else None
        if required != provided or (self.inference is not None and not provided):
            raise ValueError(
                f"incompatible inference control protocol: training requires {required!r}, got {provided!r}"
            )
