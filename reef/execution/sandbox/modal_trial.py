"""Modal sandbox creation errors for Reef trials. Harbor loads this adapter for Modal environments."""

from __future__ import annotations

from typing import Any, override

from harbor.environments.modal import ModalEnvironment
from modal import Sandbox
from modal.exception import InvalidError, ResourceExhaustedError

from reef.execution.sandbox.process import SandboxCapacityError, SandboxRequestError


class ModalTrialEnvironment(ModalEnvironment):
    """A Harbor Modal environment that distinguishes temporary capacity shortages from rejected sandbox requests.

    Trials record the exception type. Only errors during creation carry the special meaning for the rollout queue.
    """

    @override
    async def _create_sandbox(
        self,
        *,
        entrypoint: list[str] | None = None,
        block_network: bool | None = None,
        experimental_options: dict[str, Any] | None = None,
    ) -> Sandbox:
        try:
            return await super()._create_sandbox(
                entrypoint=entrypoint, block_network=block_network, experimental_options=experimental_options
            )
        except ResourceExhaustedError as exc:
            raise SandboxCapacityError(f"Modal has no resources for the sandbox now: {exc}") from exc
        except InvalidError as exc:
            raise SandboxRequestError(f"Modal refused the sandbox's request: {exc}") from exc


__all__ = ["ModalTrialEnvironment"]
