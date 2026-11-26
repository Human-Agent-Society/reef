"""The start of a prepared task image (``reef.execution.sandbox.prepared_images``) on AgentENV.

Reef's E2B trial class starts an environment with a prepared image from the image's template when the template is
ready. Otherwise it starts the image itself with AgentENV's ``POST /sandboxes-cold`` (``ColdSandbox``), also after
AgentENV answered that its worker has no snapshot of the template (``is_missing_snapshot``). So a template that is
missing or not ready never sends a task with a verified image back through a Dockerfile build.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

import aiohttp

from reef.core.errors import InternalError
from reef.execution.sandbox.agentenv.build import BUILD_OUTPUT_TAIL_CHARS, AgentEnvTemplates
from reef.execution.sandbox.prepared_images import IMAGE_REFERENCE
from reef.execution.sandbox.process import SandboxCapacityError, SandboxRequestError

#: The statuses with which AgentENV refuses a cold start for want of a free worker now (429, 503): it made no
#: sandbox, and a later start can pass. Any other failure can leave a sandbox that still starts, such as a gateway's
#: 502 or 504, or has a lasting cause, so the start is not sent again.
CAPACITY_STATUSES = (429, 503)
#: The names of E2B's network options (``SandboxNetworkOpts`` of the E2B SDK) in AgentENV's sandbox request.
NETWORK_FIELDS = {
    "allow_out": "allowOut",
    "deny_out": "denyOut",
    "allow_public_traffic": "allowPublicTraffic",
    "mask_request_host": "maskRequestHost",
}


def is_missing_snapshot(status: int | None, message: str, template_id: str) -> bool:
    """Whether an answer of AgentENV's sandbox create, with the HTTP status ``status`` and the message ``message``,
    refused the template ``template_id`` because the worker has no snapshot of it. AgentENV gives these answers before
    it allocates a sandbox: HTTP 400 ``template <ID> not found``, and HTTP 500 ``snapshot '<ID>' is not ready``. The
    message can have text before the answer, but it must name ``template_id`` whole: the answer of another template,
    or any other answer with these statuses, is not one."""
    identifier = re.escape(template_id)
    if status == 400:
        pattern = rf"(?<![^\s:])template {identifier} not found\b"
    elif status == 500:
        pattern = rf"(?<![^\s:])snapshot '{identifier}' is not ready\b"
    else:
        return False
    return re.search(pattern, message) is not None


class ColdSandboxError(InternalError):
    """AgentENV did not start a sandbox from a prepared image, or the sandbox did not pass its start. A start does not
    send its request again after this error, so it never makes a second sandbox for an answer that it could not
    read."""


@dataclass(frozen=True)
class ColdSandbox:
    """One request to AgentENV's ``POST /sandboxes-cold``: a sandbox started from an image, without a template.

    Values: ``image`` is the image by digest. ``timeout_seconds`` is the sandbox's lifetime, at least 1; AgentENV
    deletes the sandbox at its end and does not pause it. ``allow_internet_access``, ``network``, ``metadata`` and
    ``env_vars`` are the values of E2B's create call: ``network`` has the names of the E2B SDK's network options
    (``NETWORK_FIELDS``), or is None for no options. ``cpu_count`` and ``memory_mb`` are the sandbox's resources,
    None for AgentENV's default.
    Purpose: ``KillOnTimeoutE2BEnvironment`` makes one for an environment whose prepared image has no ready template,
    and ``create`` sends it.
    """

    image: str
    timeout_seconds: int
    allow_internet_access: bool
    network: Mapping[str, object] | None
    metadata: Mapping[str, str]
    env_vars: Mapping[str, str]
    cpu_count: int | None
    memory_mb: int | None

    def __post_init__(self) -> None:
        if IMAGE_REFERENCE.fullmatch(self.image) is None:
            raise ValueError(f"a cold sandbox starts an image by digest, not {self.image!r}")
        if self.timeout_seconds < 1:
            raise ValueError("a cold sandbox needs a lifetime of at least 1 s")
        unknown = set(self.network or {}) - set(NETWORK_FIELDS)
        if unknown:
            # A network option that the request cannot carry would leave the sandbox less isolated than asked.
            raise ValueError(f"a cold sandbox cannot apply the network options {', '.join(sorted(unknown))}")

    def body(self) -> dict[str, object]:
        """The JSON body of the request."""
        body: dict[str, object] = {
            "image": self.image,
            "timeout": self.timeout_seconds,
            "autoPause": False,
            "secure": True,
            "allowInternetAccess": self.allow_internet_access,
            "metadata": dict(self.metadata),
            "envVars": dict(self.env_vars),
        }
        if self.network is not None:
            body["network"] = {NETWORK_FIELDS[name]: value for name, value in self.network.items()}
        if self.cpu_count is not None:
            body["cpuCount"] = self.cpu_count
        if self.memory_mb is not None:
            body["memoryMB"] = self.memory_mb
        return body

    async def create(self, templates: AgentEnvTemplates, wait_seconds: float) -> str:
        """Send the request once to ``templates``' deployment, wait at most ``wait_seconds``, a positive number, for
        the answer, and return the ID of the new sandbox (HTTP 201).

        Raise ``SandboxCapacityError`` for a status of ``CAPACITY_STATUSES``, and ``SandboxRequestError`` for HTTP
        400. Raise ``ColdSandboxError`` for any other answer, such as 401, 403, 500, or a gateway's 502 or 504, and
        when no answer came in time or the connection was lost. After those, AgentENV can still start the sandbox,
        whose ID the caller does not know; AgentENV deletes it at the end of its lifetime. A redirect is such an
        answer too: the request is not sent again, to the same host or another one, with the API key."""
        try:
            async with (
                aiohttp.ClientSession(
                    headers={"X-API-Key": templates.api_key}, timeout=aiohttp.ClientTimeout(total=wait_seconds)
                ) as session,
                session.post(
                    f"{templates.api_url}/sandboxes-cold", json=self.body(), allow_redirects=False
                ) as response,
            ):
                if response.status == 201:
                    answer = await response.json()
                    sandbox_id = answer.get("sandboxID") if isinstance(answer, dict) else None
                    if not isinstance(sandbox_id, str) or not sandbox_id:
                        raise ColdSandboxError(f"AgentENV started a sandbox of {self.image} and named no sandbox ID")
                    return sandbox_id
                status = response.status
                text = (await response.text())[-BUILD_OUTPUT_TAIL_CHARS:]
        except TimeoutError as error:
            raise ColdSandboxError(f"AgentENV did not answer the cold start of {self.image} in time") from error
        except (aiohttp.ClientError, json.JSONDecodeError) as error:
            raise ColdSandboxError(f"the cold start of {self.image} lost its answer: {error!r}") from error
        message = f"AgentENV refused a sandbox of {self.image} (HTTP {status}): {text}"
        if status in CAPACITY_STATUSES:
            raise SandboxCapacityError(message)
        if status == 400:
            raise SandboxRequestError(message)
        raise ColdSandboxError(message)


__all__ = [
    "CAPACITY_STATUSES",
    "NETWORK_FIELDS",
    "ColdSandbox",
    "ColdSandboxError",
    "is_missing_snapshot",
]
