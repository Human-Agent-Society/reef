"""The start of a prepared task image (``reef.execution.sandbox.prepared_images``) on AgentENV.

``start_prepared_image`` starts an environment with a prepared image from the image's template when the template is
ready. If the template is unavailable before allocation, Reef starts the verified image with
AgentENV's ``POST /sandboxes-cold`` and runs its start command (``run_image_start``).
The E2B trial class implements ``PreparedImageHost`` with the E2B SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass

import aiohttp

from reef.core.errors import InternalError
from reef.execution.sandbox.agentenv.build import BUILD_OUTPUT_TAIL_CHARS, AgentEnvTemplates
from reef.execution.sandbox.prepared_images import IMAGE_REFERENCE, PreparedImage
from reef.execution.sandbox.process import SandboxCapacityError, SandboxRequestError

logger = logging.getLogger(__name__)

#: The pause between two runs of a prepared image's ready command after a cold start, in seconds, as in AgentENV's
#: template runner.
READY_POLL_SECONDS = 2.0
#: The longest wait of a cold start for a prepared image's start to be ready, in seconds, as in AgentENV's template
#: runner. The environment's start time limit can end it sooner.
READY_LIMIT_SECONDS = 600.0
#: When the start of a prepared image without a ready command is ready, in seconds after it began: AgentENV's template
#: runner then uses ``/agentenv/bin/busybox sleep 20`` as the ready command.
DEFAULT_READY_SECONDS = 20.0
#: The longest wait for AgentENV's state of a prepared image's template, in seconds.
TEMPLATE_STATUS_WAIT_SECONDS = 5.0
#: The statuses with which AgentENV refuses a cold start for want of a free worker now (429, 503): it made no
#: sandbox, and a later start can pass. Any other failure can leave a sandbox that still starts, such as a gateway's
#: 502 or 504, a lost connection, or a payload error while reading a 201 body, or has a lasting cause, so the start
#: is sent once.
CAPACITY_STATUSES = (429, 503)
#: Bounded in-place retries and jittered exponential backoff limits when AgentENV refuses a cold start with HTTP 503
#: (no free worker now, proving no sandbox was created).
COLD_START_MAX_ATTEMPTS = 3
COLD_START_FIRST_WAIT_SECONDS = 2.0
COLD_START_MAX_WAIT_SECONDS = 30.0
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
        """Send the request to ``templates``' deployment within ``wait_seconds``, a positive number, and return the ID
        of the new sandbox (HTTP 201).

        Raise ``SandboxCapacityError`` for a status of ``CAPACITY_STATUSES`` (429 immediately; 503 retries in place
        up to ``COLD_START_MAX_ATTEMPTS`` times with jittered exponential backoff within ``wait_seconds`` because
        AgentENV created no sandbox; no request is sent after ``wait_seconds``, also when a wait ends late), and ``SandboxRequestError`` immediately for HTTP 400. Raise ``ColdSandboxError``
        immediately after a single send for any other answer, such as 401, 403, 500, or a gateway's 502 or 504, and
        when no answer came in time, the connection was lost, or reading the response body failed. After those,
        AgentENV can still start the sandbox, whose ID the caller does not know; AgentENV deletes it at the end of its
        lifetime. A redirect is such an answer too: the request is not sent again, to the same host or another one,
        with the API key."""
        deadline = time.monotonic() + wait_seconds
        backoff_seconds = COLD_START_FIRST_WAIT_SECONDS
        attempt = 0
        exhausted_message = ""
        while True:
            remaining = deadline - time.monotonic()
            if attempt > 0 and remaining <= 0:
                # The wait ended later than asked. No time is left, so the last 503 stands and nothing is sent.
                raise SandboxCapacityError(exhausted_message)
            attempt += 1
            try:
                async with (
                    aiohttp.ClientSession(
                        headers={"X-API-Key": templates.api_key}, timeout=aiohttp.ClientTimeout(total=remaining)
                    ) as session,
                    session.post(
                        f"{templates.api_url}/sandboxes-cold", json=self.body(), allow_redirects=False
                    ) as response,
                ):
                    if response.status == 201:
                        answer = await response.json()
                        sandbox_id = answer.get("sandboxID") if isinstance(answer, dict) else None
                        if not isinstance(sandbox_id, str) or not sandbox_id:
                            raise ColdSandboxError(
                                f"AgentENV started a sandbox of {self.image} and named no sandbox ID"
                            )
                        return sandbox_id
                    status = response.status
                    text = (await response.text())[-BUILD_OUTPUT_TAIL_CHARS:].strip()
            except TimeoutError as error:
                raise ColdSandboxError(f"AgentENV did not answer the cold start of {self.image} in time") from error
            except (aiohttp.ClientError, json.JSONDecodeError) as error:
                raise ColdSandboxError(f"the cold start of {self.image} lost its answer: {error!r}") from error
            message = f"AgentENV refused a sandbox of {self.image} (HTTP {status}): {text}"
            if status not in CAPACITY_STATUSES:
                if status == 400:
                    raise SandboxRequestError(message)
                raise ColdSandboxError(message)
            if status == 429:
                raise SandboxCapacityError(message)
            # HTTP 503 proves AgentENV allocated no sandbox; retry in place with bounded jittered backoff.
            exhausted_message = (
                f"AgentENV cold start of {self.image} failed after {attempt} attempts (HTTP 503): {text}"
            )
            if attempt >= COLD_START_MAX_ATTEMPTS:
                raise SandboxCapacityError(exhausted_message)
            delay_seconds = min(backoff_seconds, COLD_START_MAX_WAIT_SECONDS) * random.uniform(0.5, 1.0)
            if time.monotonic() + delay_seconds >= deadline:
                raise SandboxCapacityError(exhausted_message)
            logger.warning(
                "AgentENV cold start of %s received HTTP 503 on attempt %d/%d (%s); retrying in %.2f s",
                self.image,
                attempt,
                COLD_START_MAX_ATTEMPTS,
                text,
                delay_seconds,
            )
            await asyncio.sleep(delay_seconds)
            backoff_seconds = min(backoff_seconds * 2.0, COLD_START_MAX_WAIT_SECONDS)


class PreparedImageHost(ABC):
    """A trial environment whose sandbox starts from a prepared image on AgentENV (``start_prepared_image``).

    Values: the environment's own state. It holds at most one sandbox, and each operation below that runs a command
    acts on that sandbox.
    Purpose: ``KillOnTimeoutE2BEnvironment`` (``reef.execution.sandbox.e2b.trial``) implements these operations with
    the E2B SDK, and ``start_prepared_image`` calls them.
    """

    @abstractmethod
    async def start_from_template(self, template_id: str) -> bool:
        """Create the sandbox from the template ``template_id``, hold it, and return True. Return False, and hold no
        sandbox, when the worker has no snapshot or the gateway explicitly refused placement for this template.
        Raise any other failure as the create call raises it."""

    @abstractmethod
    def cold_sandbox(self, image: str) -> ColdSandbox:
        """The cold start of ``image``, with the values of a template's create call: the environment's lifetime,
        network options, metadata and startup variables, and the resources that a template build would set."""

    @abstractmethod
    async def attach_sandbox(self, sandbox_id: str) -> None:
        """Connect to the sandbox ``sandbox_id``, which a cold start made, and hold it."""

    @abstractmethod
    async def discard_sandbox(self, sandbox_id: str) -> None:
        """Delete the sandbox ``sandbox_id`` of a start that did not finish, and hold no sandbox. A delete that fails
        is logged, and AgentENV deletes the sandbox at the end of its lifetime."""

    @abstractmethod
    def start_command(self, command: str) -> AbstractAsyncContextManager[asyncio.Future[None]]:
        """Start ``command`` in the background in the held sandbox, with no time limit, as the image's own user, in
        its working directory and with its variables. Entering the context returns once the command runs, with a future
        that ends with the command: with None for an exit with 0, else with the command's error. Leaving the context,
        also by an error or a cancellation, stops following the command, which keeps running."""

    @abstractmethod
    async def command_passes(self, command: str, timeout_seconds: int) -> bool:
        """Whether ``command`` exits with 0 in the held sandbox, run as ``start_command`` runs one but in the
        foreground, for at most ``timeout_seconds``. Raise when the command cannot run or runs past that time."""


async def start_prepared_image(
    host: PreparedImageHost, templates: AgentEnvTemplates, prepared: PreparedImage, deadline: float
) -> None:
    """Start the sandbox of ``host`` from ``prepared`` on ``templates``' deployment, without a template build, by
    ``deadline``: the ``time.monotonic()`` time of the end of the environment's start time limit.

    When AgentENV reads the template ``prepared.template_id`` as ready (``template_is_ready``), the sandbox starts from
    it, as from any template (``PreparedImageHost.start_from_template``): its snapshot already ran the image's start.
    The image starts cold if the template is absent or unready, the worker lacks its snapshot, or the gateway
    refuses this template before placement. An ambiguous create response propagates to the caller.

    The cold start (``ColdSandbox``) waits for its answer until ``deadline``, retrying only HTTP 503 in place and
    sending every other answer once. Then the image's start runs (``run_image_start``). When the start fails after
    AgentENV made the sandbox, also when it is cancelled, the host deletes the sandbox before this raises. Raise what
    ``ColdSandbox.create`` raises, and ``ColdSandboxError`` for a later failure.
    """
    if prepared.template_id is not None and await template_is_ready(templates, prepared.template_id, deadline):
        if await host.start_from_template(prepared.template_id):
            return
        logger.info("AgentENV could not place template %s. Starting %s cold.", prepared.template_id, prepared.image)
    sandbox_id = await host.cold_sandbox(prepared.image).create(templates, max(deadline - time.monotonic(), 1.0))
    try:
        await host.attach_sandbox(sandbox_id)
        await run_image_start(host, prepared, deadline)
    except BaseException as error:
        await host.discard_sandbox(sandbox_id)
        if not isinstance(error, Exception) or isinstance(error, ColdSandboxError):
            raise
        raise ColdSandboxError(f"the sandbox {sandbox_id} of {prepared.image} did not start: {error}") from error


async def template_is_ready(templates: AgentEnvTemplates, template_id: str, deadline: float) -> bool:
    """Whether AgentENV reads the template ``template_id`` as ready (``AgentEnvTemplates.is_ready``). The read waits
    at most ``TEMPLATE_STATUS_WAIT_SECONDS``, and at most until ``deadline``, a ``time.monotonic()`` time, though at
    least 1 s. A read that gives no answer in that time, a lost connection, and an answer of 429 or 5xx count as not
    ready: the read allocates nothing, so the verified image can start at once. Credential and request errors reach
    the caller."""
    try:
        return await templates.is_ready(
            template_id, min(TEMPLATE_STATUS_WAIT_SECONDS, max(deadline - time.monotonic(), 1.0))
        )
    except aiohttp.ClientResponseError as error:
        if error.status != 429 and not 500 <= error.status < 600:
            raise
    except (TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError):
        pass
    logger.warning("AgentENV could not report the state of template %s; starting from its image", template_id)
    return False


async def run_image_start(host: PreparedImageHost, prepared: PreparedImage, deadline: float) -> None:
    """Run the start of ``prepared`` in the sandbox that ``host`` holds, as AgentENV's template runner ran it before
    the template's snapshot.

    Both commands run as the image's own user, in its working directory and with its variables
    (``PreparedImageHost.start_command``). ``prepared.start`` starts in the background and keeps running. The start is
    ready when ``prepared.ready`` exits with 0; it runs again ``READY_POLL_SECONDS`` after each other exit. Without a
    ready command, the start is ready ``DEFAULT_READY_SECONDS`` after it began, and an image without either command is
    ready at once. When this returns or raises, also when it is cancelled, it has disconnected from the start command.

    Raise ``ColdSandboxError`` when the start command ends with an error before the start is ready, and when the start
    is not ready ``READY_LIMIT_SECONDS`` after it began, or at ``deadline``, a ``time.monotonic()`` time, when that
    comes first. A run of the ready command ends at that limit too."""
    if prepared.start is None and prepared.ready is None:
        return
    began = time.monotonic()
    limit_seconds = min(READY_LIMIT_SECONDS, max(deadline - began, 0.0))
    ready_by = began + limit_seconds
    # The start command runs before the first run of the ready command.
    starting: AbstractAsyncContextManager[asyncio.Future[None] | None] = (
        nullcontext(None) if prepared.start is None else host.start_command(prepared.start)
    )

    def check_start(ended: asyncio.Future[None] | None) -> None:
        # An end with 0 is a start that leaves no process of its own.
        failure = ended.exception() if ended is not None and ended.done() else None
        if failure is not None:
            raise ColdSandboxError(
                f"the start command of {prepared.image} ended before the start was ready: {failure}"
            ) from failure

    async with starting as ended:
        while True:
            check_start(ended)
            seconds_left = ready_by - time.monotonic()
            if seconds_left <= 0:
                raise ColdSandboxError(f"the start of {prepared.image} was not ready in {limit_seconds:.0f} s")
            if prepared.ready is None:
                if time.monotonic() - began >= DEFAULT_READY_SECONDS:
                    return
            elif await host.command_passes(prepared.ready, math.ceil(seconds_left)):
                # The ready command and the start command can end in the same turn of the event loop.
                if ended is not None:
                    await asyncio.sleep(0)
                    check_start(ended)
                return
            await asyncio.sleep(min(READY_POLL_SECONDS, seconds_left))


__all__ = [
    "CAPACITY_STATUSES",
    "COLD_START_FIRST_WAIT_SECONDS",
    "COLD_START_MAX_ATTEMPTS",
    "COLD_START_MAX_WAIT_SECONDS",
    "DEFAULT_READY_SECONDS",
    "NETWORK_FIELDS",
    "READY_LIMIT_SECONDS",
    "READY_POLL_SECONDS",
    "TEMPLATE_STATUS_WAIT_SECONDS",
    "ColdSandbox",
    "ColdSandboxError",
    "PreparedImageHost",
    "is_missing_snapshot",
    "run_image_start",
    "start_prepared_image",
    "template_is_ready",
]
