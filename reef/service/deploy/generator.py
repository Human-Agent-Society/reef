"""The generator service as a process ``reef serve`` starts: one more child beside inference, training and HTTP.

A deployment with a ``generator`` section gets a ``generator`` service before the HTTP service, which
depends on it, so ``${endpoints.generator}`` is published by the time the HTTP child and its recipe read
their configuration. The generator reads the same resolved deployment through ``REEF_CONFIG``; it needs
no arguments. Its executor is the ``generator`` role of ``execution:``, so a deployment can place it on a
host with Docker and the ``harbor`` command line while the rest stays local. It runs under the
same interpreter as the other children (``REEF_PYTHON``, otherwise the launcher's).
"""

from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.config import config_arguments, config_metadata, config_option, parse_config_values
from reef.service.deploy.config_utils import DeployConfigError
from reef.service.deploy.inference import http_readiness_command

GENERATOR_SERVICE = "generator"
DEFAULT_PORT = 8910
DEFAULT_CONCURRENCY = 2
DEFAULT_DESIGNER_TIMEOUT_S = 1800.0
DEFAULT_DESIGNER_POLL_S = 5.0
DEFAULT_DESIGNER_WAIT_S = 1800.0
DEFAULT_READY_TIMEOUT = 60
DESIGNER_PROMPT_SOURCES = ("fixed", "harness")


@dataclass(frozen=True)
class GeneratorSettings:
    """The generator service's own settings; the Reef address, token and served model come from the ``reef`` section."""

    tasks_root: str = config_option(help="The directory generated tasks, manifests and Harbor job files live under.")
    host: str = config_option("127.0.0.1", help="The generator's bind address.")
    port: int = config_option(DEFAULT_PORT, help="The generator's bind port.")
    work_dir: str | None = config_option(
        None, help="Where the task player keeps trials; <tasks-root>/.play by default."
    )
    agent: Mapping[str, Any] | None = dataclasses.field(
        default=None,
        metadata=config_metadata(
            "The Harbor agent the task player runs, with {model}, {base_url} and {api_key} placeholders; "
            "terminus-2 by default."
        ),
    )
    agent_host: str | None = config_option(
        None, help="An address of this host the task container can reach, for an agent that runs inside it."
    )
    harbor: str | None = config_option(
        None, help="The harbor command line for the oracle check; found on PATH by default."
    )
    concurrency: int = config_option(DEFAULT_CONCURRENCY, help="Episodes in flight per play request.")
    designer_url: str | None = config_option(
        None, help="A Reef service the designer calls go to instead of this deployment's."
    )
    designer_token: str | None = config_option(None, help="The token for designer-url.")
    designer_model: str | None = config_option(
        None, help="The served model the designer asks for; the deployment's by default."
    )
    designer_scenario: str | None = config_option(
        None, help="The scenario the designer's calls and reports go to; the proposal's own scenario by default."
    )
    designer_timeout_s: float = config_option(
        DEFAULT_DESIGNER_TIMEOUT_S, help="Seconds one designer call may take; inference.timeout-s must allow it too."
    )
    designer_options: Mapping[str, Any] | None = dataclasses.field(
        default=None,
        metadata=config_metadata('Extra fields of the designer\'s chat request, e.g. {"reasoning_effort": "none"}.'),
    )
    designer_prompt: str = config_option(
        "fixed",
        help="Where the Designer's prompt comes from: fixed, or harness for the tree designer-url serves under designer-scenario.",
    )
    designer_poll_s: float = config_option(
        DEFAULT_DESIGNER_POLL_S,
        help="Seconds between two looks at the Designer's version while a generation waits for it to change.",
    )
    designer_wait_s: float = config_option(
        DEFAULT_DESIGNER_WAIT_S,
        help=(
            "Seconds a generation's first proposal waits for the Designer's deployment to serve a new release or "
            "runtime load id after the previous generation's reports; on timeout it proceeds with a warning."
        ),
    )
    ready_timeout: int = config_option(
        DEFAULT_READY_TIMEOUT, help="Seconds reef serve waits for the generator to answer."
    )

    def __post_init__(self) -> None:
        if not isinstance(self.tasks_root, str) or not self.tasks_root.strip():
            raise ValueError("generator.tasks-root must name a directory")
        if not 1 <= self.port <= 65535:
            raise ValueError("generator.port must be between 1 and 65535")
        if self.concurrency < 1:
            raise ValueError("generator.concurrency must be at least 1")
        if self.designer_timeout_s <= 0:
            raise ValueError("generator.designer-timeout-s must be positive")
        if self.designer_poll_s <= 0:
            raise ValueError("generator.designer-poll-s must be positive")
        if self.designer_wait_s <= 0:
            raise ValueError("generator.designer-wait-s must be positive")
        if self.ready_timeout <= 0:
            raise ValueError("generator.ready-timeout must be positive")
        if self.designer_scenario is not None and not self.designer_scenario.strip():
            raise ValueError("generator.designer-scenario must name a scenario when set")
        if self.designer_prompt not in DESIGNER_PROMPT_SOURCES:
            raise ValueError(f"generator.designer-prompt must be one of {DESIGNER_PROMPT_SOURCES}")
        if self.designer_prompt == "harness" and not (self.designer_scenario or "").strip():
            raise ValueError(
                "generator.designer-prompt: harness needs generator.designer-scenario, the scenario whose harness "
                "release is the Designer's prompt"
            )
        if self.agent is not None and not (self.agent.get("name") or self.agent.get("import_path")):
            raise ValueError("generator.agent must carry a Harbor agent name or an import_path")


def generator_settings(section: Mapping[str, Any]) -> GeneratorSettings:
    """Parse the ``generator`` section, in its public hyphen spelling or the field spelling; unknown fields are refused."""
    if not isinstance(section, Mapping):
        raise ValueError("generator must be an object")
    folded = {str(key).replace("-", "_"): value for key, value in section.items()}
    if len(folded) != len(section):
        raise ValueError("generator names a field twice, in both spellings")
    values = parse_config_values(config_arguments(GeneratorSettings, prefix=("generator",)), folded)
    return GeneratorSettings(**values)


def generator_service(config: dict[str, Any]) -> dict[str, Any] | None:
    """The generator's process definition, or None when the deployment has no ``generator`` section."""
    section = config.get("generator")
    if section is None:
        return None
    try:
        generator = generator_settings(section)
    except ValueError as exc:
        raise DeployConfigError(f"generator: {exc}") from exc
    # The same interpreter as every child reef serve starts.
    python = os.environ.get("REEF_PYTHON", sys.executable)
    probe_host = "127.0.0.1" if generator.host in ("0.0.0.0", "") else generator.host
    probe_host = f"[{probe_host}]" if ":" in probe_host and not probe_host.startswith("[") else probe_host
    return {
        "name": GENERATOR_SERVICE,
        "role": "generator",
        "command": [python, "-m", "reef.record2dataset"],
        "endpoint": f"http://{{host}}:{generator.port}",
        "ready": http_readiness_command(python, f"http://{probe_host}:{generator.port}/healthz"),
        "ready_timeout": generator.ready_timeout,
    }


def attach_generator_service(config: dict[str, Any]) -> None:
    """Put the generator before the HTTP service in ``config["services"]`` and make the HTTP service depend on it."""
    service = generator_service(config)
    if service is None:
        return
    services = config.get("services", [])
    http = next((entry for entry in services if entry.get("name") == "reef"), None)
    if http is None:
        raise DeployConfigError("a generator needs the Reef HTTP service in the deployment")
    http["depends_on"] = [*http.get("depends_on", []), GENERATOR_SERVICE]
    config["services"] = [service, *services]
