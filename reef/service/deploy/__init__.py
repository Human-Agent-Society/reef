"""``reef serve`` — start managed inference, connect a provider, or run a configured stack.

Version 2 and CLI-only input describe components, not process definitions.
Reef assembles inference, training and HTTP processes; selected recipes supply
method-specific dependencies in Python. Unversioned files retain their explicit
``services`` process contract. All paths share the existing executor lifecycle,
readiness and cleanup machinery. HTTP assembly lives in :mod:`reef.service.assembly`.

Module responsibilities:
    config: YAML loading, environment interpolation and recipe source paths.
    service_config: Typed shared settings consumed by HTTP app assembly.
    component_config: Selected component schemas, public layout and validation.
    cli: CLI help, dotted override syntax and precedence.
    inference / training: Component-specific process and runtime assembly.
    execution: Process definition validation and executor selection.
    process / guard: Worker process lifecycle and remote-owner cleanup.
    orchestrator: Launch coordination, supervision and HTTP child entrypoint.

Shared type conversion lives in ``reef.core.config``; native argument encoding
lives in ``reef.runtime.executor.arguments``. Import those owners directly.
"""

from reef.artifact.git_lfs import GitLFSRepositoryBackend
from reef.service.deploy.cli import build_parser
from reef.service.deploy.config import PROJECT_ROOT, DeployConfigError, load_config
from reef.service.deploy.orchestrator import DeployStartupError, main, run_service
from reef.service.deploy.service_config import ServiceSettings, service_settings_from_config


def build_app(settings, **kwargs):
    from reef.service.assembly import build_app as _build_app

    return _build_app(settings, **kwargs)


def build_dispatcher(settings, **kwargs):
    from reef.service.assembly import build_dispatcher as _build_dispatcher

    return _build_dispatcher(settings, **kwargs)


__all__ = [
    "PROJECT_ROOT",
    "DeployConfigError",
    "DeployStartupError",
    "GitLFSRepositoryBackend",
    "ServiceSettings",
    "build_app",
    "build_dispatcher",
    "build_parser",
    "load_config",
    "main",
    "run_service",
    "service_settings_from_config",
]
