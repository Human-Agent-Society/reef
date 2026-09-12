"""``reef serve`` — start managed inference, connect a provider, or run a configured stack.

Version 2 and CLI-only input describe components, not process definitions.
Reef assembles inference, training and HTTP processes; selected recipes supply
method-specific dependencies in Python. Unversioned files retain their explicit
``services`` process contract. All paths share the existing executor lifecycle,
readiness and cleanup machinery. HTTP assembly lives in :mod:`reef.service.assembly`.
"""

from reef.artifact.git_lfs import GitLFSRepositoryBackend
from reef.service.deploy.config import PROJECT_ROOT, DeployConfigError, load_config
from reef.service.deploy.orchestrator import DeployStartupError, main
from reef.service.deploy.settings import ServiceSettings, build_parser, run_service, service_settings_from_config


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
