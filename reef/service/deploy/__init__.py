"""``reef serve`` — start managed inference, connect a provider, or run a configured stack.

Without a services list, inference settings assemble the record-only service,
optionally with managed SGLang, and weight recipes assemble the Slime driver
and HTTP service. Configuration parsing and process lifecycle use the same
deployment path as explicitly configured stacks.

When supplied, ``reef serve -c <stack>.yaml`` reads the config's ``services``
list and starts every declared process (SGLang, Slime driver, Reef, and so
on) in dependency order; see :mod:`reef.service.deploy.orchestrator`. The
Reef HTTP child is an internal service process: this package translates the
YAML into service settings (:mod:`reef.service.deploy.settings`) and
:mod:`reef.service.assembly` builds the dispatcher and app from those
settings.
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
