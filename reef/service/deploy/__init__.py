"""``reef serve`` — start a stack from a config.

``reef serve -c <stack>.yaml`` reads the config's ``services``
list and starts every declared process (SGLang, Slime driver, Reef, and so
on) in dependency order; see :mod:`reef.service.deploy.orchestrator`. The
Reef HTTP child is an internal service process: this package translates the
YAML into service settings (:mod:`reef.service.deploy.settings`) and
:mod:`reef.service.assembly` builds the dispatcher and app from those
settings.
"""

from reef.artifact.git_lfs import GitLFSRepositoryBackend
from reef.service.deploy.config import PROJECT_ROOT, DeployConfigError, load_config
from reef.service.deploy.orchestrator import main
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
