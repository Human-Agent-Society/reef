"""The HTTP child that the orchestrator of ``reef serve`` starts: ``python -m reef.deploy.service_process``.

It reads REEF_CONFIG, translates the YAML into service settings, and starts the app that
``reef.api.assembly`` builds from them.
"""

from reef.deploy.config_utils import DeployConfigError
from reef.deploy.orchestrator import run_service

if __name__ == "__main__":
    try:
        raise SystemExit(run_service())
    except DeployConfigError as exc:
        raise SystemExit(f"[reef] ERROR: {exc}") from exc
