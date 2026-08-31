"""Internal HTTP child launched by the config-driven Reef orchestrator.

Reads REEF_CONFIG through the deploy subpackage (translating YAML into
service settings) and starts the app assembly builds from them.
"""

from reef.service.deploy import DeployConfigError, run_service

if __name__ == "__main__":
    try:
        raise SystemExit(run_service())
    except DeployConfigError as exc:
        raise SystemExit(f"[reef] ERROR: {exc}") from exc
