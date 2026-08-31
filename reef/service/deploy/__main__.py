"""Allow ``python -m reef.service.deploy`` as an alias for ``reef serve``."""

from reef.service.deploy import DeployConfigError, main

if __name__ == "__main__":
    try:
        main()
    except DeployConfigError as exc:
        raise SystemExit(f"[reef] ERROR: {exc}") from exc
