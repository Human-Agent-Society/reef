"""Explicit reef-eval launcher for OpenEvolve 0.3.2."""

from __future__ import annotations

from usage import install_usage_tracking


def main() -> int:
    """Install reef-eval's usage instrumentation, then run OpenEvolve's CLI."""
    install_usage_tracking()

    from openevolve.cli import main as openevolve_main

    return openevolve_main()


if __name__ == "__main__":
    raise SystemExit(main())
