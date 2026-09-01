"""reef CLI: entry point for reef.

Usage:
  reef serve -c path/to/stack.yaml             # start a configured stack

`reef serve` reads a config's `services` list and starts every declared
process in dependency order, including the internal Reef HTTP service.
Run `reef serve --help` for config options.

Deployment stacks use the `services` layout documented in the configuration
reference. Named per-scenario recipe YAML is deployment data, not library
data: point ``REEF_RECIPE_CONFIG_DIR`` at your own directory.
"""

from __future__ import annotations

import sys

_COMMANDS = {"serve"}


def _help_text():
    return """\
usage: reef <command> [options]

  serve  Start a stack from a config

  -c CONFIG   Config file (default: reef.yaml or $REEF_CONFIG)
  --version   Print the installed reef version

Examples:
  reef serve -c path/to/local-sglang.yaml
  reef serve -c path/to/external-provider.yaml
"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(_help_text())
        sys.exit(0 if argv else 1)

    if argv[0] in ("-V", "--version"):
        from reef import __version__

        print(f"reef {__version__}")
        sys.exit(0)

    cmd = argv[0]
    rest = argv[1:]

    if cmd not in _COMMANDS:
        print(f"reef: unknown command '{cmd}'\n", file=sys.stderr)
        print(_help_text(), file=sys.stderr)
        sys.exit(2)

    from reef.service.deploy import DeployConfigError
    from reef.service.deploy import main as _serve_main

    try:
        _serve_main(rest)
    except DeployConfigError as exc:
        # Deploy config errors are typed library errors; the CLI owns the exit.
        print(f"[reef] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
