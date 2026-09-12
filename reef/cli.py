"""reef CLI: entry point for reef.

Usage:
  reef serve --upstream-url URL --upstream-model MODEL  # connect a provider
  reef serve -c path/to/stack.yaml             # start a configured stack
  reef connect                               # link an existing runtime to the console

`reef serve` starts a record-only provider deployment without YAML, or reads
a config's `services` list and starts its processes in dependency order.
Run `reef serve --help` for config options.

Deployment stacks use the `services` layout documented in the configuration
reference. Named recipe YAML is deployment data, not library data: point
``REEF_RECIPE_CONFIG_DIR`` at your own directory.
"""

from __future__ import annotations

import sys

_COMMANDS = {"serve", "connect"}


def _help_text():
    return """\
usage: reef <command> [options]

  serve  Connect an external provider or start a configured stack
  connect  Connect an existing Reef runtime to the API platform

  -c CONFIG   Optional config file; omitted means configuration-free startup
  --version   Print the installed reef version

Examples:
  reef serve --upstream-url http://localhost:8000 --upstream-model my-model
  reef serve -c path/to/local-sglang.yaml
  reef serve -c path/to/external-provider.yaml
  reef connect
"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    if not argv:
        print(_help_text(), file=sys.stderr)
        sys.exit(2)

    if argv[0] in ("-h", "--help"):
        print(_help_text())
        sys.exit(0)

    if argv[0] in ("-V", "--version"):
        from reef.core.version import __version__

        print(f"reef {__version__}")
        sys.exit(0)

    cmd = argv[0]
    rest = argv[1:]

    if cmd not in _COMMANDS:
        print(f"reef: unknown command '{cmd}'\n", file=sys.stderr)
        print(_help_text(), file=sys.stderr)
        sys.exit(2)

    if cmd == "connect":
        from reef.service.connector import main as _connect_main

        _connect_main(rest)
        return

    from reef.service.deploy import DeployConfigError, DeployStartupError
    from reef.service.deploy import main as _serve_main

    try:
        _serve_main(rest)
    except DeployConfigError as exc:
        # Deploy config errors are typed library errors; the CLI owns the exit.
        print(f"[reef] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except DeployStartupError as exc:
        print(f"[reef] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
