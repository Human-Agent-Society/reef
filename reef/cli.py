"""reef CLI: entry point for reef.

Usage:
  reef serve --inference.upstream-url URL --inference.upstream-model MODEL  # connect a provider
  reef serve -c path/to/stack.yaml             # start a configured stack
  reef connect                               # link an existing runtime to the console

`reef serve` starts local inference or a provider deployment without YAML, or reads
a config's `services` list and starts its processes in dependency order.
Run `reef serve --help` for config options.

Deployment stacks use the `services` layout documented in the configuration
reference. Named recipe YAML is deployment data, not library data: point
``REEF_RECIPE_CONFIG_DIR`` at your own directory.
"""

from __future__ import annotations

import sys

COMMANDS = {"serve", "connect", "import"}


def help_text():
    return """\
usage: reef <command> [options]

  serve  Start inference, connect a provider, or run a configured stack
  connect  Connect an existing Reef runtime to the API platform
  import   Import a records JSONL file with resumable batch uploads

  -c CONFIG   Optional config file; omitted means configuration-free startup
  --version   Print the installed reef version

Examples:
  reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model
  reef serve -c path/to/local-sglang.yaml
  reef serve -c path/to/external-provider.yaml
  reef connect
"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    if not argv:
        print(help_text(), file=sys.stderr)
        sys.exit(2)

    if argv[0] in ("-h", "--help"):
        print(help_text())
        sys.exit(0)

    if argv[0] in ("-V", "--version"):
        from reef.core.version import __version__

        print(f"reef {__version__}")
        sys.exit(0)

    cmd = argv[0]
    rest = argv[1:]

    if cmd not in COMMANDS:
        print(f"reef: unknown command '{cmd}'\n", file=sys.stderr)
        print(help_text(), file=sys.stderr)
        sys.exit(2)

    if cmd == "import":
        from reef.service.record_import import main as import_main

        import_main(rest)
        return

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
