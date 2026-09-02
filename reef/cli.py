"""reef CLI: entry point for reef.

Usage:
  reef serve -c path/to/stack.yaml             # start a configured stack
  reef adapters                                # list bundled + entry-point adapters

`reef serve` reads a config's `services` list and starts every declared
process in dependency order, including the internal Reef HTTP service.
Run `reef serve --help` for config options.

Deployment stacks use the `services` layout documented in the configuration
reference. Named recipe YAML is deployment data, not library data: point
``REEF_RECIPE_CONFIG_DIR`` at your own directory.
"""

from __future__ import annotations

import json
import sys

_COMMANDS = {"serve", "adapters"}


def _help_text():
    return """\
usage: reef <command> [options]

  serve     Start a stack from a config
  adapters  List available harness adapters

  -c CONFIG   Config file (default: reef.yaml or $REEF_CONFIG)
  --version   Print the installed reef version

Examples:
  reef serve -c path/to/local-sglang.yaml
  reef serve -c path/to/external-provider.yaml
  reef adapters
  reef adapters --format json
"""


def _adapters_main(argv):
    output = "table"
    it = iter(argv)
    for arg in it:
        if arg in ("-h", "--help"):
            print(
                "usage: reef adapters [--format table|json]\n"
                "\n"
                "List every harness adapter this reef installation resolves.\n"
            )
            return 0
        if arg.startswith("--format="):
            output = arg.split("=", 1)[1]
        elif arg == "--format":
            value = next(it, None)
            if value is None:
                print("reef adapters: --format needs a value", file=sys.stderr)
                return 2
            output = value
        else:
            print(f"reef adapters: unknown argument {arg!r}", file=sys.stderr)
            return 2
    # Delay heavy imports until the subcommand actually runs.
    from reef.harness.adapters import available_adapters, get_adapter
    from reef.harness.descriptor import DescriptorError

    entries = []
    for name in available_adapters():
        try:
            descriptor = get_adapter(name)
        except DescriptorError as exc:
            entries.append({"name": name, "error": str(exc)})
            continue
        install = descriptor.install
        entries.append(
            {
                "name": name,
                "binary": descriptor.binary,
                "trajectory_format": descriptor.trajectory_format,
                "model_bindings": sorted(descriptor.model_binding),
                "install": (
                    None
                    if install is None
                    else {"kind": install.kind, "package": install.package, "version": install.version}
                ),
            }
        )
    if output == "json":
        print(json.dumps({"adapters": entries}, indent=2))
        return 0
    if output != "table":
        print(f"reef adapters: unknown --format {output!r}", file=sys.stderr)
        return 2
    header = ("name", "binary", "trajectory", "install")
    rows = [header]
    for entry in entries:
        if "error" in entry:
            rows.append((entry["name"], "-", "-", f"(error: {entry['error']})"))
            continue
        install = entry["install"]
        install_cell = "-" if install is None else f"{install['package']}@{install['version']}"
        rows.append((entry["name"], entry["binary"], entry["trajectory_format"], install_cell))
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    for row in rows:
        print("  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)))
    return 0


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

    if cmd == "adapters":
        sys.exit(_adapters_main(rest))

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
