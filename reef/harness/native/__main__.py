"""``python -m reef.harness.native`` and the ``reef-native`` console script.

``reef-native -p PROMPT`` is the episode form, one process and one turn, and
is unchanged. ``serve``, ``turn``, ``mount`` and ``status`` are the serve
form: a resident process on an installed tree and the commands that talk to
it over its socket.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from reef.harness.native import main as episode_main

SUBCOMMANDS = ("serve", "turn", "mount", "status")


def _parser() -> argparse.ArgumentParser:
    from reef.harness.native.serve import DEFAULT_POLL_INTERVAL_S, DEFAULT_TURN_TIMEOUT_S, FOLLOW_MODES

    parser = argparse.ArgumentParser(prog="reef-native", description="Reef's native coding agent, the serve form.")
    commands = parser.add_subparsers(dest="command", required=True)
    tree_help = "the pulled tree: the directory that holds native/ and the release sidecar"

    serve = commands.add_parser("serve", help="serve an installed tree as a resident process")
    serve.add_argument("--tree", type=Path, required=True, help=tree_help)
    serve.add_argument(
        "--scenario",
        default=os.environ.get("REEF_HARNESS_SCENARIO") or os.environ.get("REEF_SCENARIO"),
        help="the Reef scenario the tree belongs to (default: REEF_HARNESS_SCENARIO)",
    )
    serve.add_argument(
        "--follow", choices=FOLLOW_MODES, default="head", help="mount a new head at once, or announce it"
    )
    serve.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S, metavar="SECONDS")
    serve.add_argument(
        "--self-tools", action="store_true", help="give the model harness_inspect, harness_try, harness_propose"
    )
    serve.add_argument(
        "--socket", type=Path, help="the Unix socket to listen on (default: native/serve.sock under the tree)"
    )
    serve.add_argument("--reef-url", help="the Reef to poll and proxy to (default: native/models.json base_url)")
    serve.add_argument("--turn-timeout", type=float, default=DEFAULT_TURN_TIMEOUT_S, metavar="SECONDS")

    turn = commands.add_parser("turn", help="send one turn to a serve process and print its events")
    turn.add_argument("--tree", type=Path, required=True, help=tree_help)
    turn.add_argument("-p", "--prompt", required=True, help="the turn's prompt")
    turn.add_argument("--session", help="continue this session; a new one starts without it")
    turn.add_argument("--workdir", help="the workspace the tools run in (default: the current directory)")
    turn.add_argument("--quiet", action="store_true", help="print only the final text")
    turn.add_argument("--socket", type=Path)

    mount = commands.add_parser("mount", help="mount one release on a serve process")
    mount.add_argument("release_id")
    mount.add_argument("--tree", type=Path, required=True, help=tree_help)
    mount.add_argument("--socket", type=Path)

    status = commands.add_parser("status", help="print a serve process's status")
    status.add_argument("--tree", type=Path, required=True, help=tree_help)
    status.add_argument("--socket", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SUBCOMMANDS:
        return episode_main(argv)
    args = _parser().parse_args(argv)
    from reef.harness.native.serve import run_command  # late: the serve form pulls in the training package's loader

    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())
