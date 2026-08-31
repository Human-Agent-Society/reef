#!/usr/bin/env python3
"""Merge one candidate witness into the cumulative solution ledger."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn


def _load_ledger_module():
    public_path = str(Path(__file__).resolve().parent / "public")
    while public_path in sys.path:
        sys.path.remove(public_path)
    sys.path.insert(0, public_path)
    import lwe_challenge.ledger as ledger_module

    return ledger_module


if __name__ != "__main__":
    ledger = _load_ledger_module()


def merge_solution(
    ledger_path: str | Path,
    instance_id: str,
    secret: Sequence[int],
) -> int:
    """Merge one new witness through the canonical locked transaction."""

    updated = ledger.merge_witness_transaction(
        ledger_path,
        instance_id=instance_id,
        secret=secret,
        replace=False,
    )
    return len(updated.solutions)


class _SanitizedArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        self.exit(2, "error: invalid command line\n")


def _parser() -> argparse.ArgumentParser:
    parser = _SanitizedArgumentParser()
    parser.add_argument("instance_id", metavar="INSTANCE_ID")
    parser.add_argument("secret", metavar="SECRET", nargs="?")
    parser.add_argument("--ledger", default="/app/solution.json")
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args, unknown = parser.parse_known_args()
    if args.secret is None and len(unknown) == 1:
        args.secret = unknown[0]
        unknown = []
    if args.secret is None or unknown:
        parser.error("INSTANCE_ID and SECRET are required")
    try:
        secret = tuple(int(component, 10) for component in args.secret.split(","))
    except ValueError:
        parser.error("SECRET must be a comma-separated integer vector")

    try:
        if args.replace:
            updated = ledger.merge_witness_transaction(
                args.ledger,
                instance_id=args.instance_id,
                secret=secret,
                replace=True,
            )
            solved_count = len(updated.solutions)
        else:
            solved_count = merge_solution(args.ledger, args.instance_id, secret)
    except (OSError, TypeError, ValueError):
        parser.exit(2, "error: unable to update ledger\n")

    print(solved_count)
    return 0


if __name__ == "__main__":
    try:
        ledger = _load_ledger_module()
    except Exception:
        sys.stderr.write("error: unable to update ledger\n")
        raise SystemExit(2)
    raise SystemExit(main())
