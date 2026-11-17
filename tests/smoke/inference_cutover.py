"""Run the cutover check of ``tests.support.inference_qualification`` against a real integration, from a shell.

``python -m tests.smoke.inference_cutover --target module:factory --output DIRECTORY`` calls ``factory`` of
``module`` with the output directory. The factory returns a context manager: it owns the isolated resources of the
integration and yields an ``InferenceQualification``. The command stores the check's records and ``report.json`` in
the output directory and prints the report. It refuses invalid arguments before it calls the factory, and exits with
status 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tests.support.inference_qualification import InferenceQualification, qualify_inference

REPOSITORY = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.smoke.inference_cutover",
        description="Run the two-artifact cutover check against the integration that a factory provides.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help=(
            "module:factory; the factory takes the output directory and returns a context manager that yields an "
            "InferenceQualification"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="an absolute path to a new or empty directory outside the repository",
    )
    arguments = parser.parse_args(argv)
    module_name, separator, factory_name = arguments.target.partition(":")
    if not separator or not module_name or not factory_name:
        parser.error(f"--target must be module:factory, not {arguments.target!r}")
    output = arguments.output
    if not output.is_absolute():
        parser.error(f"the output directory {output} must be an absolute path")
    resolved = output.resolve()
    if resolved == REPOSITORY or REPOSITORY in resolved.parents:
        parser.error(f"the output directory {output} must be outside the repository")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"the output directory {output} must not exist yet or be empty")
    namespace = vars(importlib.import_module(module_name))
    if factory_name not in namespace:
        parser.error(f"the module {module_name} defines no {factory_name}")
    factory = namespace[factory_name]
    output.mkdir(parents=True, exist_ok=True)
    with factory(output) as case:
        if not isinstance(case, InferenceQualification):
            raise TypeError(f"{arguments.target} yielded a {type(case).__name__}, not an InferenceQualification")
        report = asyncio.run(qualify_inference(case, output))
    text = json.dumps(report, indent=2) + "\n"
    (output / "report.json").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
