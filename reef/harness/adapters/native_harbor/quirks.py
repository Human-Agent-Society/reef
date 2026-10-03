"""native_harbor quirks: which tree code may run in the process that writes the verifier row.

The native loop imports every hook module into its own process, and a ``native_loop`` module runs there as the root
turn. Under this adapter that process is the runner that writes the ``verifier`` row the episode is scored by, so a
tree whose hook or loop code is not what Reef ships could write its own score. The episode is refused unless every
hook is a seed hook byte for byte and the tree carries no loop, under every executor: Reef's sandbox jails the runner
from the host, not the row from the code in the runner. In the sandbox the task must also run in Harbor's remote E2B
environment, as for terminus: local Docker cannot run in the jail. Tools are not checked: under this adapter a tool
module is imported only in the task container.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePosixPath

from reef.harness.adapters.descriptor import ExecutionValidator
from reef.harness.episodes.executor import EpisodeExecutor, EpisodeLaunchError, SandboxExecutor
from reef.harness.runners.native.seed import SEED_HOOKS
from reef.harness.runners.native.task import ENVIRONMENT_ENV
from reef.harness.tree.render import render_native_module

HOOKS_PATH = PurePosixPath("native/hooks")
LOOPS_PATH = PurePosixPath("native/loops")
TREE_PATH = "native/tree.json"


def unshipped_code(files: Mapping[str, str]) -> list[str]:
    """``native_loop <name>`` for every loop and ``native_hook <name>`` for every hook whose module is not a seed
    hook's, in the rendered files and among the enabled entries of the tree file, which a tree boot mounts from."""
    shipped = {render_native_module("native_hook", hook["config"]) for hook in SEED_HOOKS}
    found: list[str] = []
    for path, text in sorted(files.items()):
        pure = PurePosixPath(path)
        if pure.suffix != ".py":
            continue
        if pure.parent == LOOPS_PATH:
            found.append(f"native_loop {pure.stem}")
        elif pure.parent == HOOKS_PATH and text not in shipped:
            found.append(f"native_hook {pure.stem}")
    try:
        entries = json.loads(files.get(TREE_PATH, "[]"))
    except json.JSONDecodeError as exc:
        raise EpisodeLaunchError(f"native_harbor cannot read {TREE_PATH}: {exc}") from exc
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, dict) or entry.get("disabled") or not isinstance(entry.get("config"), dict):
            continue
        config = entry["config"]
        if entry.get("name") == "native_loop":
            found.append(f"native_loop {config.get('name')}")
        elif entry.get("name") == "native_hook" and render_native_module("native_hook", config) not in shipped:
            found.append(f"native_hook {config.get('name')}")
    return list(dict.fromkeys(found))


class NativeHarborExecutionValidator(ExecutionValidator):
    def __call__(self, files: Mapping[str, str], executor: EpisodeExecutor) -> None:
        """Refuse tree code in the runner under every executor, and local Docker nested in Reef's jail."""
        if isinstance(executor, SandboxExecutor):
            if executor.env.get(ENVIRONMENT_ENV) != "e2b":
                raise EpisodeLaunchError(
                    "native_harbor Docker cannot run under evolution.executor: sandbox; "
                    f"set {ENVIRONMENT_ENV}=e2b and include it in evolution.sandbox.env_from"
                )
            if not executor.egress_hosts or not executor.env.get("E2B_API_KEY"):
                raise EpisodeLaunchError(
                    "sandboxed native_harbor requires egress_hosts and E2B_API_KEY in sandbox.env_from"
                )
        unshipped = unshipped_code(files)
        if unshipped:
            raise EpisodeLaunchError(
                "native_harbor imports hook and loop code into the process that writes the verifier row; "
                f"{', '.join(unshipped)} is not code Reef ships, so no executor runs it"
            )


validate_execution = NativeHarborExecutionValidator()
