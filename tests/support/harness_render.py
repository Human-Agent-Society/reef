"""Shared pieces of the harness tree tests: native files built from their fields, the native seed, a rendered root.

A native tool, hook or loop file holds its code, then its declaration as
literal constants, which the loader reads without running the code.
``native_tool``, ``native_hook`` and ``native_loop`` write one such file from
its fields, and ``native_json`` writes a graph or an agent file, each as the
one-entry mapping a test merges into a tree's files. ``NATIVE_SEED`` is the
starter tree of the evolve-your-harness tutorial: its tools, its
``loop_guard`` hook, its ``main`` graph and a skill. ``write_rendered``
renders a tree for one adapter and writes the files under a root, for a test
that runs a loader or a binary on them. ``descriptor_data`` and
``write_descriptor`` give a test a bundled adapter's descriptor to change and
load from a directory of its own, beside the adapter's ``tree.yaml``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

import reef.harness.adapters
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.adapters.parts import TREE_FILE
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.tree import HarnessTree, render_tree
from reef.harness.tree.checks import NATIVE_LOOP_DEFAULT_MAX_STEPS

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The tutorial's native starter tree.
NATIVE_SEED = HarnessTree.read(REPO_ROOT / "tutorials" / "evolve-your-harness" / "seeds" / "native")
#: The seed's native files without its skill: the four tools, the loop_guard hook and the main graph.
NATIVE_SEED_FILES = {path: text for path, text in NATIVE_SEED.files.items() if path.startswith("native/")}
#: The seed's four tools alone.
NATIVE_SEED_TOOLS = {path: text for path, text in NATIVE_SEED.files.items() if path.startswith("native/tools/")}


def module_text(code: str, declaration: Mapping[str, Any]) -> str:
    """``code``, then one ``KEY = literal`` line per declaration field, in the order given."""
    lines = "\n".join(f"{key} = {value!r}" for key, value in declaration.items())
    return f"{code.rstrip()}\n\n{lines}\n"


def native_tool(
    name: str,
    code: str,
    *,
    description: str = "",
    parameters: Mapping[str, Any] | None = None,
    capabilities: Sequence[str] = (),
) -> dict[str, str]:
    """The file of a native tool whose ``code`` defines ``run(args, workdir)``."""
    declaration = {
        "NAME": name,
        "DESCRIPTION": description,
        "PARAMETERS": dict(parameters or {}),
        "CAPABILITIES": list(capabilities),
    }
    return {f"native/tools/{name}.py": module_text(code, declaration)}


def native_hook(name: str, event: str, code: str) -> dict[str, str]:
    """The file of a native hook whose ``code`` defines ``listen(payload, next)`` at ``event``."""
    return {f"native/hooks/{name}.py": module_text(code, {"NAME": name, "EVENT": event})}


def native_loop(name: str, code: str, max_steps: int = NATIVE_LOOP_DEFAULT_MAX_STEPS) -> dict[str, str]:
    """The file of a native loop whose ``code`` defines ``run_turn(ctx)``."""
    return {f"native/loops/{name}.py": module_text(code, {"NAME": name, "MAX_STEPS": max_steps})}


def native_json(kind: str, value: Mapping[str, Any]) -> dict[str, str]:
    """The file of a native graph (``kind`` ``graphs``) or agent (``agents``), named by ``value["name"]``."""
    return {f"native/{kind}/{value['name']}.json": json.dumps(value, indent=2, sort_keys=True) + "\n"}


def write_rendered(
    root: Path, tree: HarnessTree, descriptor: AdapterDescriptor, binding: ModelBinding | None = None
) -> dict[str, str]:
    """``tree`` rendered for ``descriptor``, pointed at ``binding`` when one is given, and written under ``root``."""
    overlays = () if binding is None else binding.config_overlays(descriptor)
    files = render_tree(tree, descriptor, overlays)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return files


__all__ = [
    "NATIVE_SEED",
    "NATIVE_SEED_FILES",
    "NATIVE_SEED_TOOLS",
    "REPO_ROOT",
    "module_text",
    "native_hook",
    "native_json",
    "native_loop",
    "native_tool",
    "write_rendered",
]


#: The directory of the bundled adapters.
ADAPTERS = Path(reef.harness.adapters.__file__).parent


def descriptor_data(adapter: str) -> dict[str, Any]:
    """The bundled ``adapter``'s ``descriptor.yaml`` as data, for a test to change."""
    return yaml.safe_load((ADAPTERS / adapter / "descriptor.yaml").read_text(encoding="utf-8"))


def write_descriptor(directory: Path, adapter: str, data: Mapping[str, Any]) -> Path:
    """Write ``data`` as ``directory/descriptor.yaml`` beside a copy of the bundled ``adapter``'s ``tree.yaml``,
    and return the descriptor's path for ``load_descriptor``."""
    shutil.copyfile(ADAPTERS / adapter / TREE_FILE, directory / TREE_FILE)
    path = directory / "descriptor.yaml"
    path.write_text(yaml.safe_dump(dict(data)), encoding="utf-8")
    return path
