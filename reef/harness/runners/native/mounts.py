"""Rendered native host directories, built before replacing the active mount."""

import shutil
from collections.abc import Sequence
from pathlib import Path

from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.runners.native.host import NativeHost
from reef.harness.runners.native.selftools import HostTool
from reef.harness.tree import HarnessTree, render_tree


class HostMounts:
    """The rendered host directories of one resident native server.

    ``root`` is a Path and ``descriptor`` is the native adapter descriptor.
    ``mount_count`` starts at zero and counts rendered build attempts. Each
    attempt uses root/<count>. A failed build removes its directory.
    ``current_directory`` and ``current_host`` are both None until the first
    successful mount; afterward they name that mount and its loaded host.
    ``live_tree`` is empty before that mount and is its source tree afterward.
    A successful mount replaces these three values before removing the old
    directory. The server serializes builds, mounts, and cleanup with turns.
    ``clear`` removes the root without changing the in-memory host.
    """

    def __init__(self, root: Path, descriptor: AdapterDescriptor) -> None:
        if not isinstance(root, Path):
            raise TypeError("mount root must be a Path")
        if not isinstance(descriptor, AdapterDescriptor) or descriptor.name != "native":
            raise ValueError("host mounts require the native adapter descriptor")
        self.root = root
        self.descriptor = descriptor
        self.mount_count = 0
        self.current_directory: Path | None = None
        self.current_host: NativeHost | None = None
        self.live_tree = HarnessTree({})

    @property
    def host(self) -> NativeHost:
        """Return the active host; raise RuntimeError before a successful mount."""
        if self.current_host is None:
            raise RuntimeError("no native host is mounted")
        return self.current_host

    def replace(self, host: NativeHost, directory: Path, tree: HarnessTree) -> None:
        """Activate a built host and its source tree, then remove the previous directory."""
        previous = self.current_directory
        self.current_host, self.current_directory, self.live_tree = host, directory, tree
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)

    def build(self, tree: HarnessTree, tools: Sequence[HostTool]) -> tuple[NativeHost, Path]:
        """Render and load a candidate with built-in tools, removing its directory on failure.

        Return its host and directory without activating them. A tree tool
        that conflicts with a built-in tool fails to load.
        """
        files = render_tree(tree, self.descriptor)
        self.mount_count += 1
        directory = self.root / str(self.mount_count)
        shutil.rmtree(directory, ignore_errors=True)
        try:
            for relative, text in files.items():
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            host = NativeHost.from_root(directory / "native")
            for tool in tools:
                host.add_tool(tool)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return host, directory

    def clear(self) -> None:
        """Remove all rendered directories, including any left by a stopped process."""
        shutil.rmtree(self.root, ignore_errors=True)
