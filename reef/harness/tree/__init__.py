"""The harness source tree: what a harness is, how it changes, and how it renders.

``tree``
    ``HarnessTree``, the immutable path-to-text value a release stores, and
    ``TreePatch``, the change a proposer returns.
``layout``
    The paths a tree may hold and the kind of each (``TreeKind``, ``locate``).
``checks``
    Content checks for single files, shared with the native loop's loader.
``validate``
    ``validate_tree`` and ``validate_patch``: every problem of a tree for one
    adapter, in one result.
``render``
    ``render_tree``: a tree and an optional model binding to the files an
    adapter's binary reads, for an episode or for a client.
``workspace``
    ``TreeWorkspace``: an editable copy for an agent session, read back as a patch.
"""

from reef.harness.tree.layout import TreeKind, TreeLocation, locate, source_path
from reef.harness.tree.render import RenderError, render_tree
from reef.harness.tree.tree import HarnessTree, TreePatch
from reef.harness.tree.validate import ValidationResult, validate_patch, validate_tree
from reef.harness.tree.workspace import TreeWorkspace, WorkspaceResult

__all__ = [
    "HarnessTree",
    "RenderError",
    "TreeKind",
    "TreeLocation",
    "TreePatch",
    "TreeWorkspace",
    "ValidationResult",
    "WorkspaceResult",
    "locate",
    "render_tree",
    "source_path",
    "validate_patch",
    "validate_tree",
]
