"""The harness source tree: what a harness is, how it changes, and how it renders.

``tree``
    ``HarnessTree``, the immutable path-to-text value a release stores, and
    ``TreePatch``, a change to a tree.
``checks``
    Content checks for single files, shared with the native loop's loader.
``validate``
    ``validate_tree`` and ``validate_patch``: every problem of a tree for one
    adapter, in one result.
``render``
    ``render_tree``: a tree and an optional model binding to the files an
    adapter's binary reads, for a run or for a client.

The paths a tree may hold for one adapter, and what each one is, are the
adapter's parts (``reef.harness.adapters.parts``), declared in the ``evolve``
section of its ``adapter.yaml``.
"""

from reef.harness.tree.render import RenderError, render_tree
from reef.harness.tree.tree import HarnessTree, TreePatch
from reef.harness.tree.validate import ValidationResult, validate_patch, validate_tree

__all__ = [
    "HarnessTree",
    "RenderError",
    "TreePatch",
    "ValidationResult",
    "render_tree",
    "validate_patch",
    "validate_tree",
]
