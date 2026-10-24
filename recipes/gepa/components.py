"""The GEPA candidate as named texts: one per evolvable file of the harness tree.

GEPA optimizes a ``dict[str, str]``: one instruction text per named
component of the program under search, and every reflective mutation
rewrites exactly one entry of that mapping. Here a component is one file of
the served tree whose kind carries instructions (``rules.md``, a skill, a
command), and its name is the file's tree path. A rewritten mapping is
therefore a patch against the served tree, path for path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from reef.harness.tree import HarnessTree, TreeKind, TreePatch, locate

#: The kinds whose file is free text GEPA can rewrite. Config and extension
#: files are structure, not instructions: a reflection reply is prose, and
#: letting it rewrite JSON or executable code would break the tree rather
#: than evolve it.
EVOLVABLE_KINDS = (TreeKind.RULES, TreeKind.SKILL, TreeKind.AGENT_COMMAND)


def texts_of(tree: HarnessTree, kinds: Sequence[TreeKind]) -> dict[str, str]:
    """The tree as a GEPA candidate: the path of every file of one of ``kinds``, to its text, in path order.

    ``kinds`` is the recipe's evolvable subset, so a deployment can evolve
    only its rules while its skills stay fixed.
    """
    return {
        path: text
        for path, text in tree.files.items()
        if (location := locate(path)) is not None and location.kind in kinds
    }


def patch_to(served: Mapping[str, str], child: Mapping[str, str]) -> TreePatch:
    """The patch that writes each component whose text the child rewrote.

    The patch is stated against the *served* tree, not against the child's
    parent: a child descended from an older archive candidate carries every
    component where the two now differ, not just the one reflection touched.
    A path the served tree does not hold is skipped: GEPA rewrites
    components, it never invents them.
    """
    return TreePatch({path: text for path, text in child.items() if path in served and served[path] != text})


def candidate_tree(served: HarnessTree, texts: Mapping[str, str]) -> HarnessTree:
    """The served tree with a candidate's texts in place of its own: the tree a minibatch episode runs."""
    return served.apply(patch_to(served.files, texts))


__all__ = ["EVOLVABLE_KINDS", "candidate_tree", "patch_to", "texts_of"]
