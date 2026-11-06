"""The GEPA candidate as named texts: one per evolvable file of the harness tree.

GEPA optimizes a ``dict[str, str]``: one instruction text per named
component of the program under search, and every reflective mutation
rewrites exactly one entry of that mapping. Here a component is one file of
the served tree whose part carries instructions (the rules, a skill, a
command), and its name is the file's tree path. A candidate's tree is
therefore the served tree with the candidate's texts in place, path for path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from reef.harness.adapters.parts import Format, TreeLayout
from reef.harness.tree import HarnessTree, TreePatch

#: The formats whose files are free text GEPA can rewrite. Config, code and
#: native files are structure, not instructions: a reflection reply is prose,
#: and letting it rewrite JSON or executable code would break the tree rather
#: than evolve it.
EVOLVABLE_FORMATS = (Format.TEXT, Format.SKILL, Format.COMMAND)


def texts_of(tree: HarnessTree, layout: TreeLayout, parts: Sequence[str]) -> dict[str, str]:
    """The tree as a GEPA candidate: the path of every file of one of ``parts``, to its text, in path order.

    ``parts`` is the method's evolvable subset, so a deployment can evolve
    only its rules while its skills stay fixed. ValueError for a name that
    is not a part of ``layout`` whose format is free text.
    """
    for name in parts:
        part = layout.parts.get(name)
        if part is None or part.format not in EVOLVABLE_FORMATS:
            evolvable = [part.name for part in layout.parts.values() if part.format in EVOLVABLE_FORMATS]
            raise ValueError(f"GEPA components must name text parts of the tree: {', '.join(evolvable)}")
    return {
        path: text
        for path, text in tree.files.items()
        if (location := layout.locate(path)) is not None and location.part.name in parts
    }


def candidate_tree(served: HarnessTree, texts: Mapping[str, str]) -> HarnessTree:
    """The served tree with a candidate's texts in place of its own: the tree a candidate's tasks run.

    Every other file of the served tree stays. The texts replace files of the
    *served* tree, not of the candidate's parent: a candidate descended from
    an older archive candidate carries every component where the two now
    differ, not just the one reflection touched. A path the served tree does
    not hold is skipped: GEPA rewrites components, it never invents them.
    """
    return served.apply(TreePatch({path: text for path, text in texts.items() if path in served.files}))


__all__ = ["EVOLVABLE_FORMATS", "candidate_tree", "texts_of"]
