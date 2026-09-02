"""The GEPA candidate as named texts, and the diff back into mutations.

GEPA optimizes a ``dict[str, str]``: one instruction text per named
component of the program under search, and every reflective mutation
rewrites exactly one entry of that mapping. Reef's evolution backend speaks
a composition tree instead - ``(kind, config)`` pairs in tree order, with
entry ids stripped - so this module is the translation both ways: which
nodes are evolvable text, what GEPA calls each one, and how a rewritten
mapping becomes the ``Mutation`` sequence the backend applies.

A component key doubles as the entry id the mutation addresses, under the
seed id convention ``GEPARecipe`` validates: a ``rules`` node has entry id
``rules``, and a named node's entry id is its ``config.name``. Because
``propose`` never sees entry ids, that convention is the only thing that
lets a proposal point back at the node a text came from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.train.cordis_backend.strategies import Mutation

#: The node kinds whose config carries one free-text field GEPA can rewrite.
#: ``config`` and ``code_extension`` nodes are structure, not instructions:
#: a reflection reply is prose, and letting it rewrite JSON or executable
#: code would break the tree rather than evolve it.
EVOLVABLE_KINDS = ("rules", "skill", "agent_command")


def component_key(kind: str, config: Mapping[str, Any]) -> str | None:
    """GEPA's name for one node, or ``None`` when the node is not evolvable."""
    if kind == "rules":
        return "rules"
    if kind in ("skill", "agent_command"):
        name = config.get("name")
        if isinstance(name, str) and name:
            return f"{kind}:{name}"
    return None


def texts_of(nodes: Sequence[tuple[str, Any]], kinds: Sequence[str]) -> dict[str, str]:
    """The composition as a GEPA candidate: component key to text, tree order.

    ``kinds`` is the recipe's evolvable subset, so a deployment can evolve
    only its rules while its skills stay fixed. A node with no text yet
    still enters the mapping, empty: it is a component GEPA may fill.
    """
    texts: dict[str, str] = {}
    for kind, config in nodes:
        if kind not in kinds:
            continue
        options = config if isinstance(config, Mapping) else {}
        key = component_key(kind, options)
        if key is not None:
            texts[key] = str(options.get("text", ""))
    return texts


def entry_id(key: str) -> str:
    """The composition entry id a component key addresses."""
    kind, separator, name = key.partition(":")
    return name if separator else kind


def component_kind(key: str) -> str:
    """The node kind a component key belongs to."""
    return key.partition(":")[0]


def mutations(served: Mapping[str, str], child: Mapping[str, str]) -> tuple[Mutation, ...]:
    """One ``update`` per component whose text the child rewrote.

    The proposal is stated against the *served* composition, not against the
    child's parent: the backend applies mutations to the served tree, so a
    child descended from an older archive candidate carries every component
    where the two now differ, not just the one reflection touched. A key the
    served composition does not have is skipped - GEPA rewrites components,
    it never invents them, and an update to an absent entry would only make
    the backend skip the whole step.
    """
    changed = []
    for key, text in child.items():
        if key not in served or served[key] == text:
            continue
        kind = component_kind(key)
        name = entry_id(key)
        config = {"text": text} if kind == "rules" else {"name": name, "text": text}
        changed.append(Mutation("update", name, {"name": kind, "config": config}))
    return tuple(changed)
