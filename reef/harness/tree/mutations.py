"""Mutation values and admission rules shared by harness serving and training."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from reef.core.errors import ReefError
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.compose import Context, FiberState
from reef.harness.compose.loader import EntryOptions, Loader
from reef.harness.tree.nodes import NODE_KINDS, RESERVED_ENTRY_IDS, flat_entry_refusal
from reef.harness.tree.render import RenderError, render_composition


class MutationError(ReefError):
    """A proposed mutation could not be applied."""


@dataclass(frozen=True)
class Mutation:
    """One proposed change to the composition tree, by entry id.

    ``create`` and ``update`` carry ``options`` (entry options without the
    id, e.g. ``{"name": "rules", "config": {"text": ...}}``); an ``update``
    merges them into the entry, with ``None`` values deleting keys, exactly
    as the compose loader reconciles. Ids are root-level: the minimal layer
    composes a flat tree, so nested (``:``-qualified) ids are rejected.
    """

    op: str
    id: str
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        mutation_ops = ("create", "update", "remove")
        if self.op not in mutation_ops:
            raise MutationError(f"mutation op must be one of {mutation_ops}, got {self.op!r}")
        if not self.id or ":" in self.id:
            raise MutationError(f"mutation id must be a non-empty root-level id, got {self.id!r}")
        if self.op in ("create", "update") and not isinstance(self.options, Mapping):
            raise MutationError(f"{self.op} mutation requires options")
        if self.op == "remove" and self.options is not None:
            raise MutationError("remove mutation takes no options")


def _nodes_from(entries: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, Any], ...]:
    return tuple((str(options["name"]), options.get("config")) for options in entries if not options.get("disabled"))


def _loader_entries(loader: Loader) -> list[EntryOptions]:
    """Live entry options in tree order."""
    result = []
    for options in loader.root.data:
        entry = loader.store.get(str(options.get("id")))
        if entry is not None:
            result.append(dict(entry.options))
    return result


def _unrenderable(descriptor: AdapterDescriptor, kind: str) -> str | None:
    """Why the adapter cannot render an entry of ``kind``; config nodes render through the config targets."""
    if kind == "config" or kind in descriptor.node_paths:
        return None
    return f"adapter {descriptor.name!r} does not render {kind} nodes"


def _load_error(loader: Loader, id_: str, descriptor: AdapterDescriptor) -> str | None:
    """Why the entry cannot serve: the flat tree rule, its kind's admission, its fiber's state, then whether the adapter renders it."""
    entry = loader.resolve(id_)
    # The seed and a recovered state never pass _apply_mutation, so the flat tree rule is read here as well.
    refusal = flat_entry_refusal(entry.options)
    if refusal is not None:
        return refusal
    kind = str(entry.options.get("name"))
    if entry.disabled:
        # Disabled is a serving state, not a validation bypass (#476): a
        # disabled entry builds no fiber, but its options persist in the
        # state verbatim, so its kind's admission gate runs directly here.
        plugin = NODE_KINDS.get(kind)
        if plugin is None:
            return f"unknown node kind {kind!r}"
        try:
            plugin(None, entry.options.get("config"))
        except ValueError as error:
            return str(error)
        return _unrenderable(descriptor, kind)
    fiber = entry.fiber
    if fiber is None:
        return f"unknown node kind {kind!r}"
    if fiber.error is not None:
        return str(fiber.error)
    if fiber.state is not FiberState.ACTIVE:
        return f"node fiber is {fiber.state.name}"
    return _unrenderable(descriptor, kind)


def _resolve(loader: Loader, id_: str) -> Any:
    try:
        return loader.resolve(id_)
    except LookupError as exc:
        raise MutationError(str(exc)) from exc


def _apply_mutation(loader: Loader, mutation: Mutation) -> None:
    """One mutation on the loader: create refuses an existing id, update a missing id or a changed kind, remove a missing id; every op refuses a group entry and a reserved id."""
    if mutation.id in RESERVED_ENTRY_IDS:
        # The seed and recovered state carry these entries; proposals cannot change them.
        raise MutationError(
            f"mutation {mutation.op} {mutation.id!r} rejected: entry {mutation.id!r} is reef's own, "
            "and a proposal cannot create, update or remove it"
        )
    if mutation.options is not None:
        # The loader would route a group entry to its Group plugin and mount the children through the kinds'
        # plugins, unseen by every check that walks the root; the tree is flat, so the shape never loads.
        refusal = flat_entry_refusal(mutation.options, partial=mutation.op == "update")
        if refusal is not None:
            raise MutationError(f"mutation {mutation.op} {mutation.id!r} rejected: {refusal}")
    if mutation.op == "create":
        try:
            loader.resolve(mutation.id)
        except LookupError:
            pass
        else:
            raise MutationError(f"entry {mutation.id!r} already exists")
        if mutation.options is None:
            raise MutationError("create mutation must carry options")
        loader.create({**mutation.options, "id": mutation.id})
    elif mutation.op == "update":
        entry = _resolve(loader, mutation.id)
        if mutation.options is None:
            raise MutationError("update mutation must carry options")
        # The kind's plugin is the entry's admission gate and stays bound to the live fiber, so an
        # update that renamed the kind would validate under the old one; a kind change is remove + create.
        kind = mutation.options.get("name")
        if kind is not None and str(kind) != str(entry.options.get("name")):
            raise MutationError(
                f"update {mutation.id!r} cannot change the entry's kind from {entry.options.get('name')!r} "
                f"to {kind!r}; remove the entry and create it under the new kind"
            )
        loader.update(mutation.id, dict(mutation.options))
    else:
        _resolve(loader, mutation.id)
        loader.remove(mutation.id)


def admit_mutations(
    entries: Sequence[Mapping[str, Any]], mutations: Sequence[Mutation], descriptor: AdapterDescriptor
) -> tuple[list[EntryOptions], str | None]:
    """Apply ``mutations`` over a fresh admission loader and try the render: the new entries and None, or the old entries and the refusal.

    The one admission every proposal meets, the method's in ``prepare_step``
    and an agent's at the proposals route: each mutation under the rules of
    ``_apply_mutation``, a FAILED fiber refused with its error, a kind the
    adapter renders no path for refused naming the kind, and a render error
    refused with its message."""
    previous = [dict(entry) for entry in entries]
    loader = Loader(Context(), NODE_KINDS.get)
    loader.root.update([dict(entry) for entry in entries])
    try:
        for mutation in mutations:
            _apply_mutation(loader, mutation)
            if mutation.op == "remove":
                continue
            error = _load_error(loader, mutation.id, descriptor)
            if error is not None:
                raise MutationError(f"mutation {mutation.op} {mutation.id!r} rejected: {error}")
        admitted = _loader_entries(loader)
        render_composition(_nodes_from(admitted), descriptor)
        refusal = _native_refusal(admitted, descriptor)
        if refusal is not None:
            raise MutationError(refusal)
    except (MutationError, RenderError) as error:
        return previous, str(error)
    return admitted, None


def _native_refusal(entries: Sequence[Mapping[str, Any]], descriptor: AdapterDescriptor) -> str | None:
    """The first config entry a native host would refuse at boot, None when the tree is not native or all pass.

    The boot rules of the config plugin (target ``models`` only, no pinned
    binding field, a positive window) are checked here too, so a tree the
    serve process would roll back never wins a gate."""
    if descriptor.tree_path is None:
        return None
    from reef.harness.runners.native.plugins import check_native_config

    for entry in entries:
        if entry.get("name") != "config" or entry.get("disabled"):
            continue
        try:
            check_native_config(entry.get("config") or {})
        except ValueError as error:
            return f"entry {entry.get('id')!r} rejected: {error}"
    return None
