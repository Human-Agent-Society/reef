"""Declarative loader: a config tree reconciled onto live plugin fibers.

An entry is one desired plugin instantiation, described by a plain options
dict (id, name, config, group, disabled, inject, intercept, isolate); a
group is an entry whose config is a list of child entries; the tree indexes
every entry by id. Reconciliation diffs desired options against the live
state and applies the minimal mutation: a new or re-enabled entry loads a
fiber, a removed or disabled one disposes it (each load is an invertible
effect of the loader's fiber, so tree teardown is plain LIFO replay), and a
changed one patches its context and reconfigures through ``fiber.update``,
restarting only the affected subtree. The live state also writes back: a
config update or self-disposal of a tracked fiber lands in its entry's
options, keeping the desired tree the single source of truth.

Plugins are named: the loader maps an entry's ``name`` to a plugin through
the resolver callable given at construction, and the tree lives in memory.
Module resolution over the filesystem, file watching, and HMR are permanent
omissions of this port (see UPSTREAM.md); the resolver is synchronous, so
entry initialization is too.

Entry options ``isolate`` (name -> True for a private realm, or a string
label for a shared one) and ``intercept`` (name -> config overlay) apply
the core's realm and interception machinery per entry.
"""

from __future__ import annotations

import asyncio
import random
from collections import ChainMap
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from .context import Context, RealmKey
from .fiber import Fiber, FiberState
from .registry import Plugin, _resolve_inject

EntryOptions = dict[str, Any]
"""One entry's desired state; see the module docstring for the keys."""

_ENTRY_KEY = "_loader_entry"
"""Override key marking a context as belonging to an entry (Entry.key)."""


def entry_of(ctx: Context) -> Entry | None:
    """The nearest entry owning ``ctx``, through the override chain."""
    node: Context | None = ctx
    while node is not None:
        overrides = node.__dict__.get("_overrides")
        if overrides and _ENTRY_KEY in overrides:
            entry: Entry = overrides[_ENTRY_KEY]
            return entry
        node = node.__dict__.get("_parent")
    return None


class Entry:
    """One desired plugin instantiation and its live fiber, if any."""

    def __init__(self, loader: Loader, parent: EntryGroup) -> None:
        self.loader = loader
        self.fiber: Fiber | None = None
        self.parent = parent
        self.options: EntryOptions = {}
        self.subgroup: EntryGroup | None = None
        self.subtree: EntryTree | None = None
        self.realm: LocalRealm | None = None
        self.ctx = loader.ctx.extend(**{_ENTRY_KEY: self})
        self.ctx.emit("loader/entry-init", self)

    @property
    def id(self) -> str:
        """The tree-qualified id: subtree entries prefix their owner's id."""
        id_ = str(self.options.get("id", ""))
        owner = entry_of(self.parent.tree.ctx)
        if owner is not None:
            return owner.id + EntryTree.sep + id_
        return id_

    @property
    def disabled(self) -> bool:
        """Disabled here or by any ancestor entry; groups are always enabled."""
        if self.options.get("group"):
            return False
        entry: Entry | None = self
        while entry is not None:
            if entry.options.get("disabled"):
                return True
            entry = entry_of(entry.parent.ctx)
        return False

    def _resolve_config(self) -> Any:
        # The reference interpolates JS expressions into the config here; a
        # permanent omission, so the options hold the config literally.
        return self.options.get("config")

    def _patch_context(self, diff: Sequence[str]) -> None:
        """Re-attach the entry context under its (possibly new) group and
        push a changed config into the live fiber (entry.ts:84-92).

        The 'loader/patch-context' waterfall lets loader plugins adjust the
        context first; the realm plugin swaps isolate tables through it.
        """

        def inner(*_: Any) -> None:
            parent_ctx = self.parent.ctx
            object.__setattr__(self.ctx, "_parent", parent_ctx)
            object.__setattr__(self.ctx, "_isolate", ChainMap(self.ctx._isolate.maps[0], *parent_ctx._isolate.maps))
            object.__setattr__(
                self.ctx, "_intercept", ChainMap(self.ctx._intercept.maps[0], *parent_ctx._intercept.maps)
            )
            if (
                self.fiber is not None
                and self.fiber.uid is not None
                and ("config" in diff or self.options.get("group"))
            ):
                self.fiber.update(self._resolve_config(), True)

        self.ctx.waterfall("loader/patch-context", self, inner)

    def refresh(self) -> None:
        """Load if this entry should be live but is not."""
        if self.fiber is not None or self.disabled:
            return
        self.init()

    def update(self, options: EntryOptions, create: bool = False, force: bool = False) -> None:
        """Reconcile new options: dispose when disabled, patch and
        reconfigure when live and changed, load when not yet live.

        In non-create mode a None value deletes its key; ``create`` replaces
        the options wholesale.
        """
        legacy = dict(self.options)
        if create:
            self.options = dict(options)
        else:
            for key, value in options.items():
                if value is None:
                    self.options.pop(key, None)
                else:
                    self.options[key] = value
        if self.disabled:
            if self.fiber is not None:
                self.fiber.dispose()
            return
        if self.fiber is not None and self.fiber.uid is not None:
            diff = [key for key in {**self.options, **legacy} if self.options.get(key) != legacy.get(key)]
            if not diff and not force:
                return
            self.ctx.emit("loader/partial-dispose", self, legacy, True)
            self._patch_context(diff)
        else:
            self.init()

    def init(self) -> None:
        """Resolve the plugin by name and instantiate it under this entry.

        A resolution failure is logged and leaves the entry without a fiber,
        as an import failure does upstream; the entry loads on a later
        update once the resolver knows the name.
        """
        try:
            plugin = self.loader.resolve_plugin(self.options)
        except Exception:
            self.ctx.logger.exception('loader cannot resolve plugin "%s"', self.options.get("name"))
            return
        if plugin is None:
            self.ctx.logger.error('loader cannot resolve plugin "%s"', self.options.get("name"))
            return
        self._patch_context([])
        self.loader.show_log(self, "apply")
        self.fiber = self.ctx.registry.plugin(plugin, self._resolve_config(), ctx=self.ctx)

        def renotify(*_: Any) -> None:
            # Entry work settled: let 'await'-intercepted loader dependents
            # re-check through the check predicate (entry.ts:152-156).
            if not self.loader.get_tasks():
                self.ctx.reflect.notify(["loader"], ctx=self.ctx)

        inertia = self.fiber._inertia
        if inertia is None:
            renotify()
        else:
            inertia.add_done_callback(renotify)


class EntryGroup:
    """A list of sibling entries reconciled together."""

    def __init__(self, ctx: Context, tree: EntryTree) -> None:
        self.ctx = ctx
        self.tree = tree
        self.data: list[EntryOptions] = []
        entry = entry_of(ctx)
        if entry is not None:
            entry.subgroup = self

    def create(self, options: EntryOptions) -> str:
        id_ = self.tree.ensure_id(options)
        entry = self.tree.store.get(id_)
        if entry is None:
            entry = self.tree.store[id_] = Entry(self.ctx.loader, self)
        # An entry may be moved from another group; update the parent reference.
        entry.parent = self
        entry.update(options, create=True, force=True)
        return entry.id

    def unlink(self, options: EntryOptions) -> None:
        for index, existing in enumerate(self.data):
            if existing.get("id") == options.get("id"):
                self.data.pop(index)
                return

    def remove(self, id_: str, is_dispose: bool = False) -> None:
        entry = self.tree.store.get(id_)
        if entry is None:
            return
        if entry.fiber is not None:
            entry.fiber.dispose()
        if not is_dispose:
            self.unlink(entry.options)
        self.tree.store.pop(id_)
        self.ctx.emit("loader/partial-dispose", entry, entry.options, False)

    def update(self, config: Sequence[EntryOptions]) -> None:
        """Reconcile the desired child list: create or update every entry
        still present, remove every entry that is not."""
        old = self.data
        self.data = list(config)
        old_map = {str(options["id"]): options for options in old if options.get("id")}
        new_map = {self.tree.ensure_id(options): options for options in self.data}
        for id_ in {**old_map, **new_map}:
            if id_ in new_map:
                try:
                    self.create(new_map[id_])
                except Exception:
                    self.ctx.logger.exception('loader failed to reconcile entry "%s"', id_)
            else:
                self.remove(id_)

    def stop(self) -> None:
        for options in list(self.data):
            self.remove(str(options.get("id")), True)


class Group(EntryGroup):
    """The group plugin: an entry whose config is its children's options.

    Reconfiguring a group entry reconciles the children in place; the
    non-global 'internal/update' hook swallows the restart the update
    waterfall would otherwise run, so the group fiber itself stays up.
    """

    def __init__(self, ctx: Context, config: Sequence[EntryOptions] | None) -> None:
        owner = entry_of(ctx)
        if owner is None:
            raise RuntimeError("the group plugin only loads under a loader entry")
        super().__init__(ctx, owner.parent.tree)
        self.config = list(config or [])
        ctx.on("internal/update", self._on_update)

    def _on_update(
        self, fiber: Fiber, config: Sequence[EntryOptions], no_save: bool, next_: Callable[[], Any]
    ) -> None:
        # Deliberately do not call through: reconfiguring a group does not restart it.
        self.update(config or [])

    def __compose_init__(self) -> Any:
        yield self.stop
        self.update(self.config)


class EntryTree:
    """The id-indexed tree of entries under one root group."""

    sep = ":"

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx.extend()
        self.enable_logs = False
        self.store: dict[str, Entry] = {}
        self.root = EntryGroup(self.ctx, self)
        entry = entry_of(self.ctx)
        if entry is not None:
            entry.subtree = self

    def entries(self) -> Iterator[Entry]:
        for entry in list(self.store.values()):
            yield entry
            if entry.subtree is not None:
                yield from entry.subtree.entries()

    def get_tasks(self) -> list[Any]:
        """The in-flight async work of every entry fiber."""
        return [
            entry.fiber._inertia
            for entry in self.entries()
            if entry.fiber is not None and entry.fiber._inertia is not None
        ]

    async def wait(self) -> None:
        """Settle every in-flight task, including the ones they spawn."""
        while True:
            tasks = self.get_tasks()
            if not tasks:
                # One extra tick so completion callbacks (the loader's own
                # re-notify among them) land before the caller proceeds.
                await asyncio.sleep(0)
                if not self.get_tasks():
                    return
                continue
            await asyncio.gather(*tasks, return_exceptions=True)

    def ensure_id(self, options: EntryOptions) -> str:
        if not options.get("id"):
            id_ = f"{random.randrange(16**8):08x}"
            while id_ in self.store:
                id_ = f"{random.randrange(16**8):08x}"
            options["id"] = id_
        return str(options["id"])

    def resolve(self, id_: str) -> Entry:
        parts = id_.split(EntryTree.sep)
        tree: EntryTree = self
        final = parts.pop()
        for part in parts:
            subtree = tree.store[part].subtree if part in tree.store else None
            if subtree is None:
                raise LookupError(f"cannot resolve entry {id_}")
            tree = subtree
        entry = tree.store.get(final)
        if entry is None:
            raise LookupError(f"cannot resolve entry {id_}")
        return entry

    def resolve_group(self, id_: str | None) -> EntryGroup:
        if not id_:
            return self.root
        entry = self.resolve(id_)
        if entry.subgroup is None:
            raise LookupError(f"entry {id_} is not a group")
        return entry.subgroup

    def create(self, options: EntryOptions, parent: str | None = None, position: int | None = None) -> str:
        group = self.resolve_group(parent)
        group.data.insert(position if position is not None else len(group.data), options)
        group.tree.write()
        return group.create(options)

    def remove(self, id_: str) -> None:
        entry = self.resolve(id_)
        entry.parent.remove(id_)
        entry.parent.tree.write()

    def update(
        self,
        id_: str,
        options: EntryOptions,
        parent: str | None = None,
        position: int | None = None,
        move: bool = False,
    ) -> None:
        """Reconcile one entry's options; with ``move``, first relocate it
        under the group ``parent`` at ``position``."""
        entry = self.resolve(id_)
        source = entry.parent
        if move:
            target = self.resolve_group(parent)
            source.unlink(entry.options)
            target.data.insert(position if position is not None else len(target.data), entry.options)
            target.tree.write()
            entry.parent = target
        source.tree.write()
        entry.update(options, create=False, force=True)

    def write(self) -> None:
        """The tree is in-memory; there is no config file to write back to."""


class Loader(EntryTree):
    """The loader service: the root entry tree plus the tracking hooks.

    ``resolver`` maps an entry's plugin name to a plugin object (or None);
    ``builtins`` are consulted first and pre-register nothing, since group
    entries route to :class:`Group` by their ``group`` flag alone.
    """

    def __init__(self, ctx: Context, resolver: Callable[[str], Plugin | None]) -> None:
        super().__init__(ctx)
        self.name = "loader"
        self.resolver = resolver
        self.builtins: dict[str, Plugin] = {}
        ctx.provide("loader", self, self._check)
        ctx.on("internal/update", self._save_config, prepend=True, glob=True)
        ctx.on("internal/update", self._log_reload, glob=True)
        ctx.on("internal/plugin", self._track_fiber)
        ctx.plugin(_entry_isolate)

    def resolve_plugin(self, options: EntryOptions) -> Plugin | None:
        if options.get("group"):
            return Group
        name = str(options.get("name"))
        builtin = self.builtins.get(name)
        if builtin is not None:
            return builtin
        return self.resolver(name)

    def _check(self, ctx: Context) -> bool:
        """Check predicate of the 'loader' service: a dependent that
        intercepts ``{"await": True}`` only sees the loader while no entry
        work is in flight (index.ts:133-137)."""
        config: dict[str, Any] = {}
        for layer in reversed(ctx._intercept.maps):
            value = layer.get("loader")
            if value:
                config.update(value)
        return not (config.get("await") and self.get_tasks())

    def show_log(self, entry: Entry, kind: str) -> None:
        if entry.options.get("group") or not entry.parent.tree.enable_logs:
            return
        self.ctx.logger.info("%s plugin %s", kind, entry.options.get("name"))

    def locate(self, fiber: Fiber | None = None) -> str | None:
        """The id of the entry owning ``fiber`` (default: the calling scope)."""
        node = fiber if fiber is not None else self.ctx.fiber
        while True:
            entry = node.entry
            if entry is not None:
                return str(entry.id)
            parent = node.parent.fiber
            if parent is node:
                return None
            node = parent

    def _save_config(self, fiber: Fiber, config: Any, no_save: bool, next_: Callable[[], Any]) -> Any:
        """Global 'internal/update' middleware: a tracked fiber's reconfigure
        writes through to its entry's options (index.ts:74-80)."""
        entry = fiber.entry
        if entry is None or no_save or fiber.parent.fiber.entry is entry:
            return next_()
        # Unparse before write-back (index.ts:76-77 Config.simplify): storing
        # the validated model itself would diff against the raw dict on the
        # next reconcile and spuriously reload the plugin.
        schema = fiber.runtime.Config if fiber.runtime is not None else None
        simplify = getattr(schema, "simplify", None) if schema is not None else None
        if callable(simplify):
            config = simplify(config)
        elif hasattr(config, "model_dump"):
            config = config.model_dump()
        entry.options["config"] = config
        entry.parent.tree.write()
        return next_()

    def _log_reload(self, fiber: Fiber, config: Any, no_save: bool, next_: Callable[[], Any]) -> Any:
        entry = fiber.entry
        if entry is not None and fiber.parent.fiber.entry is not entry:
            self.show_log(entry, "reload")
        return next_()

    def _track_fiber(self, fiber: Fiber) -> None:
        """'internal/plugin' hook: adopt created fibers into their entry, and
        write a tracked fiber's self-disposal back as disabled (index.ts:88-124)."""
        parent_entry = entry_of(fiber.parent)
        if parent_entry is not None and fiber.entry is None:
            fiber.entry = parent_entry
            fiber.inject.update(_resolve_inject(parent_entry.options.get("inject")))

        # Only ctx.fiber.dispose() from inside the plugin should mark the
        # entry disabled; filter out every other way a fiber retires.
        if fiber.uid is not None:
            return  # case 1: fiber is created, not retired
        entry = fiber.entry
        if entry is None:
            return  # case 2: fiber is not tracked by the loader
        if fiber.parent.fiber.entry is entry:
            return  # case 3: fiber is a child plugin under the entry
        runtime = fiber.runtime
        if runtime is None or not self.ctx.registry.has(runtime.callback):
            return  # case 4: fiber is disposed on behalf of plugin deletion
        tree_fiber = entry.parent.tree.ctx.fiber
        if tree_fiber.uid is None or tree_fiber.state in (FiberState.UNLOADING, FiberState.DISPOSED):
            # case 5: the entry's tree is being disposed. A retired owning
            # fiber covers the loader-as-plugin shape; the state check covers
            # a root-attached tree, whose fiber restarts instead of retiring.
            return
        self.show_log(entry, "unload")
        if entry.disabled:
            return  # case 6: fiber is disposed by loader behavior
        entry.options["disabled"] = True
        entry.parent.tree.write()


class Realm:
    """A keyspace of realm keys for isolated service names (isolate.ts:25-45)."""

    def __init__(self) -> None:
        self._store: dict[str, RealmKey] = {}

    @property
    def suffix(self) -> str:
        raise NotImplementedError

    def access(self, name: str, create: bool = False) -> RealmKey:
        if create:
            return self._store.setdefault(name, RealmKey(name + self.suffix))
        existing = self._store.get(name)
        return existing if existing is not None else RealmKey(name + self.suffix)

    def delete(self, name: str) -> None:
        self._store.pop(name, None)

    @property
    def size(self) -> int:
        return len(self._store)


class LocalRealm(Realm):
    """A realm private to one entry (isolate: True)."""

    def __init__(self, entry: Entry) -> None:
        super().__init__()
        self.entry = entry

    @property
    def suffix(self) -> str:
        return "#" + str(self.entry.options.get("id"))


class GlobalRealm(Realm):
    """A realm shared by every entry naming the same label (isolate: label)."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    @property
    def suffix(self) -> str:
        return "@" + self.label


def _entry_isolate(ctx: Context, config: Any = None) -> None:
    """Loader plugin realizing entry isolate options over the core realms.

    On every context patch: rebuild the entry's own isolate layer from its
    options, transfer a binding whose provider moved with the entry to the
    new realm key, and notify every fiber seeing the name at either realm so
    the epoch machinery restarts exactly those whose resolution changed.
    (The reference distinguishes moved-with-provider through delimiter
    symbols, isolate.ts:99-141; the ancestry test below is this port's
    equivalent, and over-notifying is harmless because unchanged epochs do
    not restart.)
    """
    realms: dict[str, GlobalRealm] = {}

    def access(entry: Entry, name: str, create: bool = False) -> RealmKey | None:
        label = (entry.options.get("isolate") or {}).get(name)
        if not label:
            return None
        realm: Realm | None
        if label is True:
            if entry.realm is None:
                entry.realm = LocalRealm(entry)
            realm = entry.realm
        elif create:
            realm = realms.setdefault(label, GlobalRealm(label))
        else:
            realm = realms.get(label)
        return None if realm is None else realm.access(name, create)

    def within(fiber: Fiber, entry: Entry) -> bool:
        node = fiber.entry
        while node is not None:
            if node is entry:
                return True
            node = entry_of(node.parent.ctx)
        return False

    def on_patch(entry: Entry, next_: Callable[[], Any]) -> None:
        new_layer = {}
        for name in entry.options.get("isolate") or {}:
            key = access(entry, name, True)
            if key is not None:
                new_layer[name] = key
        old_layer = dict(entry.ctx._isolate.maps[0])
        changed = [name for name in {**old_layer, **new_layer} if old_layer.get(name) is not new_layer.get(name)]
        old_keys = {name: entry.ctx._isolate.get(name) for name in changed}
        top = entry.ctx._isolate.maps[0]
        top.clear()
        top.update(new_layer)
        intercept_top = entry.ctx._intercept.maps[0]
        intercept_top.clear()
        intercept_top.update(entry.options.get("intercept") or {})
        next_()
        if not changed:
            return
        new_keys = {name: entry.ctx._isolate.get(name) for name in changed}
        reflect = entry.ctx.reflect
        for name in changed:
            old_key, new_key = old_keys[name], new_keys[name]
            impl = reflect.store.get(old_key) if old_key is not None else None
            if impl is not None and new_key is not None and new_key not in reflect.store and within(impl.fiber, entry):
                reflect.store[new_key] = reflect.store.pop(old_key)  # type: ignore[arg-type]

        def moved(target: Context, name: str) -> bool:
            return target._isolate.get(name) in (old_keys.get(name), new_keys.get(name))

        reflect.notify(changed, ctx=entry.ctx, filter=moved)

    def on_partial_dispose(entry: Entry, legacy: EntryOptions, active: bool) -> None:
        # Realm garbage collection: drop a global realm's key when no entry
        # references its label for that name anymore.
        for name, label in (legacy.get("isolate") or {}).items():
            if label is True:
                continue
            if active and (entry.options.get("isolate") or {}).get(name) == label:
                continue
            realm = realms.get(label)
            if realm is None:
                continue
            if any((other.options.get("isolate") or {}).get(name) == realm.label for other in ctx.loader.entries()):
                continue
            realm.delete(name)
            if not realm.size:
                realms.pop(realm.label, None)

    ctx.on("loader/patch-context", on_patch)
    ctx.on("loader/partial-dispose", on_partial_dispose)
