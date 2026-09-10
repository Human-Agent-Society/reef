"""What a release needs from the person: the ``requires`` items a training request carries.

A request posted to ``POST /reef/train`` may name what its change needs
from the person's machine, as ``{name, kind, check}`` items; the proposer
may add items its extension needs, and the commit that answered the request
carries the merged list under ``training_request.requires``. A release's
items are per step, so the manifest, the install script, ``reef-<adapter>
setup`` and the update notice read the union over the release's chain with
:func:`required_by`, the one rule for all four. Nothing here runs a check:
the checks run on the person's machine, after the person read them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

#: What a ``requires`` item asks of the person: an OS permission to grant, a variable to set, a service to connect.
REQUIRE_KINDS = ("permission", "env", "service")
#: A release names a handful of things to set up; a longer list is a request that should be split.
MAX_REQUIRES = 8
#: An item's name is shown, checked off and matched by name, so it keeps to the entry name pattern.
_REQUIRE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: An env item names a variable the wrapper and the extension read from the environment, so it is an identifier.
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_requires(value: object, *, limit: int | None = MAX_REQUIRES) -> list[dict[str, Any]]:
    """The ``requires`` list of a request as ``{name, kind, check?}`` records; a ValueError names the first bad item.

    ``value`` is the list as the route or a proposer gave it: at most
    ``limit`` objects (``MAX_REQUIRES`` for one request; ``None`` for a
    chain union, which the cap never bounds), each with a ``name`` matching
    the entry name pattern, a ``kind`` from ``REQUIRE_KINDS`` and an
    optional non empty ``check``; for kind ``env`` the variable named (the
    check, else the name) is a shell identifier. Unknown keys are dropped;
    ``None`` means no items."""
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("requires must be a list of {name, kind, check} objects")
    if limit is not None and len(value) > limit:
        raise ValueError(f"requires must have at most {limit} items")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"requires[{index}] must be an object with a name and a kind")
        name, kind, check = item.get("name"), item.get("kind"), item.get("check")
        if not isinstance(name, str) or not _REQUIRE_NAME.fullmatch(name):
            raise ValueError(f"requires[{index}].name must be a non-empty string matching {_REQUIRE_NAME.pattern}")
        if kind not in REQUIRE_KINDS:
            raise ValueError(f"requires[{index}].kind must be one of {REQUIRE_KINDS}")
        if check is not None and (not isinstance(check, str) or not check.strip()):
            raise ValueError(f"requires[{index}].check must be a non-empty string when present")
        if kind == "env" and not _ENV_NAME.fullmatch(name if check is None else check):
            field = "name" if check is None else "check"
            raise ValueError(
                f"requires[{index}].{field} must be a variable name matching {_ENV_NAME.pattern} for kind env"
            )
        parsed: dict[str, Any] = {"name": name, "kind": kind}
        if check is not None:
            parsed["check"] = check
        items.append(parsed)
    return items


def merge_requires(base: Sequence[Mapping[str, Any]], added: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``base`` then every ``added`` item whose name is new: the person's items stand, a proposer only adds."""
    merged = [dict(item) for item in base]
    names = {str(item.get("name")) for item in merged}
    for item in added:
        if str(item.get("name")) not in names:
            merged.append(dict(item))
            names.add(str(item.get("name")))
    return merged


def _row_requires(metrics: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The ``training_request.requires`` items of one release's gate metrics; empty when the step read no request."""
    request = metrics.get("training_request") if isinstance(metrics, Mapping) else None
    requires = request.get("requires") if isinstance(request, Mapping) else None
    if not isinstance(requires, Sequence) or isinstance(requires, str):
        return []
    return [dict(item) for item in requires if isinstance(item, Mapping) and isinstance(item.get("name"), str)]


def _chain(rows: Sequence[Mapping[str, Any]], release_id: str | None) -> list[Mapping[str, Any]]:
    """The content chain of ``release_id`` through the catalog ``rows``, newest first.

    ``rows`` come oldest first, as ``GET /reef/harness/releases`` lists
    them, and the row that published an id is its oldest row; a promote or
    rollback row continues at its target, any other at its parent."""
    published: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("release_id"), str):
            published.setdefault(row["release_id"], row)
    chain: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    current = release_id
    while isinstance(current, str) and current in published and current not in seen:
        seen.add(current)
        chain.append(published[current])
        current = published[current].get("rollback_target_release_id") or published[current].get("parent_release_id")
    return chain


def required_by(rows: Sequence[Mapping[str, Any]], release_id: str | None) -> list[dict[str, Any]]:
    """What a release needs from the person: every ``training_request.requires`` item over its chain, by name.

    A request's items are per release, so a later release whose request
    named nothing still carries the extension an earlier one added. The
    walk starts at ``release_id`` and follows its chain through the
    catalog ``rows``. Names keep the order of their oldest definition and
    the newest definition of a name wins. The one rule for the manifest,
    the install script, ``reef-<adapter> setup`` and the update notice."""
    merged: dict[str, dict[str, Any]] = {}
    for row in reversed(_chain(rows, release_id)):
        for item in _row_requires(row.get("metrics")):
            merged[item["name"]] = item
    return list(merged.values())


def ancestor_requiring_nothing(rows: Sequence[Mapping[str, Any]], release_id: str | None) -> str | None:
    """The newest release before ``release_id`` in its chain that requires nothing over its own chain.

    The one a machine with nothing set up installs: every later release
    in the chain needs an item, and the local release file records no completed setup.
    The creation row always qualifies; ``None`` when ``release_id`` has no
    ancestor in ``rows``."""
    ancestors = list(reversed(_chain(rows, release_id)))[:-1]
    # Oldest first, the union stays empty until the first row that named an item.
    latest = None
    for row in ancestors:
        if _row_requires(row.get("metrics")):
            break
        latest = str(row["release_id"])
    return latest


__all__ = [
    "MAX_REQUIRES",
    "REQUIRE_KINDS",
    "ancestor_requiring_nothing",
    "merge_requires",
    "parse_requires",
    "required_by",
]
