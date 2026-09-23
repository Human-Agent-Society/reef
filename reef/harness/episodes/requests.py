"""The shipped harness requests entries: two seedable entries per adapter.

Like the update notice, they are composition, not runtime code:
``evolution.requests: true`` appends the entry carrying the ``/reefine``
command, which files the person's request through native manual training,
and a ``skill`` entry carrying the adapter's API reference, which the
service proposer reads before it writes a change. For pi the command is a
``code_extension`` (``requests.ts``); for Claude Code it is an
``agent_command`` (``reefine.md``, a command file that runs the wrapper).
Both ids are reef's own (``RESERVED_ENTRY_IDS``), so a proposal cannot
rewrite or remove them; the seed and a recovered state carry them as they
are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reef.harness.adapters.descriptor import DescriptorError

REQUESTS_ENTRY_ID = "reef-requests"
REQUESTS_SKILL_ID = "reef-pi-extension-api"

_ADAPTERS = Path(__file__).parents[1] / "adapters"
#: Per adapter: the file of the ``/reefine`` command and the skill body it ships (a test points these at its own).
_ASSETS: dict[str, tuple[Path, Path]] = {
    "pi": (_ADAPTERS / "pi" / "requests.ts", _ADAPTERS / "pi" / "pi_extension_api.md"),
    "claude": (_ADAPTERS / "claude" / "reefine.md", _ADAPTERS / "claude" / "harness_api.md"),
}
#: Per adapter: the command entry's kind and name, and the skill entry's id.
_SHAPES: dict[str, tuple[str, str, str]] = {
    "pi": ("code_extension", REQUESTS_ENTRY_ID, REQUESTS_SKILL_ID),
    "claude": ("agent_command", "reefine", "reef-claude-harness-api"),
}


def _read(asset: Path, adapter: str) -> str:
    try:
        return asset.read_text(encoding="utf-8")
    except OSError as exc:
        raise DescriptorError(f"adapter {adapter!r} requests asset {asset.name} cannot be read: {exc}") from exc


def requests_skill_id(adapter: str) -> str | None:
    """The id of the API reference skill the adapter ships beside its ``/reefine`` command; None without one."""
    shape = _SHAPES.get(adapter)
    return None if shape is None else shape[2]


def request_entries(adapter: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The seed entry options for the adapter's shipped harness requests command and its skill, in seed order."""
    assets = _ASSETS.get(adapter)
    shape = _SHAPES.get(adapter)
    if assets is None or shape is None:
        raise DescriptorError(f"adapter {adapter!r} ships no requests extension")
    command, skill = assets
    kind, name, skill_id = shape
    body = "code" if kind == "code_extension" else "text"
    return (
        {
            "id": REQUESTS_ENTRY_ID,
            "name": kind,
            "config": {"name": name, body: _read(command, adapter)},
        },
        {
            "id": skill_id,
            "name": "skill",
            "config": {"name": skill_id, "text": _read(skill, adapter)},
        },
    )


__all__ = ["REQUESTS_ENTRY_ID", "REQUESTS_SKILL_ID", "request_entries", "requests_skill_id"]
