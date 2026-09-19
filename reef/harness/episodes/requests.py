"""The shipped harness requests extension: two seedable entries per adapter.

Like the update notice, the extension is composition, not runtime code:
``evolution.requests: true`` appends a ``code_extension`` entry carrying
``requests.ts`` (the ``/reef-harness`` command, which files the person's
request through native manual training) and a
``skill`` entry carrying the pi extension API reference the service proposer
reads before it writes an extension. Both
ids are reef's own (``RESERVED_ENTRY_IDS``), so a proposal cannot rewrite or
remove them; the seed and a recovered state carry them as they are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reef.harness.adapters.descriptor import DescriptorError

REQUESTS_ENTRY_ID = "reef-requests"
REQUESTS_SKILL_ID = "reef-pi-extension-api"

_ADAPTERS = Path(__file__).parents[1] / "adapters"
#: Per adapter: the extension file and the skill body it ships.
_ASSETS = {
    "pi": (_ADAPTERS / "pi" / "requests.ts", _ADAPTERS / "pi" / "pi_extension_api.md"),
}


def _read(asset: Path, adapter: str) -> str:
    try:
        return asset.read_text(encoding="utf-8")
    except OSError as exc:
        raise DescriptorError(f"adapter {adapter!r} requests asset {asset.name} cannot be read: {exc}") from exc


def request_entries(adapter: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The seed entry options for the adapter's shipped harness requests extension and its skill, in seed order."""
    assets = _ASSETS.get(adapter)
    if assets is None:
        raise DescriptorError(f"adapter {adapter!r} ships no requests extension")
    extension, skill = assets
    return (
        {
            "id": REQUESTS_ENTRY_ID,
            "name": "code_extension",
            "config": {"name": REQUESTS_ENTRY_ID, "code": _read(extension, adapter)},
        },
        {
            "id": REQUESTS_SKILL_ID,
            "name": "skill",
            "config": {"name": REQUESTS_SKILL_ID, "text": _read(skill, adapter)},
        },
    )


__all__ = ["REQUESTS_ENTRY_ID", "REQUESTS_SKILL_ID", "request_entries"]
