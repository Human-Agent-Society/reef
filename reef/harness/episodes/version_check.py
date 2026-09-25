"""The shipped update notice: a seedable entry per adapter.

The notice is composition, not runtime code: seeding the entry makes it part
of the tree the gate measures, and every pulled or installed copy carries it.
Adapters own the notice implementation as extension source or a JSON config
asset; this module loads it into the corresponding composition entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reef.harness.adapters.descriptor import DescriptorError

VERSION_CHECK_ENTRY_ID = "reef-version-check"

_ASSETS = {
    "pi": Path(__file__).parents[1] / "adapters" / "pi" / "version_check.ts",
}

_CONFIG_ASSETS = {
    "claude": Path(__file__).parents[1] / "adapters" / "claude" / "version_check.json",
}


def version_check_entry(adapter: str) -> dict[str, Any]:
    """The seed entry options for the adapter's shipped update notice."""
    asset = _ASSETS.get(adapter)
    if asset is not None:
        return {
            "id": VERSION_CHECK_ENTRY_ID,
            "name": "code_extension",
            "config": {"name": VERSION_CHECK_ENTRY_ID, "code": asset.read_text(encoding="utf-8")},
        }
    config_asset = _CONFIG_ASSETS.get(adapter)
    if config_asset is None:
        raise DescriptorError(f"adapter {adapter!r} ships no version check extension")
    return {
        "id": VERSION_CHECK_ENTRY_ID,
        "name": "config",
        "config": json.loads(config_asset.read_text(encoding="utf-8")),
    }


__all__ = ["VERSION_CHECK_ENTRY_ID", "version_check_entry"]
