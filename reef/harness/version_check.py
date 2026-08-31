"""The shipped update notice: a seedable ``code_extension`` entry per adapter.

The prompt is composition, not runtime code: seeding the entry makes it part
of the tree the gate measures, and every pulled or installed copy carries it.
It offers to run the update or skip in interactive mode, prints the instructions
in headless mode, and stays silent under ``PI_OFFLINE``, so hermetic benchmark
episodes make no network calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reef.harness.descriptor import DescriptorError

VERSION_CHECK_ENTRY_ID = "reef-version-check"

_ASSETS = {
    "pi": Path(__file__).parent / "adapters" / "pi" / "version_check.ts",
}


def version_check_entry(adapter: str) -> dict[str, Any]:
    """The seed entry options for the adapter's shipped update notice."""
    asset = _ASSETS.get(adapter)
    if asset is None:
        raise DescriptorError(f"adapter {adapter!r} ships no version check extension")
    return {
        "id": VERSION_CHECK_ENTRY_ID,
        "name": "code_extension",
        "config": {"name": VERSION_CHECK_ENTRY_ID, "code": asset.read_text(encoding="utf-8")},
    }


__all__ = ["VERSION_CHECK_ENTRY_ID", "version_check_entry"]
