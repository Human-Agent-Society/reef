"""The cookbook's tests see the Terminal-Bench campaign's terminus adapter as installed.

Installing the campaign registers its adapter on the ``reef.harness_adapters``
entry point group. These tests put the campaign's ``reef_terminus`` package on
the import path, with a distribution record that declares that entry point, so
``get_adapter("terminus")`` resolves as it does after an install. Only that
package goes on the path: the campaign's ``harness`` package would shadow the
other examples' packages of the same name.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2] / "recipes" / "meta_harness" / "examples" / "terminal_bench"


def register_terminus() -> None:
    """Put ``reef_terminus`` and its distribution record on the import path."""
    root = Path(tempfile.mkdtemp(prefix="reef-terminus-campaign-"))
    (root / "reef_terminus").symlink_to(CAMPAIGN / "reef_terminus", target_is_directory=True)
    distribution = root / "reef_meta_harness_terminal_bench-0.1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: reef-meta-harness-terminal-bench\nVersion: 0.1.0\n")
    (distribution / "entry_points.txt").write_text("[reef.harness_adapters]\nterminus = reef_terminus:descriptor\n")
    sys.path.insert(0, str(root))


register_terminus()
