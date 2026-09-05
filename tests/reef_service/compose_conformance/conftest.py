"""Fixtures for the mirrored loader specs."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from .utils import Harness


@pytest.fixture(scope="class")
def harness() -> Iterator[Harness]:
    """One ``describe`` block's shared state; cases run in definition order."""
    built = Harness()
    try:
        yield built
    finally:
        built.close()
