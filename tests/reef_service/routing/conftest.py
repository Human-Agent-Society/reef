"""Fixtures shared by the managed routing tests in this directory."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.reef_service.routing.routing_fakes import ControllerCase, controller_case


@pytest.fixture
def case(tmp_path: Path) -> Iterator[ControllerCase]:
    with controller_case(tmp_path) as built:
        yield built
