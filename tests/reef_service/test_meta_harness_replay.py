"""Phase 0 of the reproduction: Reef's selection matches upstream's, replayed.

Skipped unless ``METAHARNESS_DIR`` names an upstream ``terminal_bench_2``
checkout, because it imports and drives upstream's own ``update_frontier``
rather than a description of it. Upstream's module-scope imports have to
resolve too, so run it under an interpreter that has them::

    METAHARNESS_DIR=/path/to/terminal_bench_2 \
      uv run --extra dev --with python-dotenv python -m pytest \
      tests/reef_service/test_meta_harness_replay.py

This is where the reproduction claim lives. A live run cannot carry it: at any
affordable scale the two arms can select different candidates from sampling
noise, and nothing distinguishes that from an implementation difference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from recipes.meta_harness.examples.terminal_bench.replay import Step, compare

CHECKOUT = os.environ.get("METAHARNESS_DIR", "")

pytestmark = pytest.mark.skipif(
    not (CHECKOUT and Path(CHECKOUT, "meta_harness.py").is_file()),
    reason="METAHARNESS_DIR must name an upstream terminal_bench_2 checkout",
)


@pytest.fixture(autouse=True)
def _upstream_on_path():
    import sys

    sys.path.insert(0, CHECKOUT)
    try:
        # Upstream imports its own third-party deps at module scope. Without them
        # every test below fails on the import, which reads like the selection
        # rules diverged rather than like a missing package. Skip instead.
        try:
            import meta_harness  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"upstream meta_harness is not importable here: {exc}")
        yield
    finally:
        sys.path.remove(CHECKOUT)


@pytest.mark.unit
def test_a_strict_improvement_is_selected_by_both(tmp_path: Path) -> None:
    steps = [Step(candidate=(1.0, 1.0), current=(0.0, 0.0))]
    assert compare(steps, baseline=(0.0, 0.0), workdir=tmp_path) == []


@pytest.mark.unit
def test_a_regression_is_rejected_by_both(tmp_path: Path) -> None:
    steps = [Step(candidate=(0.0, 0.0), current=(1.0, 1.0))]
    assert compare(steps, baseline=(1.0, 1.0), workdir=tmp_path) == []


@pytest.mark.unit
def test_a_tie_is_rejected_by_both(tmp_path: Path) -> None:
    # Both rules are strict: `avg > best` upstream, `mean > mean` in Reef.
    steps = [Step(candidate=(1.0,), current=(1.0,))]
    assert compare(steps, baseline=(1.0,), workdir=tmp_path) == []


@pytest.mark.unit
def test_the_frontier_does_not_fall_back_after_a_win(tmp_path: Path) -> None:
    # A high-water mark: once 1.0 is admitted, 0.5 loses even though it beat
    # the incumbent measured in its own batch.
    steps = [
        Step(candidate=(1.0, 1.0), current=(0.0, 0.0)),
        Step(candidate=(1.0, 0.0), current=(0.0, 0.0)),
    ]
    assert compare(steps, baseline=(0.0, 0.0), workdir=tmp_path) == []


@pytest.mark.unit
def test_the_baseline_phase_is_what_stops_the_first_candidate_winning_free(tmp_path: Path) -> None:
    # Upstream seeds the frontier from its baselines before any candidate
    # ("Always seed frontier from baselines if results exist"). Reef's
    # equivalent is the seed's own evaluation. With both seeded at 1.0 a
    # weaker first candidate loses on both sides.
    steps = [Step(candidate=(0.0,), current=(1.0,))]
    assert compare(steps, baseline=(1.0,), workdir=tmp_path) == []
