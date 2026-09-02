"""Tests for the bare-``python`` swap in the orchestrator."""

from __future__ import annotations

import sys

from reef.service.deploy.orchestrator import _resolve_python


def test_bare_python_is_swapped_to_sys_executable() -> None:
    assert _resolve_python(["python", "-m", "reef.service"]) == [sys.executable, "-m", "reef.service"]


def test_bare_python3_is_swapped_to_sys_executable() -> None:
    assert _resolve_python(["python3", "-m", "reef.service"]) == [sys.executable, "-m", "reef.service"]


def test_explicit_absolute_python_is_left_alone() -> None:
    argv = ["/usr/bin/python3.11", "-m", "reef.service"]
    assert _resolve_python(argv) == argv


def test_versioned_alias_on_path_is_left_alone() -> None:
    argv = ["python3.12", "-m", "reef.service"]
    assert _resolve_python(argv) == argv


def test_non_python_command_is_left_alone() -> None:
    argv = ["curl", "-sf", "http://localhost:8900/healthz"]
    assert _resolve_python(argv) == argv


def test_empty_argv_is_left_alone() -> None:
    assert _resolve_python([]) == []
