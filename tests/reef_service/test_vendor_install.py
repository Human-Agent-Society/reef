"""Server-side vendor resolution of the harness binary: the pin gate, the
vendor command, and the prefix a client's install script shares with it.

Hermetic: the npm-kind tests shim ``npm`` on PATH and run the real
subprocess path, exactly like the install-script tests; the git-kind test
records the install steps instead of cloning and building a venv.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.vendor_install import (
    DEFAULT_PREFIX_ROOT,
    PIN_FILE,
    PREFIX_ENV,
    VendorInstallError,
    install_prefix,
    resolve_binary,
)
from reef.service.install_script import render_install_script
from reef.train.cordis_backend import CordisBackend, FailureManifest, Mutation
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

from .test_harness_recipe import MODEL, RULES, batch, evaluate, make_binary, run_backend_step

PI_PIN = "@earendil-works/pi-coding-agent@0.84.2"


def _write_executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _npm_shim(tmp_path: Path, log: Path, *, installs: str | None = None) -> Path:
    """An ``npm`` on PATH that logs its argv and optionally lands a fake binary."""
    body = f'#!/bin/sh\nprintf \'%s\\n\' "$@" >> "{log}"\n'
    if installs is not None:
        body += f'mkdir -p "$(dirname "{installs}")"\n'
        body += f'printf \'%s\\n\' "#!/bin/sh" "echo 0.84.2" > "{installs}"\nchmod +x "{installs}"\n'
    shim = tmp_path / "shim"
    _write_executable(shim / "npm", body)
    return shim


@pytest.fixture
def on_path(monkeypatch):
    """Put a shim directory first on PATH for the duration of one test."""

    def apply(shim: Path) -> None:
        monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")

    return apply


@pytest.mark.unit
def test_an_adapter_without_an_install_section_resolves_to_its_bare_binary_name() -> None:
    """Reef's native loop and terminus declare no vendor channel: PATH answers, as before."""
    descriptor = get_adapter("native")
    assert descriptor.install is None
    assert resolve_binary(descriptor) == descriptor.binary


@pytest.mark.unit
def test_the_prefix_default_is_the_root_the_served_install_script_bakes_in() -> None:
    """Server and client on one machine install the pin into one tree, not two."""
    descriptor = get_adapter("pi")
    script = render_install_script(descriptor=descriptor, files={}, release_id="v1", content_id="c1")
    prefix_line = next(line for line in script.splitlines() if line.startswith("PREFIX="))
    assert DEFAULT_PREFIX_ROOT.replace("~", "$HOME", 1) in prefix_line
    assert PREFIX_ENV in prefix_line
    assert install_prefix(descriptor) == Path(DEFAULT_PREFIX_ROOT).expanduser() / "pi"


@pytest.mark.unit
def test_the_prefix_environment_override_moves_the_root_for_every_adapter(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(PREFIX_ENV, str(tmp_path / "elsewhere"))
    assert install_prefix(get_adapter("pi")) == tmp_path / "elsewhere" / "pi"
    assert install_prefix(get_adapter("codex")) == tmp_path / "elsewhere" / "codex"


@pytest.mark.unit
def test_an_installed_pin_resolves_without_running_the_vendor_install(tmp_path, on_path) -> None:
    prefix = tmp_path / "prefix"
    binary = _write_executable(prefix / "node_modules/.bin/pi", "#!/bin/sh\necho 0.84.2\n")
    log = tmp_path / "npm.log"
    on_path(_npm_shim(tmp_path, log))

    assert resolve_binary(get_adapter("pi"), prefix=prefix) == str(binary)
    assert not log.exists()


@pytest.mark.unit
def test_an_absent_binary_runs_exactly_the_descriptors_vendor_install(tmp_path, on_path) -> None:
    prefix = tmp_path / "prefix"
    binary = prefix / "node_modules/.bin/pi"
    log = tmp_path / "npm.log"
    on_path(_npm_shim(tmp_path, log, installs=binary))

    assert resolve_binary(get_adapter("pi"), prefix=prefix) == str(binary)
    assert log.read_text(encoding="utf-8").splitlines() == ["install", "--prefix", str(prefix), PI_PIN]


@pytest.mark.unit
def test_a_version_mismatch_reinstalls_the_pin(tmp_path, on_path) -> None:
    prefix = tmp_path / "prefix"
    binary = _write_executable(prefix / "node_modules/.bin/pi", "#!/bin/sh\necho 0.1.0\n")
    log = tmp_path / "npm.log"
    on_path(_npm_shim(tmp_path, log, installs=binary))

    assert resolve_binary(get_adapter("pi"), prefix=prefix) == str(binary)
    assert PI_PIN in log.read_text(encoding="utf-8")


@pytest.mark.unit
def test_a_version_that_is_a_substring_of_another_does_not_pass_the_npm_gate(tmp_path, on_path) -> None:
    """The gate matches the pinned version as a whole word, as the install script does."""
    prefix = tmp_path / "prefix"
    binary = _write_executable(prefix / "node_modules/.bin/pi", "#!/bin/sh\necho 10.84.2999\n")
    log = tmp_path / "npm.log"
    on_path(_npm_shim(tmp_path, log, installs=binary))

    resolve_binary(get_adapter("pi"), prefix=prefix)
    assert PI_PIN in log.read_text(encoding="utf-8")


@pytest.mark.unit
def test_a_missing_vendor_tool_names_itself_in_the_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(VendorInstallError, match="npm"):
        resolve_binary(get_adapter("pi"), prefix=tmp_path / "prefix")


@pytest.mark.unit
def test_a_failing_vendor_install_carries_the_vendors_own_last_line(tmp_path, on_path) -> None:
    shim = tmp_path / "shim"
    _write_executable(shim / "npm", "#!/bin/sh\necho 'ERR! 404 not found' >&2\nexit 1\n")
    on_path(shim)

    with pytest.raises(VendorInstallError, match="404 not found"):
        resolve_binary(get_adapter("pi"), prefix=tmp_path / "prefix")


@pytest.mark.unit
def test_a_vendor_install_that_lands_no_binary_points_at_the_manual_escape_hatch(tmp_path, on_path) -> None:
    log = tmp_path / "npm.log"
    on_path(_npm_shim(tmp_path, log))

    with pytest.raises(VendorInstallError, match=r"evolution\.binary"):
        resolve_binary(get_adapter("pi"), prefix=tmp_path / "prefix")


@pytest.mark.unit
def test_the_git_kind_records_its_ref_and_reinstalls_when_only_the_ref_moved(tmp_path, monkeypatch) -> None:
    """``--version`` reports the package version, which a ref moves independently
    of, so the recorded ``repository@ref`` gates the git kind too."""
    descriptor = get_adapter("hermes")
    prefix = tmp_path / "prefix"
    binary = _write_executable(prefix / descriptor.install.binary_path, "#!/bin/sh\necho 'Hermes Agent v0.21.0'\n")
    prefix.joinpath(PIN_FILE).write_text("https://github.com/NousResearch/hermes-agent@v2020.1.1\n", encoding="utf-8")
    steps: list[list[str]] = []

    def record(command):
        steps.append(list(command))
        _write_executable(binary, "#!/bin/sh\necho 'Hermes Agent v0.21.0'\n")

    monkeypatch.setattr("reef.harness.vendor_install._run", record)

    assert resolve_binary(descriptor, prefix=prefix) == str(binary)
    assert [step[0:2] for step in steps] == [
        ["git", "clone"],
        [sys.executable, "-m"],
        [str(prefix / "venv/bin/python"), "-m"],
    ]
    assert prefix.joinpath(PIN_FILE).read_text(encoding="utf-8").strip() == (
        f"{descriptor.install.repository}@{descriptor.install.ref}"
    )
    assert resolve_binary(descriptor, prefix=prefix) == str(binary)
    assert len(steps) == 3


# -- the backend's use of it ----------------------------------------------


def _counting_resolver(monkeypatch, result):
    """Stand in for ``resolve_binary`` where the backend calls it, counting calls.

    ``result`` is the path a resolution yields, or an exception it raises.
    """
    calls: list[str] = []

    def resolve(descriptor, **kwargs):
        del kwargs
        calls.append(descriptor.name)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("reef.train.cordis_backend.backend.resolve_binary", resolve)
    return calls


def _backend(binary, propose):
    return CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one",),
        models=MODEL,
        binary=binary,
    )


def _step(backend):
    return run_backend_step(backend, batch(), {"steps": 0, "entries": []})


def _propose(nodes, samples, models):
    del nodes, samples, models
    return Mutation("create", "r1", RULES)


@pytest.mark.unit
def test_an_unset_binary_resolves_once_at_the_first_episode_and_is_reused(monkeypatch, tmp_path) -> None:
    """A deployment boots before the agent is installed: resolution waits for
    the first episode, and the step's two sides share the one result."""
    calls = _counting_resolver(monkeypatch, str(make_binary(tmp_path)))

    backend = _backend(None, _propose)
    assert calls == []

    result = _step(backend)
    assert calls == ["pi"]
    assert result.metrics["failures"] == {"new": 0, "persisting": 0, "fixed": 0}

    _step(backend)
    assert calls == ["pi"]


@pytest.mark.unit
def test_an_explicit_binary_never_reaches_the_vendor_resolution(monkeypatch, tmp_path) -> None:
    calls = _counting_resolver(monkeypatch, "/should/not/be/used")

    _step(_backend(str(make_binary(tmp_path)), _propose))
    assert calls == []


@pytest.mark.unit
def test_a_vendor_install_failure_is_a_launch_failure_and_is_retried_next_step(monkeypatch) -> None:
    """A machine without npm records why the episode could not run rather than
    crashing the deployment, and the next step tries the install again."""
    calls = _counting_resolver(monkeypatch, VendorInstallError("npm is required ... but is not on PATH"))

    backend = _backend(None, _propose)
    result = _step(backend)

    assert result.metrics["failures"] == {"new": 1, "persisting": 0, "fixed": 0}
    entry = FailureManifest.from_state(result.state["failure_manifest"]).entries[0]
    assert entry.stage == "launch"
    assert "npm is required" in entry.cause
    # One resolution per side, and the next step tries again: a transient
    # vendor failure is not remembered.
    assert calls == ["pi", "pi"]
    _step(backend)
    assert len(calls) > 2
