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
from reef.harness.executor import LocalExecutor, SandboxExecutor, SandboxUnavailable
from reef.harness.vendor_install import (
    DEFAULT_PREFIX_ROOT,
    PIN_FILE,
    PREFIX_ENV,
    VendorInstallError,
    install_prefix,
    resolve_binary,
)
from reef.service.install_script import render_install_script
from reef.train.cordis_backend import CordisBackend, Mutation
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

from .test_harness_recipe import MODEL, PI_FAKE, RULES, batch, evaluate, make_binary, run_backend_step

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


def _backend(binary, propose, *, executor=None):
    return CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one",),
        models=MODEL,
        binary=binary,
        executor=executor,
    )


def _step(backend):
    return run_backend_step(backend, batch(), {"steps": 0, "entries": []})


def _propose(nodes, samples, models):
    del nodes, samples, models
    return Mutation("create", "r1", RULES)


@pytest.mark.unit
def test_an_unset_binary_resolves_once_at_construction_and_is_reused(monkeypatch, tmp_path) -> None:
    """Startup resolves the agent, and every episode reuses that result."""
    calls = _counting_resolver(monkeypatch, str(make_binary(tmp_path)))

    backend = _backend(None, _propose)
    assert calls == ["pi"]

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
def test_a_vendor_install_failure_refuses_construction(monkeypatch) -> None:
    calls = _counting_resolver(monkeypatch, VendorInstallError("npm exited 1: ERR! 404 not found"))
    with pytest.raises(VendorInstallError, match="npm exited 1: ERR! 404 not found"):
        _backend(None, _propose)
    assert calls == ["pi"]


@pytest.mark.unit
def test_missing_npm_refuses_backend_construction(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(PREFIX_ENV, str(tmp_path / "install"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(VendorInstallError, match=r"npm.*not on PATH"):
        _backend(None, _propose)


@pytest.mark.unit
def test_version_probe_uses_the_adapters_offline_and_update_guards(monkeypatch, tmp_path, on_path) -> None:
    monkeypatch.setenv("PI_SKIP_VERSION_CHECK", "0")
    monkeypatch.setenv("PI_OFFLINE", "0")
    prefix = tmp_path / "prefix"
    binary = _write_executable(
        prefix / "node_modules/.bin/pi",
        '#!/bin/sh\n[ "$PI_SKIP_VERSION_CHECK" = 1 ] && [ "$PI_OFFLINE" = 1 ] && echo 0.84.2\n',
    )
    log = tmp_path / "npm.log"
    on_path(_npm_shim(tmp_path, log))
    assert resolve_binary(get_adapter("pi"), prefix=prefix) == str(binary)
    assert not log.exists()


@pytest.mark.unit
@pytest.mark.parametrize("custom_prefix", [False, True])
def test_sandboxed_backend_binds_the_vendor_tree_readonly(monkeypatch, tmp_path, custom_prefix) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(PREFIX_ENV, raising=False)
    if custom_prefix:
        monkeypatch.setenv(PREFIX_ENV, str(tmp_path / "custom-install"))
    prefix = install_prefix(get_adapter("pi"))
    # Match an npm install's symlink: mounting only .bin cannot resolve it.
    source = _write_executable(
        prefix / "node_modules/package/cli.py",
        PI_FAKE.replace(
            "prompt = ",
            'if "--version" in sys.argv:\n    print("0.84.2")\n    sys.exit(0)\nprompt = ',
            1,
        ),
    )
    binary = prefix / "node_modules/.bin/pi"
    binary.parent.mkdir(parents=True)
    binary.symlink_to("../package/cli.py")
    seen = []

    def launch(executor, argv, **kwargs):
        seen.append(executor._bwrap_argv(argv, **{k: v for k, v in kwargs.items() if k != "timeout"}))
        return LocalExecutor().launch(argv, **kwargs)

    monkeypatch.setattr(SandboxExecutor, "launch", launch)
    original = SandboxExecutor(egress_hosts=("localhost",))
    backend = _backend(None, _propose, executor=original)
    result = _step(backend)
    assert result.metrics["failures"] == {"new": 0, "persisting": 0, "fixed": 0}
    assert len(seen) == 2
    assert binary.resolve() == source.resolve()
    assert str(prefix) not in original.base_paths
    for argv in seen:
        mounts = [(argv[i], argv[i + 1], argv[i + 2]) for i in range(len(argv)) if argv[i] in ("--bind", "--ro-bind")]
        assert ("--ro-bind", str(prefix), str(prefix)) in mounts
        assert not any(kind == "--bind" and path == str(prefix) for kind, path, _ in mounts)
        assert not any(path == str(prefix.parent) or path == str(Path.home()) for _, path, _ in mounts)
        assert argv[argv.index("--") + 1] == str(binary)
        assert "--unshare-net" not in argv


@pytest.mark.unit
def test_an_explicit_binary_does_not_add_a_vendor_mount(monkeypatch, tmp_path) -> None:
    calls = _counting_resolver(monkeypatch, "/should/not/be/used")
    sandbox = SandboxExecutor()
    backend = _backend(str(make_binary(tmp_path)), _propose, executor=sandbox)
    assert calls == []
    assert backend._executor is sandbox


@pytest.mark.integration
def test_vendor_binary_runs_in_a_real_sandbox_with_a_readonly_install(monkeypatch, tmp_path) -> None:
    sandbox = SandboxExecutor()
    try:
        sandbox.preflight()
    except SandboxUnavailable as error:
        pytest.skip(str(error))
    monkeypatch.setenv(PREFIX_ENV, str(tmp_path / "install"))
    prefix = install_prefix(get_adapter("pi"))
    _write_executable(
        prefix / "node_modules/package/cli.sh",
        "#!/bin/sh\n"
        'if [ "$1" = --version ]; then echo 0.84.2; exit 0; fi\n'
        'if touch "$1/vendor-write" 2>/dev/null; then exit 1; fi\n'
        "touch workspace-write && echo readonly-install\n",
    )
    binary = prefix / "node_modules/.bin/pi"
    binary.parent.mkdir(parents=True)
    binary.symlink_to("../package/cli.sh")
    backend = _backend(None, _propose, executor=sandbox)
    root = tmp_path / "episode"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    outcome = backend._executor.launch(
        [backend._binary, str(prefix)],
        root=root,
        workspace=workspace,
        env={},
        timeout=10,
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert outcome.stdout.strip() == "readonly-install"
    assert (workspace / "workspace-write").exists()
    assert not (prefix / "vendor-write").exists()
