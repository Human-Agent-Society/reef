"""Reef harness wrapper: a capture proxy between the agent binary and Reef.

When invoked with agent arguments (e.g. ``reef-pi -p "fix the bug"``):

  Checks the installed release's requirements before starting the proxy or
  agent. Unmet items print their setup hints and exit 3. Required programs
  must still be on PATH, even when setup previously checked them off.

  1. Starts a local capture proxy (``reef_client.serve``) that forwards to
     Reef, injecting ``x-reef-scenario`` so the user's agent binary never
     needs to know about Reef headers.
  2. Rewrites the provider config in a temp copy of the composition to point
     the agent at the proxy instead of Reef directly.
  3. Runs the agent binary as a subprocess, with the adapter's
     ``client_args`` ahead of the person's arguments unless the first is one
     of its ``client_version_args``.
  4. After the agent exits, persists the captured receipts (the
     ``x-reef-agent-record-id`` values from each response) to disk.

  A first argument among the adapter's ``client_updater_args`` (for claude
  ``upgrade``, ``--update`` and ``--upgrade``, which run Claude Code's own
  updater) runs nothing: the wrapper says that reef pins the binary and that
  ``update`` installs the served release, and exits 1.

When invoked with ``report`` (e.g. ``reef-pi report --score 0.0 --feedback "..."``):

  1. Claims the oldest pending run's persisted receipts.
  2. POSTs a report to Reef with all captured receipts as ``references``
     (one trajectory sample), or one report per receipt with ``--per-receipt``.
  3. Clears the persisted receipts.

When invoked with ``evolve`` (e.g. ``reef-pi evolve "text me when you are blocked"``,
``reef-pi evolve "..." --wait [--timeout SECONDS]``; ``harness`` remains a compatibility alias):

  Sends an explicit manual training instruction to ``POST /reef/train`` with
  the installed release and the oldest pending session's id (or a fresh id
  when nothing is spooled). The scenario must use ``training_mode: manual``
  or ``hybrid``. Acceptance queues a step without inference receipts or a
  feedback report; the merged ``requires`` list rides ``training_request``
  in the commit's metrics. The accepted line is followed by a link to the
  request's page (``GET /reef/harness/requests/<id>/page`` with the scenario
  and the token as query parameters, so a browser opens it as is). With
  ``--wait`` the wrapper polls the release catalog every 5 s for the step
  that consumed the request (``--timeout`` seconds, 1800 by default), says
  once when the request's record shows a step took it, and prints the
  result with the next action (a pending release says it is not installed
  until promoted and names its page link; a skipped step's line quotes why
  the proposer produced nothing): exit 0 for a selected or pending release,
  1 for a rejected or skipped step, 2 when the timeout passes first. On a
  terminal the result hands over the next step: a selected release asks
  ``Install now? [Y/n]`` and, on yes, runs the setup prompts for it and
  then ``update``, closing with ``Installed release <id>. Restart reef-pi
  to use it.``; a pending release names its page, asks ``Promote now?
  [y/N]`` and, on yes, posts the promote and installs the new head the same
  way. Declined, or without a terminal, the next commands are printed.

When invoked with ``page`` (e.g. ``reef-pi page 3``, ``reef-pi page 3 --print``):

  Fetches the step's page (``GET /reef/harness/releases/<step>/page``) with
  the token and the scenario header a browser would not send into
  ``$XDG_CACHE_HOME/reef-harness/<scenario>-step-<step>.html`` (``~/.cache``
  by default), prints the path and opens it with ``open`` (macOS) or
  ``xdg-open``; ``--print`` prints the path and opens nothing.

When invoked with ``doctor`` (e.g. ``reef-pi doctor``):

  Prints one line per thing an install needs and exits 0 when they all hold:
  the interpreter behind the wrapper and whether it imports reef and
  reef-client, the service address and whether the token is accepted, the
  agent binary and its version, the tools the adapter wants on PATH, the
  installed release against the served head, every program a ``binary``
  requires item of the installed release names (looked for on PATH, never
  run), and any release that waits for a person's review with its step page.

When invoked with ``setup`` (e.g. ``reef-pi setup``, ``reef-pi setup --yes``,
``reef-pi setup --mark <name>``, ``reef-pi setup --release <id>``, ``reef-pi setup --json``,
``reef-pi setup --set NAME=VALUE``, ``reef-pi setup --run NAME``):

  1. Reads what the newest release that is not pending (``--release <id>``
     names any catalog release instead, a pending one included) requires of
     you: every ``training_request.requires`` item over the release's chain
     in ``GET /reef/harness/releases``, merged by name as the manifest merges
     them (``permission``, ``env``, ``service`` or ``binary`` items, each with
     an optional ``check`` and an optional ``prompt``, one sentence saying what
     to enter, grant or install), the check offs the ``.reef-harness-release`` release file
     records under ``setup``, and the values the ``.reef-harness-env`` env
     file beside it holds.
  2. Prints every item with its check as written and its prompt. An ``env``
     item is met when its variable (the check, else the name) is set in the
     environment or the env file; otherwise it asks for the value
     (``getpass`` when the name contains TOKEN, KEY, SECRET or PASSWORD,
     ``input`` else; ``--yes`` asks nothing) and stores it in the env file.
     For an unmet ``permission`` or ``service`` item it asks ``run it?
     [y/N]`` (``--yes`` answers yes) and runs the check through the shell,
     exit status zero meaning met. A ``binary`` item names a program: it is
     met when the program is on PATH and the item names no check, and when it
     names one, the program is looked for first and the check is then asked
     for and run the same way, so no check runs for a program that is not
     there. ``--mark <name>`` checks an item off by
     hand and runs nothing. A check off records the check it stood for, so
     an item whose check changed since counts as unmet and runs again.
  3. Records each met item in the release file's ``setup`` and exits 0 when every
     item is met, 1 otherwise. This is the one place a check ever runs: the
     install script only reads the check offs, and a session start prints
     what is unmet and exits 3 without starting the agent.

  Three flag forms serve scripts and the extensions, one item at a time:
  ``--json`` prints ``{"release_id": ..., "items": [{name, kind, check,
  prompt, met}, ...]}`` for the release to set up and runs nothing (exit 0);
  ``--set NAME=VALUE`` stores the value of the ``env`` item NAME in the env
  file and checks it off (exit 0; an unknown or non-env name is exit 2; the
  value is an argument, never shell source, and stays one line); ``--run
  NAME`` runs that one item's check without asking, the caller having
  confirmed it, and checks it off when it passes (exit 0 when met, 1
  otherwise, 2 for an unknown name).

When invoked with ``update`` (e.g. ``reef-pi update``, ``reef-pi update --release <id>``):

  Fetches the install script (``GET /reef/harness/install?adapter=<adapter>``,
  plus ``&release_id=<id>`` with ``--release``) with the token and the
  scenario header, runs it with ``bash`` for this install root, and prints
  the installed release: exit 0, or 1 when the fetch or the script fails.
  While the release requires an item that is not met, the items are printed
  and nothing is fetched: exit 3, since the script would refuse anyway;
  this says why first. When called from an active session into its own install
  directory, setup and update use that session's service, scenario, and token.
  A successful update restores the installation to the session's configuration
  even if another install rewrote it. Commands targeting another install directory
  use that directory's configuration. An unknown release reports the service and
  scenario whose catalog was queried; it never substitutes a different release.

When invoked with ``--help``, ``-h`` or ``help``:

  Prints the wrapper's own usage (the subcommands above; anything else runs
  the agent), then, for ``--help`` and ``-h``, runs the agent with the same
  arguments so its help follows.

Env vars (baked into the wrapper at install time):

  ``REEF_HARNESS_BINARY``    absolute path to the agent binary
  ``REEF_HARNESS_COMPOSE``   absolute path to the composition directory
  ``REEF_HARNESS_SCENARIO``  the reef scenario name
  ``REEF_HARNESS_ADAPTER``   adapter name (any descriptor whose env relocates its composition
                             with a {root}/<dir> entry; terminus has none and gets no wrapper)
  ``REEF_HARNESS_ENV_VAR``   the env var that relocates the composition

Optional:

  ``REEF_TOKEN``  bearer token for the reef service (if auth is enabled); unset, the wrapper
                  uses the token the install wrote into the tree's model binding

The env file, ``<install root>/.reef-harness-env`` beside the release file,
holds the values the person gave ``setup`` for ``env`` items: ``NAME=VALUE``
lines, mode 0600, written by ``setup`` alone. A session run through the
wrapper gets each variable unless the shell already sets it (the shell
wins); the values never enter the tree and are never sent anywhere. The
wrapper also exports ``REEF_HARNESS_WRAPPER``, the path of the ``reef-<adapter>``
script at the install root, so an extension in the agent can run ``update``
and ``setup`` from the session.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from reef_client.serve import CapturedTurn, CaptureStore, ServeConfig, build_handler

from reef.core.requirements import required_by
from reef.core.training_request import CLIENT_COMMANDS
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import NO_TOKEN_API_KEY, AdapterDescriptor


def _captures_dir() -> Path:
    return Path(os.environ.get("REEF_HARNESS_CAPTURES_DIR", str(Path.home() / ".reef" / "captures")))


def _scenario_key(scenario: str) -> str:
    return hashlib.sha256(scenario.encode()).hexdigest()


def _publish_captures(reef_url: str, scenario: str, turns: list[dict]) -> None:
    captures_dir = _captures_dir()
    captures_dir.mkdir(parents=True, exist_ok=True)
    key = _scenario_key(scenario)
    destination = captures_dir / f"{key}-{time.time_ns():020d}-{uuid.uuid4().hex}.pending.json"
    payload = json.dumps({"reef_url": reef_url, "scenario": scenario, "turns": turns})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=captures_dir, prefix=f".{key}-", suffix=".tmp", delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _process_is_running(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_start_id(process_id: int) -> str | None:
    try:
        fields = Path(f"/proc/{process_id}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    except (IndexError, OSError):
        return None
    return fields[19] if len(fields) > 19 else None


def _claim_is_abandoned(owner: int, owner_start_id: str) -> bool:
    if not _process_is_running(owner):
        return True
    current_start_id = _process_start_id(owner)
    return current_start_id is not None and current_start_id != owner_start_id


def _claim_captures(scenario: str) -> tuple[Path, Path] | None:
    captures_dir = _captures_dir()
    key = _scenario_key(scenario)
    pending_files = sorted(captures_dir.glob(f"{key}-*.pending.json")) if captures_dir.exists() else []

    # A process that dies after claiming a spool entry cannot restore it. Once
    # its PID is gone, make that exact entry eligible for the next report.
    if captures_dir.exists():
        for claimed_file in sorted(captures_dir.glob(f"{key}-*.reporting-*.json")):
            owner_parts = claimed_file.name.rsplit(".reporting-", 1)[1].split("-", 2)
            if (
                len(owner_parts) >= 2
                and owner_parts[0].isdigit()
                and _claim_is_abandoned(int(owner_parts[0]), owner_parts[1])
            ):
                pending_files.append(claimed_file)
        pending_files.sort(key=lambda path: path.name.split(".", 1)[0])

    legacy_file = captures_dir / f"{scenario}.json"
    if legacy_file.is_file():
        pending_files.insert(0, legacy_file)

    for pending_file in pending_files:
        if pending_file == legacy_file:
            stem = f"{key}-00000000000000000000-legacy"
        else:
            stem = pending_file.name.split(".pending.json", 1)[0].split(".reporting-", 1)[0]
        process_id = os.getpid()
        process_start_id = _process_start_id(process_id) or "unknown"
        claimed_file = captures_dir / f"{stem}.reporting-{process_id}-{process_start_id}-{uuid.uuid4().hex}.json"
        try:
            os.replace(pending_file, claimed_file)
        except FileNotFoundError:
            continue
        return pending_file, claimed_file
    return None


class WrapperError(Exception):
    """The tree cannot be run through the proxy; the message says why and what to change."""


@dataclass(frozen=True)
class _Binding:
    """One place the adapter's model binding writes ``{base_url}``: the target file, the key path, the template."""

    target: str
    path: tuple[str, ...]
    template: str

    @property
    def suffix(self) -> str:
        return self.template.split("{base_url}", 1)[1]


def _bindings(descriptor: AdapterDescriptor, placeholder: str = "{base_url}") -> list[_Binding]:
    """The places the adapter's model binding renders ``placeholder``: Reef's address, or with ``{api_key}`` its token."""
    found: dict[tuple[str, tuple[str, ...]], _Binding] = {}
    for templates in descriptor.model_binding.values():
        for node in templates:
            target = str(node.get("target", "primary"))
            stack: list[tuple[tuple[str, ...], Any]] = [((), node.get("data", {}))]
            while stack:
                path, value = stack.pop()
                if isinstance(value, Mapping):
                    stack.extend(((*path, str(key)), item) for key, item in value.items())
                elif isinstance(value, str) and placeholder in value:
                    found.setdefault((target, path), _Binding(target, path, value))
    return list(found.values())


def _binding_file(descriptor: AdapterDescriptor, compose_dir: Path, binding: _Binding) -> Path:
    """The binding's target file under the composition directory; a target that escapes it is refused."""
    _, subdir = descriptor.compose_relocation()
    path = PurePosixPath(descriptor.config_targets[binding.target].path)
    if path.is_absolute() or ".." in path.parts:
        raise WrapperError(f"binding file {str(path)!r} escapes the tree")
    if subdir != "." and subdir not in {str(parent) for parent in path.parents}:
        raise WrapperError(f"binding file {str(path)!r} is outside the composition directory {subdir!r}")
    return compose_dir / (path.relative_to(subdir) if subdir != "." else path)


#: A config key and the URL it holds, in JSON, YAML, TOML or dotenv spelling.
_KEYED_URL = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\"?\s*[:=]\s*[\"']?(?P<url>https?://[^\s\"'`<>\\,;]+)")


def _has_key(text: str, key: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_-]){re.escape(key)}(?![A-Za-z0-9_-])", text) is not None


def _locate(text: str, binding: _Binding) -> re.Match[str] | None:
    """The one URL under the binding's key path.

    Candidates share the leaf key; when several do, the nearest parent keys
    in the text before each candidate settle it, so a second provider in
    the same file cannot be taken for Reef. The outermost key is the
    container every entry sits in, so it never settles anything."""
    matches = [m for m in _KEYED_URL.finditer(text) if m.group("key") == binding.path[-1]]
    if len(matches) <= 1:
        return matches[0] if matches else None
    starts = [0, *(m.end() for m in matches[:-1])]
    candidates = list(zip(starts, matches, strict=True))
    for parent in reversed(binding.path[1:-1]):
        narrowed = [(start, m) for start, m in candidates if _has_key(text[start : m.start()], parent)]
        if len(narrowed) == 1:
            return narrowed[0][1]
        if narrowed:
            candidates = narrowed
    keys = "/".join(binding.path)
    raise WrapperError(f"{len(candidates)} entries hold a URL at {keys}; keep one Reef entry there")


def _extract_reef_url(adapter: str, compose_dir: Path) -> str | None:
    """Reef's base URL, read from where the adapter's binding writes it, without the template's own suffix."""
    descriptor = get_adapter(adapter)
    for binding in _bindings(descriptor):
        file = _binding_file(descriptor, compose_dir, binding)
        if not file.is_file():
            continue
        match = _locate(file.read_text(encoding="utf-8"), binding)
        if match is None:
            continue
        url = match.group("url").rstrip("/")
        suffix = binding.suffix.rstrip("/")
        return url[: -len(suffix)] if suffix and url.endswith(suffix) else url
    return None


def _parse_binding_file(file: Path) -> Any:
    """The binding file as the adapter's quirks wrote it: JSON, TOML, YAML or dotenv, by its name."""
    text = file.read_text(encoding="utf-8")
    suffix = file.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        return tomllib.loads(text)
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if file.name == ".env" or suffix == ".env":
        pairs = (line.split("=", 1) for line in text.splitlines() if "=" in line and not line.lstrip().startswith("#"))
        return {key.strip(): value.strip().strip("\"'") for key, value in pairs}
    raise WrapperError(f"binding file {file.name!r} is in a format the wrapper does not read")


def _extract_reef_token(adapter: str, compose_dir: Path) -> str | None:
    """The token the install wrote into the tree, at the key path where the adapter's binding renders ``{api_key}``.

    The file is parsed, not searched: a second provider's key in the same
    file is never taken for Reef's, and a Reef entry the install left without a token (no ``REEF_TOKEN`` in the
    installing shell: ``NO_TOKEN_API_KEY``, or empty from an older install) yields nothing."""
    descriptor = get_adapter(adapter)
    for binding in _bindings(descriptor, "{api_key}"):
        file = _binding_file(descriptor, compose_dir, binding)
        if not file.is_file():
            continue
        try:
            value = _parse_binding_file(file)
        except (ValueError, yaml.YAMLError) as exc:
            raise WrapperError(f"binding file {file.name!r} does not parse: {exc}") from None
        for key in binding.path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, str) and value and value != NO_TOKEN_API_KEY:
            return value
    return None


def client_report() -> dict[str, Any]:
    """This machine as a request reports it, so the proposer builds for it rather than for its own sandbox: the
    platform, and which of ``CLIENT_COMMANDS`` are on the PATH. Read from the PATH; nothing is run."""
    import platform

    return {
        "platform": sys.platform,
        "arch": platform.machine(),
        "release": platform.release(),
        "commands": {name: shutil.which(name) is not None for name in CLIENT_COMMANDS},
    }


def _reef_token(adapter: str, compose_dir: str) -> str | None:
    """The bearer token: ``REEF_TOKEN`` when set, else the one the install wrote into the tree's model binding."""
    token = os.environ.get("REEF_TOKEN")
    if token or not compose_dir:
        return token or None
    try:
        return _extract_reef_token(adapter, Path(compose_dir))
    except WrapperError as exc:
        sys.exit(f"reef-{adapter}: {exc}")


def _strip_v1(url: str) -> str:
    return url[:-3] if url.endswith("/v1") else url


def _materialize(temp: Path, compose: Path, relative: PurePosixPath) -> Path:
    """The path for ``relative`` under the temp copy, with every symlinked ancestor replaced by a real directory.

    A rewrite must never follow a directory symlink into the installed tree,
    so each ancestor becomes a directory of symlinks to its siblings."""
    here, there = temp, compose
    for part in relative.parts[:-1]:
        here, there = here / part, there / part
        if here.is_symlink():
            here.unlink()
            here.mkdir()
            for item in there.iterdir():
                os.symlink(item, here / item.name)
    return here / relative.parts[-1]


def _rewrite_config(adapter: str, compose_dir: Path, temp_dir: Path, proxy_port: int) -> None:
    """Copy each binding file into the temp copy with the binding's URL, and only it, pointed at the proxy.

    The rewritten value is the proxy plus the template's own suffix (``/v1``
    where the adapter expects it), whatever the tree spelled, so the agent's
    request paths land where the proxy captures them."""
    descriptor = get_adapter(adapter)
    proxy = f"http://127.0.0.1:{proxy_port}"
    for binding in _bindings(descriptor):
        src = _binding_file(descriptor, compose_dir, binding)
        if not src.is_file():
            continue
        text = src.read_text(encoding="utf-8")
        match = _locate(text, binding)
        if match is None:
            continue
        span = match.span("url")
        text = text[: span[0]] + proxy + binding.suffix + text[span[1] :]
        dst = _materialize(temp_dir, compose_dir, PurePosixPath(src.relative_to(compose_dir).as_posix()))
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.write_text(text, encoding="utf-8")


def _create_temp_composition(adapter: str, compose_dir: str, proxy_port: int) -> str:
    """Symlink the composition into a temp dir, overriding the binding files."""
    compose = Path(compose_dir)
    temp_dir = tempfile.mkdtemp(prefix="reef-harness-")
    temp = Path(temp_dir)

    for item in compose.iterdir():
        dst = temp / item.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(item, dst)

    _rewrite_config(adapter, compose, temp, proxy_port)
    return temp_dir


def _wait_for_proxy(port: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_captures", timeout=0.5)
            return True
        except OSError:
            time.sleep(0.05)
    return False


#: The response header Reef sets on every inference answer of a file serving scenario: the head release id.
RELEASE_HEADER = "x-reef-release-id"
#: The paths Reef serves inference on; a receipt rides on each. The proxy matches the path with its query, and
#: the Anthropic SDK posts /v1/messages?beta=true under beta headers, so that form is listed.
CAPTURE_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages?beta=true")


class ReleaseObserver(ABC):
    """Where the proxy hands the release id an inference response names."""

    @abstractmethod
    def observe(self, release_id: str) -> None: ...


class _Tags(MutableMapping[str, str]):
    """The proxy's tag channel: each name rides every forwarded call as ``x-reef-tag-<name>``."""

    def __init__(self, config: ServeConfig, fixed: Mapping[str, str]) -> None:
        self._config = config
        self._fixed = dict(fixed)
        self._tags: dict[str, str] = {}
        self._publish()

    def _publish(self) -> None:
        # A fresh dict per change: a handler reads config.override_headers whole per request, never a torn one.
        tagged = {f"x-reef-tag-{name}": value for name, value in self._tags.items()}
        self._config.override_headers = {**self._fixed, **tagged}

    def __getitem__(self, name: str) -> str:
        return self._tags[name]

    def __setitem__(self, name: str, value: str) -> None:
        self._tags[name] = value
        self._publish()

    def __delitem__(self, name: str) -> None:
        self._tags.pop(name)
        self._publish()

    def __iter__(self) -> Iterator[str]:
        return iter(dict(self._tags))

    def __len__(self) -> int:
        return len(self._tags)


class _TaggedStore(CaptureStore):
    """The capture store plus, per capture, the tags in force when it landed, so a trial's receipts are told apart."""

    def __init__(self, tags: Mapping[str, str]) -> None:
        super().__init__()
        self._tags = tags
        self._tagged: list[dict[str, Any]] = []
        self._guard = threading.Lock()

    def add(self, turn: CapturedTurn) -> None:
        with self._guard:
            super().add(turn)
            self._tagged.append({**asdict(turn), "tags": dict(self._tags)})

    def drain(self) -> list[dict[str, Any]]:
        """The captures since the last drain, with their tags; what one publish carries."""
        with self._guard:
            turns, self._tagged = self._tagged, []
            return turns

    def clear(self) -> int:
        # An agent that reported its receipts clears the proxy; the list a publish drains must go with them.
        with self._guard:
            self._tagged = []
            return super().clear()


def _observing_handler(base: type[BaseHTTPRequestHandler], observer: ReleaseObserver) -> type[BaseHTTPRequestHandler]:
    class Handler(base):  # type: ignore[valid-type,misc]
        def _relay(self, response: Any, *args: Any) -> None:
            # Before the body reaches the agent: a head the answer names is queued by the time the agent reads it.
            release = response.getheader(RELEASE_HEADER)
            if release:
                observer.observe(str(release))
            super()._relay(response, *args)

    return Handler


class CaptureProxy:
    """The capture proxy between an agent and Reef, in process.

    Forwards to ``upstream`` with ``x-reef-scenario`` and the token, tags every
    call with ``tags`` as ``x-reef-tag-<name>`` headers, keeps the receipts,
    and hands the release id an answer names to ``observer``."""

    def __init__(
        self,
        upstream: str,
        scenario: str,
        token: str | None = None,
        *,
        tags: Mapping[str, str] | None = None,
        observer: ReleaseObserver | None = None,
        listen_host: str = "127.0.0.1",
    ) -> None:
        self.upstream = upstream
        self.scenario = scenario
        self.listen_host = listen_host
        fixed: dict[str, str] = {"x-reef-scenario": scenario}
        if token:
            fixed["authorization"] = f"Bearer {token}"
        self._config = ServeConfig(upstream=upstream, listen_port=0, capture_paths=CAPTURE_PATHS)
        #: The tag channel: the record keeps which release (and which trial) answered, under ``metadata.tags``.
        self.tags: MutableMapping[str, str] = _Tags(self._config, fixed)
        for name, value in (tags or {}).items():
            self.tags[name] = value
        self._store = _TaggedStore(self.tags)
        handler = build_handler(self._config, self._store)
        self._handler = handler if observer is None else _observing_handler(handler, observer)
        self._server: ThreadingHTTPServer | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise WrapperError("the capture proxy is not running")
        return int(self._server.server_address[1])

    def start(self) -> None:
        server = ThreadingHTTPServer((self.listen_host, 0), self._handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self._server = server
        if not _wait_for_proxy(self.port):
            self.stop()
            raise WrapperError("capture proxy failed to start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def drain(self) -> list[dict[str, Any]]:
        """The captures since the last drain or publish, each with the tags in force when it landed."""
        return self._store.drain()

    def publish_turn(self) -> int:
        """Spool the receipts captured since the last publish, so ``report`` claims them; the count written."""
        turns = self.drain()
        if turns:
            _publish_captures(self.upstream, self.scenario, turns)
        return len(turns)


#: The release file the install script and harness_pull write at the tree
#: root; version_check.ts reads the same name.
HARNESS_RELEASE_FILE = ".reef-harness-release"


def _release_file_path(compose_dir: str) -> Path:
    return Path(compose_dir).parent / HARNESS_RELEASE_FILE


def _read_release_info(compose_dir: str) -> dict[str, Any] | None:
    """The release file beside the installed tree as a dict; None when there is none or it is not JSON."""
    try:
        record = json.loads(_release_file_path(compose_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _write_release_info(compose_dir: str, record: Mapping[str, Any]) -> None:
    # Written beside and renamed over, so a session starting meanwhile reads the old record or the new, never half.
    release_file = _release_file_path(compose_dir)
    staging = release_file.with_name(f".{release_file.name}.part")
    staging.write_text(json.dumps(dict(record), indent=2) + "\n", encoding="utf-8")
    os.replace(staging, release_file)


#: The values the person gave ``setup`` for ``env`` items, beside the release file; ``run_agent`` reads it.
HARNESS_ENV_FILE = ".reef-harness-env"
#: A variable named like one of these holds a credential, so its value is asked for without echo.
_SECRET_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD")


def _env_file_path(compose_dir: str) -> Path:
    return Path(compose_dir).parent / HARNESS_ENV_FILE


def _read_env_file(compose_dir: str) -> dict[str, str]:
    """The env file as a dict, ``NAME=VALUE`` per line; blank lines, ``#`` comments and lines without ``=`` are skipped."""
    try:
        lines = _env_file_path(compose_dir).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in lines:
        name, separator, value = line.partition("=")
        if not separator or not name.strip() or name.lstrip().startswith("#"):
            continue
        values[name.strip()] = value.strip()
    return values


def _write_env_file(compose_dir: str, values: Mapping[str, str]) -> None:
    """Write the env file whole, readable by the person alone (mode 0600), renamed over so a run reads old or new."""
    path = _env_file_path(compose_dir)
    staging = path.with_name(f".{path.name}.part")
    staging.unlink(missing_ok=True)
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("".join(f"{name}={value}\n" for name, value in values.items()))
    os.replace(staging, path)


def _env_variable(item: Mapping[str, Any]) -> str:
    """The variable an ``env`` item stands for, its check else its name: what the extension reads."""
    return str(item.get("check") or item["name"])


def _env_met(item: Mapping[str, Any], values: Mapping[str, str]) -> bool:
    """Whether an ``env`` item's variable is set: in the environment, else in the env file ``values``."""
    variable = _env_variable(item)
    return bool(os.environ.get(variable) or values.get(variable))


def _binary_found(item: Mapping[str, Any]) -> str | None:
    """Where a ``binary`` item's program is on PATH, None when it is not there; runs nothing."""
    name = item.get("name")
    return shutil.which(name) if isinstance(name, str) and name else None


def _auto_met(item: Mapping[str, Any], values: Mapping[str, str]) -> bool:
    """Whether an item is met without a check off, by looking and never by running.

    An ``env`` item whose variable the environment or the env file sets, and a
    ``binary`` item with no check whose program is on PATH. A ``binary`` item
    that names a check is met only once that check has run and been checked
    off, as a ``permission`` or ``service`` item is. Nothing here is recorded:
    the look is cheap and stays true to the machine, so a program that goes
    away stops meeting its item."""
    kind = item.get("kind")
    if kind == "env":
        return _env_met(item, values)
    if kind == "binary":
        return not item.get("check") and _binary_found(item) is not None
    return False


def _installed_release(compose_dir: str) -> str | None:
    """The release id of the installed tree, from the release file beside it."""
    release = (_read_release_info(compose_dir) or {}).get("release_id")
    return release if isinstance(release, str) and release else None


def _named_items(value: Any) -> list[dict[str, Any]]:
    """The objects with a string ``name`` in a list; the release file's ``requires`` and ``setup`` and a row's are read alike."""
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping) and isinstance(item.get("name"), str)]


def _met(item: Mapping[str, Any], record: Mapping[str, Any] | None) -> bool:
    """Whether a check off meets an item: it names it and the check it recorded is the item's.

    A check off without a recorded check, from an older release file, counts by
    name; the install script's evaluation applies the same rule."""
    return record is not None and ("check" not in record or record.get("check") == item.get("check"))


def _unmet(requires: Any, setup: Any) -> list[dict[str, Any]]:
    """The required items no check off meets."""
    checked = {item["name"]: item for item in _named_items(setup)}
    return [item for item in _named_items(requires) if not _met(item, checked.get(item["name"]))]


def _item_line(item: Mapping[str, Any]) -> str:
    """One required item as every listing prints it: the name, the kind and the check as written."""
    check = item.get("check")
    return f"{item['name']} ({item.get('kind', 'unknown')})" + (f": {check}" if check else "")


def run_agent(binary: str, compose_dir: str, scenario: str, adapter: str, env_var: str, args: list[str]) -> None:
    upstream = _reef_url_of(adapter, compose_dir)

    record = _read_release_info(compose_dir) or {}
    release = _installed_release(compose_dir)
    stored = _read_env_file(compose_dir)
    checked = {item["name"]: item for item in _named_items(record.get("setup"))}
    unmet = []
    missing_programs: set[str] = set()
    for item in _named_items(record.get("requires")):
        # A saved check off cannot make an uninstalled program available.
        if item.get("kind") == "binary" and _binary_found(item) is None:
            missing_programs.add(item["name"])
            unmet.append(item)
        elif not _met(item, checked.get(item["name"])) and not _auto_met(item, stored):
            unmet.append(item)
    if unmet:
        # Keep stdout clean for scripted runs and stop before creating a session.
        print(
            f"reef-{adapter}: cannot start agent; this release has unmet requirements:",
            file=sys.stderr,
        )
        for item in unmet:
            print(f"  {_item_line(item)}", file=sys.stderr)
            if item["name"] in missing_programs:
                print(f"    {item['name']} is not on PATH; install it before starting the agent", file=sys.stderr)
            if item.get("prompt"):
                print(f"    {item['prompt']}", file=sys.stderr)
        named_release = f" --release {release}" if release is not None else ""
        print(
            f"reef-{adapter}: run reef-{adapter} setup{named_release}, then start the agent again",
            file=sys.stderr,
        )
        sys.exit(3)
    # Every call carries the session as a tag, so the spool and the agent records name the session an ask refers to.
    tags = {"session": str(uuid.uuid4()), **({"release": release} if release else {})}
    token = _reef_token(adapter, compose_dir)
    proxy = CaptureProxy(upstream, scenario, token, tags=tags)
    try:
        proxy.start()
    except WrapperError as exc:
        sys.exit(f"reef-{adapter}: {exc}")

    descriptor = get_adapter(adapter)
    # The temp copy is removed after the run: state kept in the installed tree is linked into it
    # instead, so a later run finds the sessions and settings an earlier run saved.
    kept = {
        state: PurePosixPath(state.path).relative_to(descriptor.compose_relocation()[1])
        for state in descriptor.client_state
    }
    for state, relative in kept.items():
        path = Path(compose_dir) / relative
        if state.kind == "directory":
            path.mkdir(parents=True, exist_ok=True)
        elif state.kind == "sqlite" and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.closing(sqlite3.connect(path)) as database:
                database.execute("VACUUM")  # writes the database header, so the file is a database
    temp_dir = _create_temp_composition(adapter, compose_dir, proxy.port)
    env = os.environ.copy()
    env[env_var] = temp_dir
    # What an interactive run needs beyond the episode env; the person's own setting wins.
    for key, value in descriptor.client_env.items():
        env.setdefault(key, value)
    # The values the person gave setup, for the extensions that read them; a variable the shell sets wins.
    for key, value in stored.items():
        if not env.get(key):
            env[key] = value
    # The update notice extension needs the service address, the scenario,
    # and the true install root; the relocated temp copy carries none of them.
    install_root = Path(compose_dir).resolve().parent
    env["REEF_SERVICE_URL"] = upstream
    env["REEF_SCENARIO"] = scenario
    env["REEF_HARNESS_DEST"] = str(install_root)
    wrapper = install_root / f"reef-{adapter}"
    if wrapper.is_file():
        # The wrapper the install wrote, so an extension can run its update and setup from the session.
        env["REEF_HARNESS_WRAPPER"] = str(wrapper)
    if token:
        env["REEF_TOKEN"] = token  # the extensions in the agent reach reef with the token the proxy uses
    # An evolved tool that starts a second agent session finds this harness's own binary first.
    env["PATH"] = os.pathsep.join([str(Path(binary).resolve().parent), env.get("PATH", "")])
    if adapter == "native":
        # The loop's session log outlives the temp copy: it lands beside the installed tree.
        env.setdefault("REEF_NATIVE_SESSION_DIR", str(Path(compose_dir).resolve() / "sessions"))

    # Ahead of the person's arguments: a binary that reads the last of a repeated flag keeps the person's.
    # A version flag starts no session, and a binary may take it only when nothing is ahead of it.
    leading_args = () if args and args[0] in descriptor.client_version_args else descriptor.client_args
    try:
        result = subprocess.run([binary, *leading_args, *args], env=env)
    finally:
        proxy.publish_turn()
        proxy.stop()
        try:
            for state, relative in kept.items():
                written = Path(temp_dir) / relative
                # A file still linked was written through the link; a real one is new, or renamed over the link.
                if state.kind != "file" or written.is_symlink() or not written.is_file():
                    continue
                destination = Path(compose_dir) / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                file_descriptor, staging = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}-")
                os.close(file_descriptor)
                shutil.copy2(written, staging)  # with its mode: dsh refuses credentials readable beyond their owner
                os.replace(staging, destination)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    sys.exit(result.returncode)


def _reportable(turn: Mapping[str, Any]) -> bool:
    """A captured exchange with a receipt that was not a trial's: a trial never reaches the evaluation."""
    return bool(turn.get("receipt")) and "trial" not in (turn.get("tags") or {})


def _reef_headers(scenario: str, token: str | None) -> dict[str, str]:
    """Scenario and authentication headers for Reef's record routes."""
    headers = {"Content-Type": "application/json", "x-reef-scenario": scenario}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return headers


def report(scenario: str, adapter: str, score: float, feedback: str, per_receipt: bool = False) -> None:
    # Resolved before any claim: a token the tree cannot yield exits without a claim to restore.
    compose_dir = os.environ.get("REEF_HARNESS_COMPOSE", "")
    headers = _reef_headers(scenario, _reef_token(adapter, compose_dir))
    while True:
        claim = _claim_captures(scenario)
        if claim is None:
            sys.exit(f"reef-{adapter}: no captured receipts for scenario {scenario!r}")
        pending_file, captures_file = claim

        try:
            data = json.loads(captures_file.read_text(encoding="utf-8"))
            reef_url = data["reef_url"]
            receipts = [t["receipt"] for t in data["turns"] if _reportable(t)]
        except BaseException:
            os.replace(captures_file, pending_file)
            raise
        if receipts:
            break
        captures_file.unlink()

    # One report referencing the whole run batches as one trajectory sample;
    # --per-receipt sends the same score against each receipt on its own.
    reference_lists = [[receipt] for receipt in receipts] if per_receipt else [receipts]
    release = _installed_release(compose_dir)
    metadata = {"client_release": release} if release else {}

    def restore_unsent(sent: int) -> None:
        # A partial per-receipt failure must not resend what already posted:
        # the retry claim keeps only the receipts that never went out.
        posted = {receipt for references in reference_lists[:sent] for receipt in references}
        data["turns"] = [turn for turn in data["turns"] if turn.get("receipt") not in posted]
        captures_file.write_text(json.dumps(data), encoding="utf-8")
        os.replace(captures_file, pending_file)

    for sent, references in enumerate(reference_lists):
        body: dict[str, Any] = {"score": score, "feedback": feedback, "references": references}
        if metadata:
            body["metadata"] = metadata
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{reef_url}/reef/report",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            restore_unsent(sent)
            detail = exc.read().decode(errors="replace")
            sys.exit(f"reef-{adapter}: report failed ({exc.code}): {detail}")
        except BaseException:
            restore_unsent(sent)
            raise

    captures_file.unlink()
    mode = "report per receipt" if per_receipt else "one report"
    print(f"reef-{adapter}: reported {len(receipts)} receipt(s) to {scenario} ({mode})")


def _spooled_session(scenario: str) -> str | None:
    """Read the session id without claiming or consuming feedback receipts."""
    directory = _captures_dir()
    paths = [directory / f"{scenario}.json", *sorted(directory.glob(f"{_scenario_key(scenario)}-*.pending.json"))]
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("turns"), list):
            continue
        for turn in data["turns"]:
            if not isinstance(turn, dict):
                continue
            tags = turn.get("tags")
            session = (tags.get("session") if isinstance(tags, dict) else None) or turn.get("session_id")
            if isinstance(session, str) and session:
                return session
    return None


def _clip(text: str, limit: int) -> str:
    """The first ``limit`` characters of a text, the cut marked."""
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _page_link(upstream: str, path: str, scenario: str, token: str | None) -> str:
    """A page a browser opens: the query carries what the wrapper sends as headers, the scenario and the token."""
    query = f"scenario={urllib.parse.quote(scenario, safe='')}"
    if token:
        query += f"&token={urllib.parse.quote(token, safe='')}"
    return f"{upstream}{path}?{query}"


def _request_page_link(upstream: str, scenario: str, token: str | None, record_id: str) -> str:
    """The link to a request's page, ``GET /reef/harness/requests/<id>/page``."""
    return _page_link(
        upstream, f"/reef/harness/requests/{urllib.parse.quote(record_id, safe='')}/page", scenario, token
    )


def _step_page_link(upstream: str, scenario: str, token: str | None, step: int) -> str:
    """The link to a step's page, ``GET /reef/harness/releases/<step>/page``."""
    return _page_link(upstream, f"/reef/harness/releases/{step}/page", scenario, token)


def _metrics_of(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = row.get("metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _request_of(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The request a step consumed, as its row records it."""
    request = _metrics_of(row).get("training_request")
    return request if isinstance(request, Mapping) else {}


def result_of(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] = ()) -> str:
    """A row's result as the extension reads it.

    A pending row stays pending in the catalog; a later promote row naming
    it makes it ``promoted at vN``. A settled step is ``selected``,
    ``rejected`` or ``skipped``; any other row reads as its operation."""
    if row.get("pending"):
        for step, other in enumerate(rows):
            if other.get("operation") == "promote" and other.get("rollback_target_release_id") == row.get(
                "release_id"
            ):
                return f"promoted at v{step}"
        return "pending"
    metrics = _metrics_of(row)
    selected = metrics.get("selected")
    if isinstance(selected, bool):
        return "selected" if selected else "rejected"
    if metrics.get("skipped"):
        return "skipped"
    return str(row.get("operation") or "unknown")


def _uncovered(row: Mapping[str, Any]) -> list[str]:
    """What the step's review left uncovered, when it recorded one."""
    notes = _metrics_of(row).get("proposal_notes")
    review = notes.get("review") if isinstance(notes, Mapping) else None
    items = review.get("uncovered") if isinstance(review, Mapping) else None
    if not isinstance(items, list):
        return []
    return [item.strip() for item in items if isinstance(item, str) and item.strip()]


def _failure_of(row: Mapping[str, Any]) -> str:
    """Why the proposer produced nothing, when the step recorded it under ``proposal_notes.failure``."""
    notes = _metrics_of(row).get("proposal_notes")
    failure = notes.get("failure") if isinstance(notes, Mapping) else None
    return failure.strip() if isinstance(failure, str) else ""


def result_line(adapter: str, step: int, rows: Sequence[Mapping[str, Any]], page: str) -> str:
    """One line for a settled step: its result and the next action, quoting the request's first 60 characters.

    The extension's watch says the same in the session; ``evolve --wait``
    and ``doctor`` say it here. ``page`` is the step's page link, which the
    pending line names as the review."""
    row = rows[step]
    metrics = _metrics_of(row)
    ask = _clip(str(_request_of(row).get("text") or "").strip(), 60)
    release = str(row.get("release_id") or "")[:8]
    selection_result = result_of(row, rows)
    if selection_result == "selected":
        return f"'{ask}' is published as release {release}. Restart reef-{adapter} to install it (the update notice offers it)."
    if selection_result == "pending":
        return (
            f"'{ask}' is ready as release {release}. This release changes an extension, so read it before it "
            f"runs: /versions v{step} opens the page, /versions v{step} install serves it. Page: {page}"
        )
    if selection_result == "rejected":
        selection = metrics.get("selection")
        reason = (selection.get("reason") if isinstance(selection, Mapping) else None) or "no reason recorded"
        return f"'{ask}' did not pass the checks ({reason}). Nothing changed; rephrase or split the request."
    if selection_result == "skipped":
        # The proposer's own reason, when the step recorded one: a failed model call, a reply with no entry.
        failure = _failure_of(row)
        why = f"{metrics.get('skipped')}: {failure}" if failure else str(metrics.get("skipped"))
        return f"'{ask}' produced no change ({why}). Nothing changed."
    return f"'{ask}' settled as {selection_result} (release {release}); reef-{adapter} page {step} shows it."


def _step_of(rows: Sequence[Mapping[str, Any]], record_id: str) -> int | None:
    """The step whose row consumed the request ``record_id``: its position in the catalog, oldest first."""
    return next((step for step, row in enumerate(rows) if _request_of(row).get("id") == record_id), None)


def _request_state(upstream: str, scenario: str, token: str | None, record_id: str) -> str:
    """Where the request stands by its record: ``started`` once a step took it (``compacted_at`` set), ``gone``
    when the service answers 404 (its scenario was reset), else ``waiting``.

    A read that fails for any other reason is no reason to stop waiting, so
    it reads as waiting and the next poll asks again."""
    path = f"/reef/scenarios/{urllib.parse.quote(scenario, safe='')}/records/{urllib.parse.quote(record_id, safe='')}"
    req = urllib.request.Request(f"{upstream}{path}", headers=_reef_headers(scenario, token))
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            record = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return "gone" if exc.code == 404 else "waiting"
    except (OSError, ValueError):
        return "waiting"
    if isinstance(record, Mapping) and record.get("compacted_at") is not None:
        return "started"
    return "waiting"


def _await_step(
    upstream: str,
    scenario: str,
    adapter: str,
    token: str | None,
    record_id: str,
    ask: str,
    *,
    timeout_s: float,
    poll_s: float,
) -> tuple[int, list[dict[str, Any]]] | str:
    """Poll the catalog until a step has consumed the request: its index and the catalog.

    ``"timeout"`` once ``timeout_s`` passes, the step still running, and
    ``"gone"`` when two polls in a row find no record of the request (its
    scenario was reset; one missing read can be a record not written yet).
    One line says when the record shows a step took it, so the wait is seen
    to move."""
    deadline = time.monotonic() + timeout_s
    started = False
    missing = 0
    while True:
        rows = _catalog(upstream, scenario, adapter, token)
        step = _step_of(rows, record_id)
        if step is not None:
            return step, rows
        if time.monotonic() >= deadline:
            print(
                f"reef-{adapter}: no result yet for '{ask}' after {timeout_s:g} s; /versions shows it when it settles"
            )
            return "timeout"
        if not started:
            state = _request_state(upstream, scenario, token, record_id)
            missing = missing + 1 if state == "gone" else 0
            if missing >= 2:
                print(
                    f"reef-{adapter}: request {record_id[:8]} is no longer on the service (its scenario was reset); "
                    "ask again"
                )
                return "gone"
            if state == "started":
                started = True
                print(f"reef-{adapter}: the step started; usually one to three minutes")
        time.sleep(poll_s)


def _confirm(adapter: str, question: str, *, default_yes: bool) -> bool:
    """Ask ``question`` on the terminal; an empty answer takes the default the question shows in capitals."""
    print(f"reef-{adapter}: {question} ", end="", flush=True)
    answer = sys.stdin.readline().strip().lower()
    if not answer:
        return default_yes
    return answer in ("y", "yes")


def _promote(upstream: str, scenario: str, adapter: str, token: str | None, release: str) -> str | None:
    """Promote a pending release through ``POST /reef/scenarios/<scenario>/promote``; the new head's id, or None, said."""
    req = urllib.request.Request(
        f"{upstream}/reef/scenarios/{urllib.parse.quote(scenario, safe='')}/promote",
        data=json.dumps({"release_id": release}).encode(),
        headers=_reef_headers(scenario, token),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        print(f"reef-{adapter}: promote failed ({exc.code}): {exc.read().decode(errors='replace')}", file=sys.stderr)
        return None
    except (OSError, ValueError) as exc:
        print(f"reef-{adapter}: promote failed: {exc}", file=sys.stderr)
        return None
    head = answer.get("release_id") if isinstance(answer, Mapping) else None
    if not isinstance(head, str) or not head:
        print(
            f"reef-{adapter}: reef answered the promote without a release_id: {json.dumps(answer)[:200]}",
            file=sys.stderr,
        )
        return None
    print(f"reef-{adapter}: promoted; the served head is release {head[:8]}")
    return head


def _next_commands(adapter: str, step: int, selection_result: str) -> str:
    """The commands that take the next step by hand, for a person who declined it or has no terminal."""
    if selection_result == "pending":
        return (
            f"/versions v{step} install in a reef-{adapter} session, or reef-{adapter} setup and reef-{adapter} update"
        )
    return f"reef-{adapter} setup, then reef-{adapter} update"


def _install(scenario: str, adapter: str, compose_dir: str, release: str) -> int:
    """The setup prompts for ``release``, then its install; the install's status when it fails, else 0, said."""
    # Setup first: the install script refuses a release whose items are not checked off.
    setup(scenario, adapter, compose_dir, release=release)
    status = update(scenario, adapter, compose_dir, release=release)
    if status != 0:
        return status
    print(f"reef-{adapter}: Installed release {release[:8]}. Restart reef-{adapter} to use it.")
    return 0


def _next_step(
    scenario: str,
    adapter: str,
    compose_dir: str,
    upstream: str,
    token: str | None,
    row: Mapping[str, Any],
    step: int,
    selection_result: str,
) -> int:
    """After a release, hand the person the next step: install a selected one, promote then install a pending one.

    On a terminal each step is a question; declined, or without a terminal,
    the commands are printed for later. 0 unless a step the person took
    failed, then its status."""
    release = str(row.get("release_id") or "")
    if selection_result not in ("selected", "pending") or not release:
        return 0
    if not sys.stdin.isatty():
        print(f"reef-{adapter}: next: {_next_commands(adapter, step, selection_result)}")
        return 0
    if selection_result == "pending":
        print(f"reef-{adapter}: read the change first: reef-{adapter} page {step}")
        if not _confirm(adapter, "Promote now? [y/N]", default_yes=False):
            print(f"reef-{adapter}: next: {_next_commands(adapter, step, selection_result)}")
            return 0
        head = _promote(upstream, scenario, adapter, token, release)
        if head is None:
            return 1
        release = head
    elif not _confirm(adapter, "Install now? [Y/n]", default_yes=True):
        print(f"reef-{adapter}: next: {_next_commands(adapter, step, selection_result)}")
        return 0
    return _install(scenario, adapter, compose_dir, release)


def harness(
    scenario: str,
    adapter: str,
    compose_dir: str,
    text: str,
    *,
    wait: bool = False,
    timeout_s: float = 1800.0,
    poll_s: float = 5.0,
) -> int:
    """Submit a native manual training request, leaving feedback receipts available; with ``wait``, report its result.

    The status is 0 once the request is accepted. With ``wait`` it is the
    result's: 0 for a release to install or review, 1 for a step that
    changed nothing, 2 when the timeout passes first; on a terminal a
    release hands over its next step and a failed step's status stands."""
    text = text.strip()
    if not text:
        sys.exit(f"reef-{adapter} evolve: the request is empty")
    release = _installed_release(compose_dir)
    if release is None:
        sys.exit(
            f"reef-{adapter}: no {HARNESS_RELEASE_FILE} release file at {Path(compose_dir).resolve().parent}: this "
            "tree did not come through reef's install channel, so a request cannot name the release it runs; "
            "nothing was sent"
        )
    upstream = _reef_url_of(adapter, compose_dir)

    # Session and release identify where the request came from; they do not select an inference batch.
    session = _spooled_session(scenario) or str(uuid.uuid4())

    body = {"text": text, "session": session, "release_id": release, "client": client_report()}
    token = _reef_token(adapter, compose_dir)
    req = urllib.request.Request(
        f"{upstream}/reef/train",
        data=json.dumps(body).encode(),
        headers=_reef_headers(scenario, token),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        sys.exit(f"reef-{adapter}: request failed ({exc.code}): {detail}")
    except OSError as exc:
        sys.exit(f"reef-{adapter}: reef unreachable at {upstream}: {exc}")
    record_id = answer.get("agent_record_id") if isinstance(answer, dict) else None
    if not isinstance(record_id, str):
        # A 200 without the id is a reef this wrapper does not know; say so instead of a traceback.
        sys.exit(f"reef-{adapter}: reef answered 200 without an agent_record_id: {json.dumps(answer)[:200]}")
    print(f"reef-{adapter}: training request {record_id} accepted")
    print(f"reef-{adapter}: watch it here: {_request_page_link(upstream, scenario, token, record_id)}")
    if not wait:
        print(f"reef-{adapter}: reef is running the step; add --wait to stay here, or check /versions later")
        return 0
    print(f"reef-{adapter}: reef is running the step; waiting up to {timeout_s:g} s for its result")
    settled = _await_step(
        upstream, scenario, adapter, token, record_id, _clip(text, 60), timeout_s=timeout_s, poll_s=poll_s
    )
    if isinstance(settled, str):
        return 2 if settled == "timeout" else 1  # timeout: the step still runs; gone: nothing will come
    step, rows = settled
    print(f"reef-{adapter}: {result_line(adapter, step, rows, _step_page_link(upstream, scenario, token, step))}")
    uncovered = _uncovered(rows[step])
    if uncovered:
        print(f"reef-{adapter}: not covered: {'; '.join(uncovered)}")
    selection_result = result_of(rows[step], rows)
    if selection_result in ("rejected", "skipped"):
        return 1
    return _next_step(scenario, adapter, compose_dir, upstream, token, rows[step], step, selection_result)


def _page_cache_dir() -> Path:
    """Where ``page`` keeps the pages it fetched: ``$XDG_CACHE_HOME/reef-harness``, under ``~/.cache`` by default."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "reef-harness"


def _open_in_browser(path: Path) -> bool:
    """Hand the file to the desktop's opener, ``open`` on macOS and ``xdg-open`` elsewhere; False when there is none."""
    opener = shutil.which("open" if sys.platform == "darwin" else "xdg-open")
    if opener is None:
        return False
    subprocess.run([opener, str(path)], check=False)
    return True


def step_of_version(text: str) -> int:
    """The step a version names, as ``/versions`` lists it: ``v3`` or ``3``."""
    digits = text.removeprefix("v")
    if not digits.isascii() or not digits.isdigit():
        raise argparse.ArgumentTypeError(f"{text!r} is no version; write v3 or 3")
    return int(digits)


def page(scenario: str, adapter: str, compose_dir: str, step: int, *, open_page: bool = True) -> int:
    """Fetch a step's page into the cache directory, print its path and open it; 0 once it is written.

    The route needs the token and the scenario header a browser would not
    send, so the wrapper fetches the page and hands the file over. A step
    the catalog does not hold exits with the route's 404 text."""
    upstream = _reef_url_of(adapter, compose_dir)
    req = urllib.request.Request(
        f"{upstream}/reef/harness/releases/{step}/page",
        headers=_reef_headers(scenario, _reef_token(adapter, compose_dir)),
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        sys.exit(f"reef-{adapter}: page read failed ({exc.code}): {detail}")
    except OSError as exc:
        sys.exit(f"reef-{adapter}: reef unreachable at {upstream}: {exc}")
    # The scenario names the file; a character no file system takes becomes a dash.
    path = _page_cache_dir() / f"{re.sub(r'[^A-Za-z0-9._-]+', '-', scenario)}-step-{step}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(html)
    print(path)
    if open_page and not _open_in_browser(path):
        print(f"reef-{adapter}: no open or xdg-open on PATH; open the file in a browser", file=sys.stderr)
    return 0


def _reef_url_of(adapter: str, compose_dir: str) -> str:
    """Reef's base URL from the installed tree, or an exit naming what is missing."""
    try:
        reef_url = _extract_reef_url(adapter, Path(compose_dir))
    except WrapperError as exc:
        sys.exit(f"reef-{adapter}: {exc}")
    if reef_url is None:
        sys.exit(f"reef-{adapter}: no Reef URL in the tree's model binding files")
    return _strip_v1(reef_url)


def _catalog(upstream: str, scenario: str, adapter: str, token: str | None) -> list[dict[str, Any]]:
    """The release catalog as ``GET /reef/harness/releases`` lists it, oldest first."""
    req = urllib.request.Request(f"{upstream}/reef/harness/releases", headers=_reef_headers(scenario, token))
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            catalog = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        sys.exit(f"reef-{adapter}: releases read failed ({exc.code}): {detail}")
    except OSError as exc:
        sys.exit(f"reef-{adapter}: reef unreachable at {upstream}: {exc}")
    rows = catalog.get("releases") if isinstance(catalog, Mapping) else None
    return [dict(row) for row in rows or [] if isinstance(row, Mapping)]


def _release_to_set_up(rows: Sequence[Mapping[str, Any]], release: str | None) -> Mapping[str, Any] | None:
    """The row ``setup`` reads: the one named, else the newest that is not pending; a pending release waits for a promote."""
    if release is not None:
        return next((row for row in rows if row.get("release_id") == release), None)
    return next((row for row in reversed(rows) if not row.get("pending")), None)


@dataclass
class _Setup:
    """What every ``setup`` form and ``update`` read first: the release to set up, its items and what is met so far.

    ``row`` is the catalog row to set up, None when nothing is served yet;
    ``requires`` its chain's union; ``recorded`` the check offs the release
    file holds and ``checked`` the working copy a form fills; ``values`` the
    env file; ``upstream`` and ``token`` reach reef for what comes next."""

    record: dict[str, Any]
    row: Mapping[str, Any] | None
    requires: list[dict[str, Any]]
    recorded: dict[str, dict[str, Any]]
    checked: dict[str, dict[str, Any]]
    values: dict[str, str]
    scenario: str
    upstream: str
    token: str | None

    @property
    def release_id(self) -> str | None:
        release = (self.row or {}).get("release_id")
        return release if isinstance(release, str) and release else None

    @property
    def label(self) -> str:
        """``release <id> `` for the messages; empty when the row names no id."""
        return f"release {self.release_id} " if self.release_id else ""

    def item(self, name: str) -> dict[str, Any] | None:
        return next((item for item in self.requires if item["name"] == name), None)

    def met(self, item: Mapping[str, Any]) -> bool:
        """Checked off with its check, or met by the machine itself: see :func:`_auto_met`."""
        if _met(item, self.checked.get(item["name"])):
            return True
        return _auto_met(item, self.values)

    def unmet(self) -> list[dict[str, Any]]:
        return [item for item in self.requires if not self.met(item)]

    def check_off(self, item: Mapping[str, Any]) -> None:
        """Record the item as met; the check rides beside the name, so a release that changes it asks again."""
        self.checked[item["name"]] = {"name": item["name"], "checked_at": time.time(), "check": item.get("check")}


def service_display_url(url: str) -> str:
    """Show the service location without URL credentials, query parameters, or fragments."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


def _load_setup(scenario: str, adapter: str, compose_dir: str, release: str | None, prog: str) -> _Setup | None:
    """Load setup state from the active session, or from the installed configuration outside that session.

    The release is ``release`` when named (a pending or trial release
    included, so its items are checked off before its install), else the
    newest that is not pending; what it requires is its chain's union, as
    the manifest lists it. A tree without a release file exits: it did not
    come through the install channel, so there is nothing to set up or
    update."""
    record = _read_release_info(compose_dir)
    if record is None:
        sys.exit(
            f"reef-{adapter}: no {HARNESS_RELEASE_FILE} release file at {Path(compose_dir).resolve().parent}: this "
            "tree did not come through reef's install channel, so there is no installed release to set up or update"
        )
    session_root = os.environ.get("REEF_HARNESS_DEST")
    session_service = os.environ.get("REEF_SERVICE_URL")
    session_scenario = os.environ.get("REEF_SCENARIO")
    # Keep a session's setup and update on the service it selected the release from,
    # even when another install has replaced this directory's wrapper or model binding.
    if (
        session_root
        and session_service
        and session_scenario
        and Path(session_root).resolve() == Path(compose_dir).resolve().parent
    ):
        upstream = _strip_v1(session_service.rstrip("/"))
        scenario = session_scenario
        token = os.environ.get("REEF_TOKEN") or None
    else:
        upstream = _reef_url_of(adapter, compose_dir)
        token = _reef_token(adapter, compose_dir)
    rows = _catalog(upstream, scenario, adapter, token)
    row = _release_to_set_up(rows, release)
    if row is None and release is not None:
        print(
            f"reef-{adapter} {prog}: no release {release} in the catalog\n"
            f"catalog: {service_display_url(upstream)} (scenario {scenario!r}). "
            "Check the service and scenario, then refresh the release list before retrying.",
            file=sys.stderr,
        )
        return None
    release_id = row.get("release_id") if row is not None else None
    requires = required_by(rows, release_id if isinstance(release_id, str) else None)
    recorded = {item["name"]: item for item in _named_items(record.get("setup"))}
    return _Setup(
        record, row, requires, recorded, dict(recorded), _read_env_file(compose_dir), scenario, upstream, token
    )


def _save_setup(compose_dir: str, state: _Setup) -> None:
    """Check off the ``env`` items whose variable is set and write the release file when the check offs changed."""
    for item in state.requires:
        if (
            item.get("kind") == "env"
            and not _met(item, state.checked.get(item["name"]))
            and _env_met(item, state.values)
        ):
            state.check_off(item)
    if state.checked != state.recorded:
        _write_release_info(compose_dir, {**state.record, "setup": list(state.checked.values())})


def _store_env_value(compose_dir: str, state: _Setup, item: Mapping[str, Any], value: str) -> None:
    """Store an ``env`` item's value in the env file, under the variable the item stands for."""
    state.values[_env_variable(item)] = value
    _write_env_file(compose_dir, state.values)


def _ask_env_value(item: Mapping[str, Any]) -> str:
    """Ask the person for an ``env`` item's value, without echo when the name looks like a credential; "" skips."""
    variable = _env_variable(item)
    secret = any(word in f"{item['name']} {variable}".upper() for word in _SECRET_WORDS)
    try:
        if secret:
            return getpass.getpass(f"    value for {variable} (not echoed; blank to skip): ").strip()
        return input(f"    value for {variable} (blank to skip): ").strip()
    except EOFError:
        return ""


def _settle_env(compose_dir: str, state: _Setup, item: Mapping[str, Any], yes: bool) -> bool:
    """Whether an ``env`` item is met: its variable set, else the value the person gives here, kept in the env file.

    ``yes`` is for scripts, which cannot answer, so it asks nothing."""
    if _env_met(item, state.values):
        print("    met", flush=True)
        return True
    value = "" if yes else _ask_env_value(item)
    if not value:
        print("    not set", flush=True)
        return False
    _store_env_value(compose_dir, state, item, value)
    print(f"    met (stored in {_env_file_path(compose_dir)})", flush=True)
    return True


def _settle_binary(item: Mapping[str, Any], yes: bool) -> bool:
    """Whether a ``binary`` item is met: its program on PATH, and its check, when it names one, run and passed.

    The look costs nothing and runs nothing, so it happens first: a check
    that names a program the machine does not have would only fail, and the
    person needs to install it either way."""
    name = str(item["name"])
    found = _binary_found(item)
    if found is None:
        print(f"    {name} is not on PATH; install it, then run setup again", flush=True)
        return False
    print(f"    found at {found}", flush=True)
    return True if not item.get("check") else _settle_command(item, yes)


def _settle_command(item: Mapping[str, Any], yes: bool) -> bool:
    """Whether a ``permission`` or ``service`` item is met once its check ran, after the person confirmed it."""
    check = item.get("check")
    if not check:
        print(f"    no check; mark it with --mark {item['name']} once it is done", flush=True)
        return False
    if not yes:
        print("    run it? [y/N] ", end="", flush=True)
        if sys.stdin.readline().strip().lower() not in ("y", "yes"):
            print("    skipped", flush=True)
            return False
    # The person read the command and said yes: it runs in their shell with their privileges, output and all.
    status = subprocess.run(check, shell=True).returncode
    print("    met" if status == 0 else f"    not met (exit {status})", flush=True)
    return status == 0


def setup(
    scenario: str,
    adapter: str,
    compose_dir: str,
    *,
    yes: bool = False,
    marks: Sequence[str] = (),
    release: str | None = None,
) -> int:
    """Check off what a release requires: list, ask for what is missing, record, 0 when every item is met.

    An unmet ``env`` item asks for its value and keeps it in the env file
    (``yes`` asks nothing); an unmet ``permission`` or ``service`` item runs
    its check once the person confirms it (``yes`` confirms); an unmet
    ``binary`` item is looked for on PATH, and its check, when it names one,
    runs on the same confirmation. ``marks`` are
    items checked off by hand, running nothing; an unknown name is exit 2.
    A check runs here and nowhere else."""
    state = _load_setup(scenario, adapter, compose_dir, release, "setup")
    if state is None:
        return 2
    if state.row is None:
        print(f"reef-{adapter} setup: no served release yet")
        return 0
    names = [item["name"] for item in state.requires]
    unknown = [name for name in marks if name not in names]
    if unknown:
        print(
            f"reef-{adapter} setup: no item named {', '.join(unknown)}; {state.label}requires {', '.join(names) or 'nothing'}",
            file=sys.stderr,
        )
        return 2
    if not state.requires:
        print(f"reef-{adapter} setup: {state.label}requires nothing")
        return 0
    print(f"reef-{adapter} setup: {state.label}requires {len(state.requires)} item(s)", flush=True)
    for item in state.requires:
        name = item["name"]
        print(f"  {_item_line(item)}", flush=True)
        if _met(item, state.checked.get(name)):
            print("    met (checked off)", flush=True)
            continue
        if name in state.checked:
            print("    the check changed since it was checked off", flush=True)
        if item.get("prompt"):
            print(f"    {item['prompt']}", flush=True)
        if name in marks:
            print("    met (marked by hand)", flush=True)
        elif item.get("kind") == "env":
            if not _settle_env(compose_dir, state, item, yes):
                continue
        elif item.get("kind") == "binary":
            if not _settle_binary(item, yes):
                continue
        elif not _settle_command(item, yes):
            continue
        state.check_off(item)
    _save_setup(compose_dir, state)
    unmet = [item["name"] for item in state.unmet()]
    if unmet:
        print(f"reef-{adapter} setup: {len(unmet)} item(s) not met: {', '.join(unmet)}")
        return 1
    print(f"reef-{adapter} setup: every item is met; reef-{adapter} update installs the release")
    return 0


def setup_json(scenario: str, adapter: str, compose_dir: str, *, release: str | None = None) -> int:
    """Print the release to set up and its items, each with whether it is met, as one JSON object; runs nothing.

    0, or 2 when ``release`` is not in the catalog; nothing served yet is
    a null release with no items."""
    state = _load_setup(scenario, adapter, compose_dir, release, "setup")
    if state is None:
        return 2
    items = [
        {
            "name": item["name"],
            "kind": item.get("kind"),
            "check": item.get("check"),
            "prompt": item.get("prompt"),
            "met": state.met(item),
        }
        for item in state.requires
    ]
    print(json.dumps({"release_id": state.release_id, "items": items}))
    return 0


def setup_set(scenario: str, adapter: str, compose_dir: str, assignment: str, *, release: str | None = None) -> int:
    """Store the value of the ``env`` item in ``NAME=VALUE`` in the env file and check it off; 0, or 2, said on stderr.

    The value is an argument, never shell source, and stays one line; an
    unknown name or a non-env item is refused."""
    name, separator, value = assignment.partition("=")
    value = value.strip()
    if not separator or not name:
        print(f"reef-{adapter} setup: --set takes NAME=VALUE", file=sys.stderr)
        return 2
    if not value or "\n" in value or "\r" in value:
        print(f"reef-{adapter} setup: the value for {name} must be one non-empty line", file=sys.stderr)
        return 2
    state = _load_setup(scenario, adapter, compose_dir, release, "setup")
    if state is None:
        return 2
    item = state.item(name)
    if item is None:
        names = ", ".join(required["name"] for required in state.requires) or "nothing"
        print(f"reef-{adapter} setup: no item named {name}; {state.label}requires {names}", file=sys.stderr)
        return 2
    if item.get("kind") != "env":
        print(
            f"reef-{adapter} setup: {name} is a {item.get('kind')} item; --set stores the value of an env item, "
            "--run runs a check",
            file=sys.stderr,
        )
        return 2
    _store_env_value(compose_dir, state, item, value)
    state.check_off(item)
    _save_setup(compose_dir, state)
    print(f"reef-{adapter} setup: {name} stored in {_env_file_path(compose_dir)}")
    return 0


def setup_run(scenario: str, adapter: str, compose_dir: str, name: str, *, release: str | None = None) -> int:
    """Run one item's check without asking, the caller having confirmed it, and check it off when it passes.

    0 when the item is met, 1 when it is not, 2 for a name the release does
    not require; an ``env`` item's check is reading its variable, and a
    ``binary`` item's is looking for its program on PATH, before the check
    it names, if any, runs."""
    state = _load_setup(scenario, adapter, compose_dir, release, "setup")
    if state is None:
        return 2
    item = state.item(name)
    if item is None:
        names = ", ".join(required["name"] for required in state.requires) or "nothing"
        print(f"reef-{adapter} setup: no item named {name}; {state.label}requires {names}", file=sys.stderr)
        return 2
    if item.get("kind") == "env":
        met = _env_met(item, state.values)
        detail = "met" if met else f"not met ({_env_variable(item)} is not set; --set {name}=VALUE stores it)"
    elif item.get("kind") == "binary" and _binary_found(item) is None:
        met, detail = False, f"not met ({name} is not on PATH; install it, then run this again)"
    elif item.get("kind") == "binary" and not item.get("check"):
        met, detail = True, f"met ({name} at {_binary_found(item)})"
    elif not item.get("check"):
        met, detail = False, f"not met (no check; --mark {name} checks it off by hand)"
    else:
        # The caller confirmed the command: it runs in the person's shell with their privileges, output and all.
        status = subprocess.run(str(item["check"]), shell=True).returncode
        met, detail = status == 0, "met" if status == 0 else f"not met (exit {status})"
    if met:
        state.check_off(item)
    _save_setup(compose_dir, state)
    print(f"reef-{adapter} setup: {name} {detail}")
    return 0 if met else 1


def _run_install_script(script: bytes, install_root: Path, token: str | None) -> int:
    """Run a fetched install script with ``bash`` for ``install_root``; its exit status, 127 when bash cannot run.

    ``REEF_TOKEN`` rides in the script's environment, so the binding it
    writes keeps the token the wrapper reaches reef with."""
    env = os.environ.copy()
    if token:
        env["REEF_TOKEN"] = token
    with tempfile.NamedTemporaryFile(prefix="reef-harness-install-", suffix=".sh", delete=False) as handle:
        handle.write(script)
        path = Path(handle.name)
    try:
        return subprocess.run(["bash", str(path), str(install_root)], env=env).returncode
    except OSError as exc:
        print(f"reef-harness: cannot run bash: {exc}", file=sys.stderr)
        return 127
    finally:
        path.unlink(missing_ok=True)


def update(scenario: str, adapter: str, compose_dir: str, *, release: str | None = None) -> int:
    """Install the served release, or ``release``, into this install root through the install script; 0 once it ran.

    The script comes from ``GET /reef/harness/install`` with the token and
    the scenario header and runs with ``bash`` and the install root as its
    destination; 1 when the fetch or the script fails. While the release
    requires an item that is not met the items are printed and nothing is
    fetched: 3, the script would refuse anyway, this says why first."""
    state = _load_setup(scenario, adapter, compose_dir, release, "update")
    if state is None:
        return 1
    if state.row is None:
        print(f"reef-{adapter} update: no served release yet", file=sys.stderr)
        return 1
    # The env items the environment or the env file meets are checked off, so the script's evaluation sees them met.
    _save_setup(compose_dir, state)
    unmet = state.unmet()
    if unmet:
        print(f"reef-{adapter} update: {state.label}requires setup first:", file=sys.stderr)
        for item in unmet:
            print(f"  {_item_line(item)}", file=sys.stderr)
        named = f" --release {release}" if release is not None else ""
        print(f"reef-{adapter} update: run reef-{adapter} setup{named}, then update again", file=sys.stderr)
        return 3
    query = f"adapter={urllib.parse.quote(adapter, safe='')}"
    if release is not None:
        query += f"&release_id={urllib.parse.quote(release, safe='')}"
    req = urllib.request.Request(
        f"{state.upstream}/reef/harness/install?{query}", headers=_reef_headers(state.scenario, state.token)
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            script = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"reef-{adapter} update: install script read failed ({exc.code}): {detail}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"reef-{adapter} update: reef unreachable at {state.upstream}: {exc}", file=sys.stderr)
        return 1
    status = _run_install_script(script, Path(compose_dir).resolve().parent, state.token)
    if status != 0:
        print(f"reef-{adapter} update: the install script exited {status}", file=sys.stderr)
        return 1
    print(f"reef-{adapter} update: installed release {_installed_release(compose_dir) or state.release_id}")
    return 0


def _doctor_row(ok: bool, label: str, value: str) -> str:
    return f"{'ok' if ok else '!!'}  {label:<12} {value}"


def doctor(scenario: str, adapter: str, compose_dir: str, binary: str) -> int:
    """One line per thing an install needs; 0 when every line holds, 1 otherwise.

    Every check exists somewhere already (an install warning, a run time
    warning, a route error); this is the one place that runs them all and
    says which failed. A release awaiting a review gets a line of its own,
    the one ``evolve --wait`` prints, since a person who runs this is
    usually asking what happened to their request."""
    rows: list[tuple[bool, str, str]] = []
    catalog: list[Mapping[str, Any]] | None = None
    token: str | None = None
    prog = f"reef-{adapter}"
    try:
        from reef.core.version import __version__

        rows.append((True, "interpreter", f"{sys.executable} (reef {__version__}, reef-client importable)"))
    except Exception as exc:  # pragma: no cover - the wrapper itself imports both
        rows.append((False, "interpreter", f"{sys.executable} does not import reef: {exc}"))
    try:
        upstream = _extract_reef_url(adapter, Path(compose_dir))
    except WrapperError as exc:
        upstream = None
        rows.append((False, "service", f"binding unreadable: {exc}"))
    if upstream is None:
        rows.append((False, "service", "no Reef URL in the tree's model binding files"))
    else:
        upstream = _strip_v1(upstream)
        token = _reef_token(adapter, compose_dir)
        # The release catalog, not /reef/status: every client of a harness can read it, direct or through the
        # API platform, which does not expose the service wide status.
        req = urllib.request.Request(f"{upstream}/reef/harness/releases", headers=_reef_headers(scenario, token))
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                listed = json.loads(response.read()).get("releases")
            catalog = [row for row in listed if isinstance(row, Mapping)] if isinstance(listed, list) else []
            rows.append((True, "service", f"{upstream} answers, token {'accepted' if token else 'not needed'}"))
        except urllib.error.HTTPError as exc:
            rows.append(
                (False, "service", f"{upstream} answered {exc.code}: {exc.read().decode(errors='replace')[:120]}")
            )
        except (OSError, ValueError) as exc:
            rows.append((False, "service", f"{upstream} unreachable: {exc}"))
    if Path(binary).is_file():
        try:
            version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=20)
            first = (version.stdout or version.stderr).strip().splitlines()
            rows.append((version.returncode == 0, "binary", f"{binary} ({first[0] if first else 'no output'})"))
        except (OSError, subprocess.TimeoutExpired) as exc:
            rows.append((False, "binary", f"{binary} did not run: {exc}"))
    else:
        rows.append((False, "binary", f"{binary} missing; rerun the install"))
    for command, package in get_adapter(adapter).client_tools:
        found = shutil.which(command)
        rows.append(
            (found is not None, "tool", f"{command} {'at ' + found if found else 'missing: install ' + package}")
        )
    # The programs the installed release names, looked for the same way. A release's check is a command it
    # wrote, and one runs only where the person confirmed it, in setup; doctor says what is there, and nothing more.
    for item in _named_items((_read_release_info(compose_dir) or {}).get("requires")):
        if item.get("kind") != "binary":
            continue
        found = _binary_found(item)
        hint = f": {item['prompt']}" if item.get("prompt") else f"; {prog} setup says what it is for"
        rows.append((found is not None, "program", f"{item['name']} {'at ' + found if found else 'missing' + hint}"))
    installed = _installed_release(compose_dir)
    if installed is None:
        rows.append(
            (
                False,
                "release",
                f"no {HARNESS_RELEASE_FILE} beside the tree; this tree did not come through the install",
            )
        )
    elif catalog is not None:
        head = next((row.get("release_id") for row in reversed(catalog) if not row.get("pending")), None)
        if head == installed:
            rows.append((True, "release", f"{installed[:8]} installed, the served head"))
        else:
            rows.append(
                (
                    True,
                    "release",
                    f"{installed[:8]} installed; served head {str(head)[:8]}, the next session offers it",
                )
            )
    else:
        rows.append((True, "release", f"{installed[:8]} installed"))
    # A release held for review is not something the install needs, but it is what the person is waiting on.
    if catalog is not None and upstream is not None:
        for step, waiting in _waiting_for_review(catalog):
            page = _step_page_link(upstream, scenario, token, step)
            rows.append((True, "review", f"{str(waiting.get('release_id'))[:8]} waits for your review: {page}"))
    for ok, label, value in rows:
        print(_doctor_row(ok, label, value))
    return 0 if all(ok for ok, _, _ in rows) else 1


def _waiting_for_review(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, Mapping[str, Any]]]:
    """The pending rows no later promote row names, each with its step: held for a person, served to nobody."""
    promoted = {row.get("rollback_target_release_id") for row in rows if row.get("operation") == "promote"}
    return [
        (step, row) for step, row in enumerate(rows) if row.get("pending") and row.get("release_id") not in promoted
    ]


def pinned_version_line(adapter: str) -> str:
    """What the wrapper says instead of running the binary's own updater."""
    install = get_adapter(adapter).install
    pinned = f"{adapter} {install.version}" if install is not None else adapter
    return f"reef pins {pinned} and does not run {adapter}'s own updater; reef-{adapter} update installs the served release"


def _usage(adapter: str) -> str:
    """The wrapper's own subcommands, printed before the agent's help."""
    prog = f"reef-{adapter}"
    descriptor = get_adapter(adapter)
    lines = [
        f"{prog}: run {adapter} through reef's capture proxy, or one of",
        f"  {prog} report --score S [--feedback TEXT] [--per-receipt]      score the last run's receipts",
        f'  {prog} evolve "<what it should do>" [--wait] [--timeout SECONDS]   ask for a harness change',
        f"  {prog} page <step> [--print]                                     fetch a step's page and open it",
        f"  {prog} doctor                                                     check what the install needs",
        f"  {prog} setup [--yes] [--mark NAME] [--release ID]                 check off what a release requires",
        f"  {prog} setup --json | --set NAME=VALUE | --run NAME [--release ID]  one item at a time, for scripts",
        f"  {prog} update [--release ID]                                       install the served release here",
    ]
    if descriptor.client_updater_args:
        lines.append(f"{prog} {' | '.join(descriptor.client_updater_args)}: {pinned_version_line(adapter)}.")
    if descriptor.client_args:
        version_flags = ", ".join(descriptor.client_version_args)
        exception = f", unless the first is one of {version_flags}" if version_flags else ""
        lines.append(
            f"Anything else runs {adapter} with {shlex.join(descriptor.client_args)} ahead of the same arguments"
            f"{exception}; --help and -h print its help after this."
        )
    else:
        lines.append(f"Anything else runs {adapter} with the same arguments; --help and -h print its help after this.")
    return "\n".join(lines)


def main() -> None:
    binary = os.environ.get("REEF_HARNESS_BINARY")
    compose = os.environ.get("REEF_HARNESS_COMPOSE")
    scenario = os.environ.get("REEF_HARNESS_SCENARIO")
    adapter = os.environ.get("REEF_HARNESS_ADAPTER")
    env_var = os.environ.get("REEF_HARNESS_ENV_VAR")
    if not binary or not compose or not scenario or not adapter or not env_var:
        sys.exit(
            "reef-harness: missing REEF_HARNESS_BINARY/REEF_HARNESS_COMPOSE/REEF_HARNESS_SCENARIO"
            "/REEF_HARNESS_ADAPTER/REEF_HARNESS_ENV_VAR"
        )

    args = sys.argv[1:]
    if args and args[0] in ("--help", "-h", "help"):
        print(_usage(adapter))
        if args[0] == "help":
            return
        # --help and -h go on to the agent below, so its own help follows.
    if args and args[0] == "report":
        parser = argparse.ArgumentParser(prog=f"reef-{adapter} report")
        parser.add_argument("--score", type=float, required=True)
        parser.add_argument("--feedback", default="")
        parser.add_argument(
            "--per-receipt",
            action="store_true",
            help="send one report per captured receipt instead of one for the run",
        )
        ns = parser.parse_args(args[1:])
        report(scenario, adapter, ns.score, ns.feedback, per_receipt=ns.per_receipt)
    elif args and args[0] in ("evolve", "harness"):
        parser = argparse.ArgumentParser(prog=f"reef-{adapter} evolve")
        parser.add_argument("request", nargs="*", help="what the harness should do, in plain words")
        parser.add_argument("--wait", action="store_true", help="stay until the step settles and print its result")
        parser.add_argument(
            "--timeout", type=float, default=1800.0, metavar="SECONDS", help="how long --wait waits (default 1800)"
        )
        # Intermixed, so the flags read the same before and after the request.
        ns = parser.parse_intermixed_args(args[1:])
        sys.exit(harness(scenario, adapter, compose, " ".join(ns.request), wait=ns.wait, timeout_s=ns.timeout))
    elif args and args[0] == "page":
        parser = argparse.ArgumentParser(prog=f"reef-{adapter} page")
        parser.add_argument("step", type=step_of_version, help="the version, as /versions lists it: v3 or 3")
        parser.add_argument("--print", dest="print_only", action="store_true", help="print the path; open nothing")
        ns = parser.parse_args(args[1:])
        sys.exit(page(scenario, adapter, compose, ns.step, open_page=not ns.print_only))
    elif args and args[0] == "doctor":
        argparse.ArgumentParser(prog=f"reef-{adapter} doctor").parse_args(args[1:])
        sys.exit(doctor(scenario, adapter, compose, binary))
    elif args and args[0] == "setup":
        parser = argparse.ArgumentParser(prog=f"reef-{adapter} setup")
        parser.add_argument("--yes", action="store_true", help="run every check without asking (for scripts)")
        parser.add_argument(
            "--mark", action="append", default=[], metavar="NAME", help="check an item off by hand, running nothing"
        )
        parser.add_argument(
            "--release", default=None, metavar="ID", help="the release to set up (default: the newest not pending)"
        )
        form = parser.add_mutually_exclusive_group()
        form.add_argument(
            "--json", action="store_true", help="print the items and whether each is met, as JSON; run nothing"
        )
        form.add_argument(
            "--set", dest="assignment", default=None, metavar="NAME=VALUE", help="store an env item's value"
        )
        form.add_argument(
            "--run", dest="run_name", default=None, metavar="NAME", help="run one item's check without asking"
        )
        ns = parser.parse_args(args[1:])
        chosen = {"release": ns.release} if ns.release is not None else {}
        if ns.json:
            sys.exit(setup_json(scenario, adapter, compose, **chosen))
        if ns.assignment is not None:
            sys.exit(setup_set(scenario, adapter, compose, ns.assignment, **chosen))
        if ns.run_name is not None:
            sys.exit(setup_run(scenario, adapter, compose, ns.run_name, **chosen))
        sys.exit(setup(scenario, adapter, compose, yes=ns.yes, marks=tuple(ns.mark), **chosen))
    elif args and args[0] == "update":
        parser = argparse.ArgumentParser(prog=f"reef-{adapter} update")
        parser.add_argument(
            "--release", default=None, metavar="ID", help="the release to install (default: the served head)"
        )
        ns = parser.parse_args(args[1:])
        chosen = {"release": ns.release} if ns.release is not None else {}
        sys.exit(update(scenario, adapter, compose, **chosen))
    elif args and args[0] in get_adapter(adapter).client_updater_args:
        # The binary's own updater would replace the version reef pins, so it never runs from here.
        sys.exit(f"reef-{adapter}: {pinned_version_line(adapter)}")
    else:
        run_agent(binary, compose, scenario, adapter, env_var, args)


if __name__ == "__main__":
    main()
