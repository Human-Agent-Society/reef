"""Reef harness wrapper: a capture proxy between the agent binary and Reef.

When invoked with agent arguments (e.g. ``reef-pi -p "fix the bug"``):

  1. Starts a local capture proxy (``reef_client.serve``) that forwards to
     Reef, injecting ``x-reef-scenario`` so the user's agent binary never
     needs to know about Reef headers.
  2. Rewrites the provider config in a temp copy of the composition to point
     the agent at the proxy instead of Reef directly.
  3. Runs the agent binary as a subprocess.
  4. After the agent exits, persists the captured receipts (the
     ``x-reef-agent-record-id`` values from each response) to disk.

When invoked with ``report`` (e.g. ``reef-pi report --score 0.0 --feedback "..."``):

  1. Claims the oldest pending run's persisted receipts.
  2. POSTs a report to Reef with all captured receipts as ``references``
     (one trajectory sample), or one report per receipt with ``--per-receipt``.
  3. Clears the persisted receipts.

Env vars (baked into the wrapper at install time):

  ``REEF_HARNESS_BINARY``    absolute path to the agent binary
  ``REEF_HARNESS_COMPOSE``   absolute path to the composition directory
  ``REEF_HARNESS_SCENARIO``  the reef scenario name
  ``REEF_HARNESS_ADAPTER``   adapter name (any descriptor whose env relocates its composition
                             with a {root}/<dir> entry; terminus has none and gets no wrapper)
  ``REEF_HARNESS_ENV_VAR``   the env var that relocates the composition

Optional:

  ``REEF_TOKEN``  bearer token for the reef service (if auth is enabled)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

from reef_client.serve import CaptureStore, ServeConfig, build_handler

from reef.harness.adapters import get_adapter
from reef.harness.descriptor import AdapterDescriptor


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


def _bindings(descriptor: AdapterDescriptor) -> list[_Binding]:
    found: dict[tuple[str, tuple[str, ...]], _Binding] = {}
    for templates in descriptor.model_binding.values():
        for node in templates:
            target = str(node.get("target", "primary"))
            stack: list[tuple[tuple[str, ...], Any]] = [((), node.get("data", {}))]
            while stack:
                path, value = stack.pop()
                if isinstance(value, Mapping):
                    stack.extend((path + (str(key),), item) for key, item in value.items())
                elif isinstance(value, str) and "{base_url}" in value:
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
        except OSError:  # noqa: PERF203
            time.sleep(0.05)
    return False


#: The release sidecar the install script and harness_pull write at the tree
#: root; version_check.ts reads the same name.
HARNESS_RELEASE_SIDECAR = ".reef-harness-release"


def _installed_release(compose_dir: str) -> str | None:
    """The release id of the installed tree, from the sidecar beside it."""
    sidecar = Path(compose_dir).parent / HARNESS_RELEASE_SIDECAR
    try:
        release = json.loads(sidecar.read_text(encoding="utf-8")).get("release_id")
    except (OSError, json.JSONDecodeError):
        return None
    return release if isinstance(release, str) and release else None


def run_agent(binary: str, compose_dir: str, scenario: str, adapter: str, env_var: str, args: list[str]) -> None:
    try:
        reef_url = _extract_reef_url(adapter, Path(compose_dir))
    except WrapperError as exc:
        sys.exit(f"reef-{adapter}: {exc}")
    if reef_url is None:
        sys.exit(f"reef-{adapter}: no Reef URL in the tree's model binding files")
    upstream = _strip_v1(reef_url)

    store = CaptureStore()
    override: dict[str, str] = {"x-reef-scenario": scenario}
    release = _installed_release(compose_dir)
    if release:
        # The opaque tag channel: the record keeps which release answered on
        # the client, under metadata.tags.release.
        override["x-reef-tag-release"] = release
    token = os.environ.get("REEF_TOKEN")
    if token:
        override["authorization"] = f"Bearer {token}"

    config = ServeConfig(
        upstream=upstream,
        listen_port=0,
        override_headers=override,
        # The inference paths Reef serves; a receipt rides on each. The proxy matches the path with its
        # query, and the Anthropic SDK posts /v1/messages?beta=true under beta headers, so that form is listed.
        # Reef has no Responses route yet, so a codex tree bound to Reef sends its calls to a path nothing answers.
        capture_paths=("/v1/chat/completions", "/v1/messages", "/v1/messages?beta=true"),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(config, store))
    proxy_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if not _wait_for_proxy(proxy_port):
        sys.exit(f"reef-{adapter}: capture proxy failed to start")

    temp_dir = _create_temp_composition(adapter, compose_dir, proxy_port)
    env = os.environ.copy()
    env[env_var] = temp_dir
    # The update notice extension needs the service address, the scenario,
    # and the true install root; the relocated temp copy carries none of them.
    env["REEF_SERVICE_URL"] = upstream
    env["REEF_SCENARIO"] = scenario
    env["REEF_HARNESS_DEST"] = str(Path(compose_dir).resolve().parent)
    if adapter == "native":
        # The loop's session log outlives the temp copy: it lands beside the installed tree.
        env.setdefault("REEF_NATIVE_SESSION_DIR", str(Path(compose_dir).resolve() / "sessions"))

    try:
        result = subprocess.run([binary, *args], env=env)
    finally:
        _publish_captures(upstream, scenario, store.snapshot())
        server.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)

    sys.exit(result.returncode)


def report(scenario: str, adapter: str, score: float, feedback: str, per_receipt: bool = False) -> None:
    while True:
        claim = _claim_captures(scenario)
        if claim is None:
            sys.exit(f"reef-{adapter}: no captured receipts for scenario {scenario!r}")
        pending_file, captures_file = claim

        try:
            data = json.loads(captures_file.read_text(encoding="utf-8"))
            reef_url = data["reef_url"]
            receipts = [t["receipt"] for t in data["turns"] if t.get("receipt")]
        except BaseException:
            os.replace(captures_file, pending_file)
            raise
        if receipts:
            break
        captures_file.unlink()

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "x-reef-scenario": scenario,
    }
    token = os.environ.get("REEF_TOKEN")
    if token:
        headers["authorization"] = f"Bearer {token}"

    # One report referencing the whole run batches as one trajectory sample;
    # --per-receipt sends the same score against each receipt on its own.
    reference_lists = [[receipt] for receipt in receipts] if per_receipt else [receipts]
    release = _installed_release(os.environ.get("REEF_HARNESS_COMPOSE", ""))
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
    else:
        run_agent(binary, compose, scenario, adapter, env_var, args)


if __name__ == "__main__":
    main()
