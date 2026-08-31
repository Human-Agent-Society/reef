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

  1. Reads the persisted receipts from the last agent run.
  2. POSTs a report to Reef with all captured receipts as ``references``.
  3. Clears the persisted receipts.

Env vars (baked into the wrapper at install time):

  ``REEF_HARNESS_BINARY``    absolute path to the agent binary
  ``REEF_HARNESS_COMPOSE``   absolute path to the composition directory
  ``REEF_HARNESS_SCENARIO``  the reef scenario name
  ``REEF_HARNESS_ADAPTER``   adapter name (pi, opencode, ...)
  ``REEF_HARNESS_ENV_VAR``   the env var that relocates the composition

Optional:

  ``REEF_TOKEN``  bearer token for the reef service (if auth is enabled)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from reef_client.serve import CaptureStore, ServeConfig, build_handler


def _captures_dir() -> Path:
    return Path(os.environ.get("REEF_HARNESS_CAPTURES_DIR", str(Path.home() / ".reef" / "captures")))


def _extract_reef_url(adapter: str, compose_dir: Path) -> str | None:
    """Extract the upstream Reef base URL from the composition's config."""
    if adapter == "pi":
        models_path = compose_dir / "models.json"
        if not models_path.exists():
            return None
        models = json.loads(models_path.read_text(encoding="utf-8"))
        for provider in models.get("providers", {}).values():
            url = provider.get("baseUrl")
            if url:
                return url.rstrip("/")
    elif adapter == "opencode":
        config_path = compose_dir / "opencode.json"
        if not config_path.exists():
            return None
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for provider in config.get("provider", {}).values():
            url = provider.get("options", {}).get("baseURL")
            if url:
                return url.rstrip("/")
    return None


def _strip_v1(url: str) -> str:
    return url[:-3] if url.endswith("/v1") else url


def _rewrite_config(adapter: str, compose_dir: Path, temp_dir: Path, proxy_port: int) -> None:
    """Copy the config file into the temp dir with the proxy URL substituted."""
    proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
    if adapter == "pi":
        src = compose_dir / "models.json"
        if not src.exists():
            return
        dst = temp_dir / "models.json"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        models = json.loads(src.read_text(encoding="utf-8"))
        for provider in models.get("providers", {}).values():
            if "baseUrl" in provider:
                provider["baseUrl"] = proxy_url
        dst.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    elif adapter == "opencode":
        src = compose_dir / "opencode.json"
        if not src.exists():
            return
        dst = temp_dir / "opencode.json"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        config = json.loads(src.read_text(encoding="utf-8"))
        for provider in config.get("provider", {}).values():
            options = provider.get("options", {})
            if "baseURL" in options:
                options["baseURL"] = proxy_url
        dst.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _create_temp_composition(adapter: str, compose_dir: str, proxy_port: int) -> str:
    """Symlink the composition into a temp dir, overriding the config file."""
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


def run_agent(binary: str, compose_dir: str, scenario: str, adapter: str, env_var: str, args: list[str]) -> None:
    reef_url = _extract_reef_url(adapter, Path(compose_dir))
    if reef_url is None:
        sys.exit(f"reef-{adapter}: no provider baseUrl found in composition")
    upstream = _strip_v1(reef_url)

    store = CaptureStore()
    override: dict[str, str] = {"x-reef-scenario": scenario}
    token = os.environ.get("REEF_TOKEN")
    if token:
        override["authorization"] = f"Bearer {token}"

    config = ServeConfig(
        upstream=upstream,
        listen_port=0,
        override_headers=override,
        capture_paths=("/v1/chat/completions",),
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

    try:
        result = subprocess.run([binary, *args], env=env)
    finally:
        captures_dir = _captures_dir()
        captures_dir.mkdir(parents=True, exist_ok=True)
        captures_file = captures_dir / f"{scenario}.json"
        captures_file.write_text(
            json.dumps({"reef_url": upstream, "scenario": scenario, "turns": store.snapshot()}),
            encoding="utf-8",
        )
        server.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)

    sys.exit(result.returncode)


def report(scenario: str, adapter: str, score: float, feedback: str) -> None:
    captures_file = _captures_dir() / f"{scenario}.json"
    if not captures_file.exists():
        sys.exit(f"reef-{adapter}: no captured receipts for scenario {scenario!r}")

    data = json.loads(captures_file.read_text(encoding="utf-8"))
    reef_url = data["reef_url"]
    receipts = [t["receipt"] for t in data["turns"] if t.get("receipt")]
    if not receipts:
        sys.exit(f"reef-{adapter}: no captured receipts for scenario {scenario!r}")

    payload = json.dumps({"score": score, "feedback": feedback, "references": receipts}).encode()
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "x-reef-scenario": scenario,
    }
    token = os.environ.get("REEF_TOKEN")
    if token:
        headers["authorization"] = f"Bearer {token}"

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
        body = exc.read().decode(errors="replace")
        sys.exit(f"reef-{adapter}: report failed ({exc.code}): {body}")

    captures_file.unlink()
    print(f"reef-{adapter}: reported {len(receipts)} receipt(s) to {scenario}")


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
        ns = parser.parse_args(args[1:])
        report(scenario, adapter, ns.score, ns.feedback)
    else:
        run_agent(binary, compose, scenario, adapter, env_var, args)


if __name__ == "__main__":
    main()
