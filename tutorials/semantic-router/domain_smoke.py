"""Opt-in functional smoke of domain-classifier routing through a running ``reef serve`` and the real SR.

It is not part of the test suite and never starts or stops a service. ``manifests`` writes two manifests from
``manifest.example.yaml``: both add the pinned Vela Domain classifier at ``--model-directory`` and a
``math_request`` domain rule to each Scenario; the second only sends ``code-opd``'s rule to its student.
``check`` then sends requests to a ``reef serve`` started with the first manifest and the CPU stand-ins of
``fake_provider.py``, whose answers name the model that answered. It checks, for two Scenarios: a domain rule
match, the default target, an explicit pin, a refused cross-Scenario pin, JSON and SSE receipts, feedback
linked to each Scenario's receipts, and a policy update by ``SIGHUP``. It prints a JSON summary and exits 1
if any check fails.

The prompts are the fixed ones the real classifier sorted into ``math`` and ``biology`` in a direct SR probe.
They test that routing works, not how accurate the classifier is; a classifier that predicts another label for
them fails the domain or default check, which then says so.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.request
from email.message import Message
from http.client import HTTPResponse
from pathlib import Path

import yaml

#: vllm-sr/Vela-1.0-Encoder-307M-Domain at revision 736c049c8710e16011c0216ff1bd79c5a1c399db (Apache-2.0).
VELA_DOMAIN_FILES = {
    "config.json": "sha256:2741d5ac33b7c56fd3c716e6b434e4616d6f0d943b9f0634fbf3bb1e7dd8cafd",
    "tokenizer.json": "sha256:609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
    "model.safetensors": "sha256:be89c0202dc630796e0487b2cb661103782fc2d8aa181a8f92bd67b697d089a4",
    "category_mapping.json": "sha256:48ce5b71a0d3479e460af7e5a39dd05a7c66728380f201a1f794b335de915b1a",
}
MATH_PROMPT = (
    "Solve the quadratic equation x squared minus five x plus six equals zero. "
    "Explain the algebraic factorization and calculate both roots."
)
BIOLOGY_PROMPT = (
    "Explain photosynthesis in plant cells and how chloroplasts convert sunlight, carbon dioxide and water "
    "into glucose."
)
EXAMPLE = Path(__file__).resolve().with_name("manifest.example.yaml")


def write_manifests(model_directory: Path, output: Path) -> None:
    manifest = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    manifest["deployment"]["domain_classifier"] = {
        "model_directory": os.path.realpath(model_directory),
        "threshold": 0.5,
        "files": VELA_DOMAIN_FILES,
    }
    for scenario in ("code-opd", "math-opd"):
        rule = {"name": "math_request", "domains": ["math"], "target": scenario.split("-")[0] + "-teacher"}
        manifest["scenarios"][scenario]["rules"].append(rule)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    manifest["scenarios"]["code-opd"]["rules"][-1]["target"] = "code-student"
    (output / "manifest.updated.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(json.dumps({"manifest": str(output / "manifest.yaml"), "updated": str(output / "manifest.updated.yaml")}))


class NoRedirects(urllib.request.HTTPRedirectHandler):
    """A stateless handler that refuses redirects to keep the Reef token at the requested endpoint."""

    def redirect_request(
        self, req: urllib.request.Request, fp: HTTPResponse, code: int, msg: str, headers: Message, newurl: str
    ) -> None:
        """Return no replacement request; urllib reports the redirect as an HTTP error."""
        return


class SmokeClient:
    """Requests to one Reef service with one service token; ``checks`` collects every outcome in order."""

    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirects())
        self.checks: list[dict[str, object]] = []

    def post(self, path: str, scenario: str, body: dict[str, object]) -> tuple[int, str, dict[str, str]]:
        """Status, body text and headers, with lowercase header names."""
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": "Bearer " + self.token,
                "x-reef-scenario": scenario,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=60) as response:
                headers = {name.lower(): value for name, value in response.headers.items()}
                return response.status, response.read().decode(), headers
        except urllib.error.HTTPError as error:
            headers = {name.lower(): value for name, value in error.headers.items()}
            return error.code, error.read().decode(), headers

    def chat(self, scenario: str, model: str, prompt: str, stream: bool) -> tuple[int, str, str | None]:
        """Status, answer text and receipt of one chat completion; the SSE receipt must directly precede [DONE]."""
        body = {"model": model, "stream": stream, "messages": [{"role": "user", "content": prompt}]}
        status, text, headers = self.post("/v1/chat/completions", scenario, body)
        if status != 200:
            return status, text, None
        if not stream:
            answer = json.loads(text)["choices"][0]["message"]["content"]
            return status, answer, headers.get("x-reef-agent-record-id")
        events = [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]
        if len(events) < 2 or events[-1] != "[DONE]":
            return status, text, None
        receipt = json.loads(events[-2]).get("reef", {}).get("agent_record_id")
        answer = "".join(
            choice.get("delta", {}).get("content") or ""
            for event in events[:-2]
            for choice in json.loads(event).get("choices", [])
        )
        return status, answer, receipt

    def expect(self, name: str, passed: bool, **details: object) -> None:
        self.checks.append({"check": name, "passed": passed, **details})


def run_matrix(client: SmokeClient, code_math_target: str) -> dict[str, list[str]]:
    """One round of requests; returns each Scenario's receipts."""
    receipts: dict[str, list[str]] = {"code-opd": [], "math-opd": []}
    cases = [
        ("code-opd", "auto", MATH_PROMPT, False, code_math_target, "domain rule"),
        ("code-opd", "auto", MATH_PROMPT, True, code_math_target, "domain rule"),
        ("code-opd", "auto", BIOLOGY_PROMPT, False, "code-student", "default"),
        ("code-opd", "code-teacher", BIOLOGY_PROMPT, True, "code-teacher", "explicit pin"),
        ("math-opd", "auto", MATH_PROMPT, True, "math-teacher", "domain rule"),
        ("math-opd", "auto", BIOLOGY_PROMPT, False, "math-student", "default"),
    ]
    for scenario, model, prompt, stream, target, kind in cases:
        status, answer, receipt = client.chat(scenario, model, prompt, stream)
        client.expect(
            f"{scenario} {kind} {'SSE' if stream else 'JSON'}",
            status == 200 and answer == f"{target} answered." and receipt is not None,
            status=status,
            answer=answer[:200],
            expected=f"{target} answered.",
        )
        if receipt is not None:
            receipts[scenario].append(receipt)
    status, text, _ = client.chat("code-opd", "math-teacher", MATH_PROMPT, False)
    client.expect("cross-Scenario pin refused", status == 403, status=status, body=text[:200])
    return receipts


def check(arguments: argparse.Namespace) -> int:
    if arguments.reef_pid <= 1:
        raise ValueError("--reef-pid must identify one reef serve process with a PID greater than 1")
    client = SmokeClient(arguments.url, os.environ["REEF_TOKEN"])
    receipts = run_matrix(client, "code-teacher")
    # SIGHUP applies the edited manifest; requests get 503 until the new policy is published.
    with tempfile.NamedTemporaryFile(
        "w", dir=arguments.manifest.parent, delete=False, encoding="utf-8"
    ) as replacement:
        replacement.write(arguments.updated_manifest.read_text(encoding="utf-8"))
    os.chmod(replacement.name, 0o600)
    os.replace(replacement.name, arguments.manifest)
    os.kill(arguments.reef_pid, signal.SIGHUP)
    deadline = time.monotonic() + arguments.update_timeout
    answer = ""
    while time.monotonic() < deadline:
        status, answer, _ = client.chat("code-opd", "auto", MATH_PROMPT, False)
        if status == 200 and answer == "code-student answered.":
            break
        time.sleep(1)
    client.expect("policy update published", answer == "code-student answered.", answer=answer[:200])
    updated = run_matrix(client, "code-student")
    for scenario in ("code-opd", "math-opd"):
        references = receipts[scenario] + updated[scenario]
        status, text, _ = client.post("/reef/report", scenario, {"score": 1, "references": references})
        client.expect(f"{scenario} feedback links its receipts", status == 200, status=status, body=text[:200])
    status, text, _ = client.post("/reef/report", "math-opd", {"score": 1, "references": receipts["code-opd"][:1]})
    client.expect("foreign-Scenario feedback refused", status == 400, status=status, body=text[:200])
    all_receipts = [receipt for scenario in receipts.values() for receipt in scenario] + [
        receipt for scenario in updated.values() for receipt in scenario
    ]
    client.expect("receipts are unique", len(set(all_receipts)) == len(all_receipts), count=len(all_receipts))
    passed = all(item["passed"] for item in client.checks)
    print(json.dumps({"passed": passed, "checks": client.checks}, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    manifests = commands.add_parser("manifests", help="Write the initial and updated manifests.")
    manifests.add_argument("--model-directory", type=Path, required=True, help="Directory of the four model files.")
    manifests.add_argument("--output-directory", type=Path, required=True)
    smoke = commands.add_parser("check", help="Run the requests against a running reef serve.")
    smoke.add_argument("--url", default="http://127.0.0.1:8900")
    smoke.add_argument("--manifest", type=Path, required=True, help="The manifest reef serve reads; it is replaced.")
    smoke.add_argument("--updated-manifest", type=Path, required=True)
    smoke.add_argument("--reef-pid", type=int, required=True, help="The PID of reef serve, to send SIGHUP.")
    smoke.add_argument("--update-timeout", type=float, default=180.0)
    arguments = parser.parse_args()
    if arguments.command == "manifests":
        write_manifests(arguments.model_directory, arguments.output_directory)
        return 0
    return check(arguments)


if __name__ == "__main__":
    sys.exit(main())
