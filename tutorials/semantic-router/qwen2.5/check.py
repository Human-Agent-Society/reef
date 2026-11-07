"""Prepare the two policies and verify real Qwen2.5 requests through a local managed Reef deployment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import closing
from pathlib import Path

import aiohttp
import yaml
from hot_update import CHAT, observe_update, owner_events

from reef.service.routing import RoutingSnapshot
from reef.storage.sqlite import SQLiteScenarioStorage

SCENARIO = "qwen-demo"
MODELS = {"student": "Qwen/Qwen2.5-0.5B-Instruct", "teacher": "Qwen/Qwen2.5-1.5B-Instruct"}
PROMPT = "What is two plus two? Answer in one sentence."
KEYWORD = "reef_teacher_required"


def write_manifests(args: argparse.Namespace) -> None:
    """Write policies A and B. Only the default target changes."""
    if args.student_port == args.teacher_port:
        raise ValueError("the two vLLM ports must differ")
    targets = {}
    for role, revision, port in (
        ("student", args.student_revision, args.student_port),
        ("teacher", args.teacher_revision, args.teacher_port),
    ):
        targets[role] = {
            "provider_model": MODELS[role],
            "revision": revision,
            "url": f"http://127.0.0.1:{port}",
            "role": role,
        }
    args.output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, default in (("manifest.yaml", "student"), ("manifest.updated.yaml", "teacher")):
        manifest = {
            "schema_version": 2,
            "automatic_model": "auto",
            "deployment": {"semantic_router_image": args.image},
            "targets": targets,
            "scenarios": {
                SCENARIO: {
                    "default_target": default,
                    "targets": list(MODELS),
                    "rules": [{"name": "teacher_requested", "keywords": [KEYWORD], "target": "teacher"}],
                }
            },
        }
        RoutingSnapshot.from_mapping(manifest)
        with (args.output_directory / name).open("x", encoding="utf-8") as output:
            yaml.safe_dump(manifest, output, sort_keys=False)


async def completion(
    session: aiohttp.ClientSession,
    selector: str,
    prompt: str,
    target: str,
    rule: str | None,
    version: str,
    first_content: asyncio.Event | None = None,
    stream_tokens: int = 384,
) -> dict[str, object]:
    """Check the provider model, generated content, token count and Reef receipt in the actual JSON or SSE reply."""
    stream = first_content is not None
    body = {
        "model": selector,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": stream,
        "max_tokens": stream_tokens if stream else 24,
    }
    if stream:
        body.update(min_tokens=stream_tokens, stream_options={"include_usage": True})
    start = time.monotonic()
    first = last = None
    async with session.post(CHAT, json=body, allow_redirects=False) as response:
        if response.status != 200:
            raise RuntimeError(f"{selector} returned HTTP {response.status}: {(await response.text())[:300]}")
        if not stream:
            reply = await response.json()
            models = {reply["model"]}
            answer = reply["choices"][0]["message"]["content"]
            receipt = response.headers.get("x-reef-agent-record-id")
            tokens = reply["usage"]["completion_tokens"]
        else:
            models = set()
            receipt = None
            tokens = None
            chunks = []
            async for line in response.content:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                if receipt:
                    raise RuntimeError("SSE receipt must directly precede [DONE]")
                chunk = json.loads(data)
                models.add(chunk["model"])
                receipt = chunk.get("reef", {}).get("agent_record_id")
                if chunk.get("usage"):
                    tokens = chunk["usage"]["completion_tokens"]
                for choice in chunk["choices"]:
                    content = choice["delta"].get("content")
                    if content:
                        last = time.monotonic()
                        if first is None:
                            first = last
                            first_content.set()
                        chunks.append(content)
            else:
                raise RuntimeError("SSE ended without [DONE]")
            answer = "".join(chunks)
    end = time.monotonic()
    if models != {MODELS[target]} or not answer.strip() or not receipt:
        raise RuntimeError(
            f"expected {MODELS[target]} with output and a receipt; models={models}, content={answer[:120]!r}"
        )
    if type(tokens) is not int or tokens <= 0 or (stream and tokens != stream_tokens):
        raise RuntimeError(f"unexpected completion token count: {tokens}")
    return {
        "selector": selector,
        "target": target,
        "model": next(iter(models)),
        "rule": rule,
        "snapshot": version,
        "receipt": receipt,
        "completion_tokens": tokens,
        "answer_excerpt": answer.strip()[:120],
        "start_monotonic_seconds": start,
        "end_monotonic_seconds": end,
        "latency_seconds": end - start,
        "first_content_monotonic_seconds": first,
        "last_content_monotonic_seconds": last,
    }


async def check(args: argparse.Namespace, pidfd: int) -> dict[str, object]:
    directory = args.run_directory
    source = directory / "manifest.yaml"
    initial, updated = source.read_bytes(), (directory / "manifest.updated.yaml").read_bytes()
    events = owner_events(args.deployment_id, args.reef_pid)
    # Rendering adds a configuration digest. Read it from this generation to calculate both exact snapshot IDs.
    rendered = yaml.safe_load((events.parent / "manifest.yaml").read_bytes())
    versions = []
    for contents in (initial, updated):
        policy = yaml.safe_load(contents)
        policy["deployment"]["configuration_digest"] = rendered["deployment"]["configuration_digest"]
        versions.append(RoutingSnapshot.from_mapping(policy).version)
    if versions[0] == versions[1] or RoutingSnapshot.from_mapping(rendered).version != versions[0]:
        raise RuntimeError("expected a fresh owner serving this tutorial's initial manifest")
    results = []
    updates = []
    headers = {"Authorization": "Bearer " + os.environ["REEF_TOKEN"], "x-reef-scenario": SCENARIO}
    async with aiohttp.ClientSession(
        base_url=f"http://127.0.0.1:{args.reef_port}", headers=headers, timeout=aiohttp.ClientTimeout(total=180)
    ) as session:
        phases = [("initial", "student", versions[0], 0)]
        for tokens in args.stream_tokens:
            phases.extend((("idle", "teacher", versions[1], 0), ("under_load", "student", versions[0], tokens)))
        for phase, target, version, tokens in phases:
            old_reply = None
            if phase == "under_load":
                first_content = asyncio.Event()
                async with asyncio.TaskGroup() as group:
                    stream = group.create_task(
                        completion(
                            session,
                            "auto",
                            "Explain how rivers shape a landscape in great detail.",
                            "teacher",
                            "default",
                            versions[1],
                            first_content,
                            tokens,
                        )
                    )
                    async with asyncio.timeout(120):
                        await first_content.wait()
                    timing = await observe_update(session, pidfd, events, source, initial, version)
                    # Run on V2 before joining the teacher task: the old stream must outlive this answer.
                    reply = await completion(session, "auto", PROMPT, target, "default", version)
                old_reply = stream.result()
                if old_reply["last_content_monotonic_seconds"] <= reply["end_monotonic_seconds"]:
                    raise RuntimeError("expected more V1 teacher content after the V2 student answer")
                old_reply["phase"] = "under_load_old_stream"
                results.append(old_reply)
            else:
                if phase == "idle":
                    timing = await observe_update(session, pidfd, events, source, updated, version)
                reply = await completion(session, "auto", PROMPT, target, "default", version)
            reply["phase"] = phase
            results.append(reply)
            if phase != "initial":
                signal_time = timing["signal_monotonic_seconds"]
                timing.update(
                    phase=phase,
                    signal_to_first_new_completion_seconds=reply["end_monotonic_seconds"] - signal_time,
                    first_new_request_latency_seconds=reply["latency_seconds"],
                    first_new_request_start_seconds_from_signal=reply["start_monotonic_seconds"] - signal_time,
                )
                if old_reply:
                    last_from_signal = old_reply["last_content_monotonic_seconds"] - signal_time
                    timing["old_stream_last_content_seconds_from_signal"] = last_from_signal
                    timing["old_stream_done_seconds_from_signal"] = old_reply["end_monotonic_seconds"] - signal_time
                    timing["old_stream_tokens"] = tokens
                updates.append(timing)
            for selector, expected, rule in (("auto", "teacher", "teacher_requested"), ("student", "student", None)):
                reply = await completion(session, selector, f"{KEYWORD} {PROMPT}", expected, rule, version)
                reply["phase"] = phase
                results.append(reply)

    # The routed public API returns the receipt. Read it back from this run's local Reef record store.
    receipts = [result["receipt"] for result in results]
    if len(set(receipts)) != len(receipts):
        raise RuntimeError("duplicate Reef receipts")
    with closing(SQLiteScenarioStorage(directory / "records")) as storage, closing(storage.open(SCENARIO)) as scenario:
        for result in results:
            receipt = result["receipt"]
            record = scenario.records.get(SCENARIO, receipt)
            if record is None:
                raise RuntimeError(f"missing record for receipt {receipt}")
            route = record.payload["metadata"]["routing"]
            expected = {
                "original_selector": result["selector"],
                "target_id": result["target"],
                "provider_model": result["model"],
                "rule_name": result["rule"],
                "snapshot_version": result["snapshot"],
            }
            if any(route[k] != value for k, value in expected.items()):
                raise RuntimeError(f"receipt {receipt} has unexpected routing metadata: {route}")
            if bool(route["router_config_hash"]) != (result["selector"] == "auto"):
                raise RuntimeError(f"receipt {receipt} has unexpected SR runtime hash")
            result["router_config_hash"] = route["router_config_hash"]
    return {
        "passed": True,
        "deployment_id": args.deployment_id,
        "generation": events.parent.name,
        "completions": results,
        "updates": updates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifests = commands.add_parser("manifests", help="Generate two manifests in a fresh run directory.")
    manifests.add_argument("--image", required=True)
    manifests.add_argument("--output-directory", required=True, type=Path)
    for role in MODELS:
        manifests.add_argument(f"--{role}-port", type=int, required=True)
        manifests.add_argument(f"--{role}-revision", required=True)
    verify = commands.add_parser("check", help="Run on the same Linux host as this tutorial's reef serve.")
    verify.add_argument("--run-directory", required=True, type=Path)
    verify.add_argument("--deployment-id", required=True)
    verify.add_argument("--reef-port", required=True, type=int)
    verify.add_argument("--reef-pid", required=True, type=int)
    verify.add_argument("--stream-tokens", nargs="+", type=int, default=[384], help="One update pair per length")
    args = parser.parse_args()
    if args.command == "manifests":
        write_manifests(args)
    else:
        if any(not 1 <= count <= 1536 for count in args.stream_tokens):
            verify.error("--stream-tokens entries must be between 1 and 1536")
        # Linux pidfd_send_signal accepts an open /proc/PID directory and rejects recycled PIDs.
        pidfd = os.open(f"/proc/{args.reef_pid}", os.O_RDONLY | os.O_DIRECTORY)
        try:
            result = asyncio.run(check(args, pidfd))
        finally:
            os.close(pidfd)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, aiohttp.ClientError, ExceptionGroup) as exc:
        errors = exc.exceptions if isinstance(exc, ExceptionGroup) else (exc,)
        print(json.dumps({"passed": False, "errors": [str(error) for error in errors]}))
        sys.exit(1)
