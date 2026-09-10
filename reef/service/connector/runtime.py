"""Translate the small connector command vocabulary into native Reef calls."""

from __future__ import annotations

import asyncio
import ipaddress
import math
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp

BODY_LIMIT = 192 * 1024


class HTTPFailure(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def endpoint_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname == "localhost" or hostname.endswith(".localhost")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("Use HTTPS, or HTTP on localhost")
    if not hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint URLs cannot contain credentials, a query, or a fragment")
    _ = parsed.port
    return value.rstrip("/")


class JSONClient:
    """HTTP boundary with bounded responses, no cookies, redirects, or implicit retries."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str, token: str = ""):
        self.session = session
        self.base_url = endpoint_url(base_url)
        self.token = token

    async def request(
        self, path: str, *, body: dict[str, Any] | None = None, scenario: str | None = None, timeout: float = 15
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        if scenario:
            headers["x-reef-scenario"] = scenario
        async with self.session.request(
            "GET" if body is None else "POST",
            self.base_url + path,
            headers=headers,
            json=body,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=False,
        ) as response:
            chunks = bytearray()
            async for chunk in response.content.iter_chunked(8192):
                chunks.extend(chunk)
                if len(chunks) > 2 * 1024 * 1024:
                    raise HTTPFailure(502, "Response exceeds the connector size limit")
            if response.status < 200 or response.status >= 300:
                # Do not upload runtime error bodies: they can contain credentials, paths, or provider payloads.
                raise HTTPFailure(response.status, f"HTTP {response.status}")
            import json

            try:
                result = json.loads(chunks)
            except (ValueError, UnicodeDecodeError) as exc:
                raise HTTPFailure(502, "Expected a JSON object") from exc
            if not isinstance(result, dict):
                raise HTTPFailure(502, "Expected a JSON object")
            return result


class ReefRuntime:
    def __init__(self, client: JSONClient):
        self.client = client

    async def snapshot(self) -> dict[str, Any]:
        try:
            listing, status = await asyncio.gather(
                self.client.request("/reef/scenarios", timeout=5), self.client.request("/reef/status", timeout=5)
            )
            entries = listing.get("scenarios")
            if not isinstance(entries, list) or len(entries) > 200:
                raise ValueError("Expected at most 200 scenarios")
            scenarios = []
            for item in entries:
                name = scenario_name(item.get("scenario"))
                row = {"scenario": name}
                if isinstance(item.get("release_id"), str):
                    row["release_id"] = item["release_id"][:180]
                mode = status.get("scenarios", {}).get(name, {}).get("training_mode")
                if mode in ("auto", "manual", "hybrid"):
                    row["training_mode"] = mode
                scenarios.append(row)
            return {"reachable": True, "scenarios": scenarios}
        except (aiohttp.ClientError, TimeoutError, ValueError, HTTPFailure, TypeError, AttributeError):
            return {
                "reachable": False,
                "scenarios": [],
                "error": "Cannot read Reef. Check the local URL, service token and runtime status.",
            }

    async def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        action = command.get("action")
        if action == "refresh":
            return await self.snapshot()
        name = scenario_name(command.get("scenario"))
        path = "/reef/scenarios/" + quote(name, safe="")
        if action == "releases":
            result = await self.client.request(path + "/releases")
            rows = result.get("releases")
            if not isinstance(rows, list):
                raise ValueError("Reef did not return a release list")
            return {"releases": [release_summary(row) for row in rows[:100]], "truncated": len(rows) > 100}
        if action == "create_scenario":
            await self.client.request("/reef/scenarios", body={"name": name}, timeout=60)
        elif action == "set_training_mode":
            mode = command.get("training_mode")
            if mode not in ("auto", "manual", "hybrid"):
                raise ValueError("Invalid training mode")
            await self.client.request(path + "/update", body={"training_mode": mode}, timeout=60)
        elif action in ("promote", "rollback"):
            release_id = command.get("release_id")
            if not isinstance(release_id, str) or not release_id or len(release_id) > 180:
                raise ValueError("Invalid release ID")
            await self.client.request(path + "/" + action, body={"release_id": release_id}, timeout=60)
        elif action == "train":
            instruction = command.get("text")
            if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4000:
                raise ValueError("Invalid training instruction")
            await self.client.request(
                "/reef/train", scenario=name, body={"text": instruction, "agent_record_id": command["id"]}, timeout=60
            )
            return {"scenario": name, "agent_record_id": command["id"]}
        else:
            raise ValueError("Unsupported connector action")
        return {"scenario": name}


def scenario_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 180
        or value in (".", "..")
        or any(character in value for character in ("/", "\\", "\r", "\n", "\0"))
    ):
        raise ValueError("Invalid scenario name")
    return value


def release_summary(row: Any) -> dict[str, Any]:
    """Only publish catalog identifiers and numeric gate results, never artifact files or prompts."""
    if not isinstance(row, dict):
        raise ValueError("Invalid release row")
    result: dict[str, Any] = {}
    for key in ("release_id", "parent_release_id", "content_id", "operation", "rollback_target_release_id"):
        if isinstance(row.get(key), str):
            result[key] = row[key][:180]
    for key in ("current", "pending", "restorable", "checkpoint"):
        if isinstance(row.get(key), bool):
            result[key] = row[key]
    if isinstance(row.get("recorded_at"), (int, float)) and math.isfinite(row["recorded_at"]):
        result["recorded_at"] = row["recorded_at"]
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        result["metrics"] = {
            key: metrics[key]
            for key in ("selected", "wins", "losses", "ties", "candidate_score", "current_score")
            if isinstance(metrics.get(key), (bool, int, float)) and math.isfinite(metrics[key])
        }
        if metrics.get("skipped"):
            result["metrics"]["skipped"] = True
    return result
