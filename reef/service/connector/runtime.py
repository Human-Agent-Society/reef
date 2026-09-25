"""Translate the small connector command vocabulary into native Reef calls."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import re
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp

from reef.service.release_page import mutations_of, result_of

#: A harness adapter name as the console builds ``reef-<adapter>`` commands from it.
ADAPTER_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
#: The newest harness requests one ``requests`` command reports, and the record pages it reads to find them.
MAX_REQUESTS = 100
MAX_REQUEST_PAGES = 50
#: The step metrics a release summary copies when they are finite numbers.
SUMMARY_METRICS = (
    "selected",
    "wins",
    "losses",
    "ties",
    "candidate_score",
    "current_score",
    "passed",
    "failed",
    "floor_score",
)


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
        self,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        scenario: str | None = None,
        timeout: float = 15,
        method: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        if scenario:
            headers["x-reef-scenario"] = scenario
        async with self.session.request(
            method or ("GET" if body is None else "POST"),
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
        url = self.client.base_url
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
                if isinstance(item.get("adapter"), str) and ADAPTER_NAME.fullmatch(item["adapter"]):
                    row["adapter"] = item["adapter"]
                mode = status.get("scenarios", {}).get(name, {}).get("training_mode")
                if mode in ("auto", "manual", "hybrid"):
                    row["training_mode"] = mode
                scenarios.append(row)
            return {"reachable": True, "scenarios": scenarios, "reef_url": url}
        # The code tells the console which setting to check; the message never includes a response body.
        except TimeoutError:
            error_code, error = "timeout", f"Reef at {url} did not answer within 5 seconds."
        except aiohttp.ClientConnectorError:
            error_code = "connection_failed"
            error = f"Nothing answered at {url}. Start Reef there, or connect with the address Reef serves on."
        except HTTPFailure as exc:
            if exc.status in (401, 403):
                error_code = "unauthorized"
                error = f"Reef at {url} rejected the connector's service token (HTTP {exc.status})."
            elif exc.status == 404:
                error_code, error = "not_reef", f"The service at {url} is not a Reef runtime (HTTP 404)."
            else:
                error_code, error = "invalid_response", f"Reef at {url} returned an unexpected response ({exc})."
        except (aiohttp.ClientError, ValueError, TypeError, AttributeError):
            error_code, error = "invalid_response", f"Reef at {url} returned an unexpected response."
        return {"reachable": False, "scenarios": [], "reef_url": url, "error_code": error_code, "error": error}

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
        if action == "requests":
            return await self.harness_requests(name, path)
        if action == "create_scenario":
            await self.client.request("/reef/scenarios", body={"name": name}, timeout=60)
        elif action == "delete_scenario":
            await self.client.request(path, method="DELETE", timeout=60)
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

    async def harness_requests(self, name: str, path: str) -> dict[str, Any]:
        """The scenario's harness requests, newest first: ids, states and filed times, never the request text.

        The retained training instructions come from the record list, oldest
        first, and each one's state from its progress read, the reading the
        request page and ``reef-<adapter> wait`` use. Only the state code and
        the step leave the machine; the activity lines stay.
        """
        records: list[dict[str, Any]] = []
        cursor: int | None = 0
        for _ in range(MAX_REQUEST_PAGES):
            page = await self.client.request(f"{path}/records?request_type=train&limit=100&after_sequence={cursor}")
            rows, following = page.get("records"), page.get("next_after_sequence")
            if not isinstance(rows, list):
                raise ValueError("Reef did not return a record list")
            if following is not None and (not isinstance(following, int) or isinstance(following, bool)):
                raise ValueError("Reef returned an invalid record cursor")
            # A Reef that predates the type filter lists every record, so the type is checked here too.
            records.extend(row for row in rows if isinstance(row, dict) and row.get("request_type") == "train")
            cursor = following
            if cursor is None:
                break
        requests = []
        for record in records[::-1][:MAX_REQUESTS]:
            record_id = record.get("agent_record_id")
            if not isinstance(record_id, str) or not record_id or len(record_id) > 180:
                raise ValueError("Reef returned an invalid request ID")
            progress = await self.client.request(
                "/reef/harness/requests/" + quote(record_id, safe="") + "/progress", scenario=name
            )
            state, step = progress.get("state"), progress.get("step")
            entry: dict[str, Any] = {"id": record_id, "state": state[:40] if isinstance(state, str) else "unknown"}
            if isinstance(step, int) and not isinstance(step, bool):
                entry["step"] = step
            created_at = record.get("created_at")
            if isinstance(created_at, (int, float)) and math.isfinite(created_at):
                entry["created_at"] = created_at
            requests.append(entry)
        return {"requests": requests, "truncated": cursor is not None or len(records) > MAX_REQUESTS}


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


def finite_numbers(values: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """The named values that are finite numbers or booleans, so the summary stays valid JSON."""
    return {
        key: values[key]
        for key in keys
        if isinstance(values.get(key), (bool, int, float)) and math.isfinite(values[key])
    }


def release_summary(row: Any) -> dict[str, Any]:
    """Only publish catalog identifiers, numeric evaluation results and short codes.

    Never artifact files, request text, entry bodies or prompts: a request
    keeps its id and the names and kinds of what it requires, an agent's
    proposal its id, a recheck its reason code, a mutation its op, id and
    kind, and a selection its outcome, policy, evaluator and counts.
    ``result`` is the release page's result for a step (``selected``,
    ``rejected``, ``skipped``, ``failed`` or ``pending``).
    """
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
        summary = finite_numbers(metrics, SUMMARY_METRICS)
        if metrics.get("skipped"):
            summary["skipped"] = True
        request = metrics.get("training_request")
        if isinstance(request, dict) and isinstance(request.get("id"), str):
            requires = request.get("requires")
            summary["training_request"] = {
                "id": request["id"][:180],
                "requires": [
                    {"name": item["name"][:180], "kind": item["kind"][:40]}
                    for item in (requires if isinstance(requires, list) else [])[:50]
                    if isinstance(item, dict)
                    and isinstance(item.get("name"), str)
                    and isinstance(item.get("kind"), str)
                ],
            }
        # An agent's proposal keeps its id; its reason is text and stays.
        proposal = metrics.get("proposal")
        if isinstance(proposal, dict) and isinstance(proposal.get("id"), str):
            summary["proposal"] = {"id": proposal["id"][:180]}
        if metrics.get("recheck") is True:
            summary["recheck"] = True
            if isinstance(metrics.get("recheck_reason"), str):
                summary["recheck_reason"] = metrics["recheck_reason"][:40]
        mutations = mutations_of(metrics)
        if mutations:
            summary["mutation_count"] = len(mutations)
            summary["mutations"] = []
            for mutation in mutations[:50]:
                entry: dict[str, Any] = {
                    key: mutation[key][:180] for key in ("op", "id") if isinstance(mutation.get(key), str)
                }
                options = mutation.get("options")
                # The kind is the entry's ``options.name``; its ``config`` is the entry body and stays.
                if isinstance(options, dict) and isinstance(options.get("name"), str):
                    entry["options"] = {"name": options["name"][:80]}
                summary["mutations"].append(entry)
        selection = metrics.get("selection")
        if isinstance(selection, dict):
            decision: dict[str, Any] = {
                key: selection[key][:80] for key in ("outcome", "policy") if isinstance(selection.get(key), str)
            }
            if isinstance(selection.get("metrics"), dict):
                decision["metrics"] = finite_numbers(selection["metrics"], ("passed", "failed", "floor_score"))
            evaluation = selection.get("evaluation")
            if isinstance(evaluation, dict) and isinstance(evaluation.get("evaluator"), str):
                decision["evaluation"] = {"evaluator": evaluation["evaluator"][:80]}
            summary["selection"] = decision
        result["metrics"] = summary
    # The operation already names a row that is no step; a step gets its result.
    code = result_of(row)
    if code != str(row.get("operation") or "unknown"):
        result["result"] = code
    return result
