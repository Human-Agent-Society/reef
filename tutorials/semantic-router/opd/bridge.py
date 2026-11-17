"""Expose native OPD student and teacher services as two loopback routing targets for the feasibility test."""

import asyncio
import hashlib
import json
import os
import time
from contextlib import AsyncExitStack
from urllib.parse import quote

from aiohttp import ClientError, ClientSession, ClientTimeout, web

SCENARIO = os.environ.get("OPD_ROUTING_SCENARIO", "route-opd-smoke")
MARKER = "REEF_TEACHER_CONTROL"
NATIVE_URL = os.environ.get("OPD_NATIVE_URL", "http://127.0.0.1:8900").rstrip("/")
TEACHER_URL = os.environ.get("OPD_TEACHER_URL", "http://127.0.0.1:30001").rstrip("/")


def sampling_digest(payload: dict) -> str:
    """Identify the prompt, seed and sampling parameters shared across the two Reef records."""
    fields = {key: payload[key] for key in ("messages", "seed", "temperature", "top_p", "max_tokens")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def target_app(role: str, session: ClientSession) -> web.Application:
    """One provider alias forwarding to its native backend, with at most one generation per experiment input."""
    student = role == "student"
    base = NATIVE_URL if student else TEACHER_URL
    token = os.environ["REEF_TOKEN"] if student else os.environ.get("OPD_TEACHER_API_KEY")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if student:
        headers["x-reef-scenario"] = SCENARIO
    served_model = (
        os.environ.get("OPD_STUDENT_MODEL", "Qwen3-0.6B") if student else os.environ.get("OPD_TEACHER_MODEL")
    )
    seen: set[str] = set()

    async def models(request: web.Request) -> web.Response:
        nonlocal served_model
        try:
            path = "/healthz" if student else "/v1/models"
            async with session.get(base + path, headers=headers, allow_redirects=False, timeout=15) as reply:
                if reply.status != 200:
                    raise web.HTTPServiceUnavailable(text=f"{role} readiness returned HTTP {reply.status}")
                if not student:
                    available = [row["id"] for row in (await reply.json())["data"]]
                    if served_model is None and len(available) == 1:
                        served_model = available[0]
                    if served_model not in available:
                        raise web.HTTPServiceUnavailable(text="Set OPD_TEACHER_MODEL to a served teacher model ID")
        except (ClientError, TimeoutError, KeyError, ValueError) as error:
            raise web.HTTPServiceUnavailable(text=f"{role} readiness failed") from error
        return web.json_response({"object": "list", "data": [{"id": role, "object": "model"}]})

    async def complete(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if body.get("model") != role or body.get("stream") or type(body.get("seed")) is not int:
                raise ValueError("expected a seeded non-streaming request to this provider alias")
            key = sampling_digest(body)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise web.HTTPBadRequest(text="Invalid feasibility test completion request") from error
        if served_model is None:
            raise web.HTTPServiceUnavailable(text="Check provider readiness before inference")
        if key in seen:
            raise web.HTTPConflict(text="Refusing to generate twice for the same test input")
        seen.add(key)
        started = time.monotonic()
        try:
            async with session.post(
                base + "/v1/chat/completions",
                json={**body, "model": served_model},
                headers=headers,
                allow_redirects=False,
            ) as reply:
                if reply.status != 200:
                    raise web.HTTPBadGateway(text=f"{role} generation returned HTTP {reply.status}")
                result = await reply.json()
                if not isinstance(result, dict) or not result.get("choices"):
                    raise ValueError("Backend returned no chat choices")
                receipt = reply.headers.get("x-reef-agent-record-id")
            generation_seconds = time.monotonic() - started
            artifact = None
            if student:
                if not receipt:
                    raise web.HTTPBadGateway(text="Native student response has no Reef receipt")
                record_url = f"{base}/reef/scenarios/{quote(SCENARIO, safe='')}/records/{quote(receipt, safe='')}"
                async with session.get(record_url, headers=headers, allow_redirects=False) as reply:
                    if reply.status != 200:
                        raise web.HTTPBadGateway(text=f"Native record lookup returned HTTP {reply.status}")
                    record = await reply.json()
                artifact = record["artifact_id"]
                if not artifact or record["agent_record_id"] != receipt:
                    raise web.HTTPBadGateway(text="Native receipt has no matching weight artifact")
            elif receipt is not None:
                raise web.HTTPBadGateway(text="Teacher target must be its serving engine, not a training service")
        except (ClientError, TimeoutError, KeyError, ValueError) as error:
            raise web.HTTPBadGateway(text=f"{role} returned an invalid backend reply") from error
        result.pop("training", None)
        result["experiment_bridge"] = {
            "role": role,
            "request_sha256": key,
            "native_receipt": receipt,
            "artifact_id": artifact,
            "generation_seconds": generation_seconds,
        }
        return web.json_response(result)

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", complete)
    return app


async def main() -> None:
    if not os.environ.get("REEF_TOKEN"):
        raise SystemExit("Set REEF_TOKEN to the native OPD Reef token")
    async with ClientSession(timeout=ClientTimeout(total=1800), trust_env=False) as session, AsyncExitStack() as stack:
        for role, port in (("student", 18911), ("teacher", 18912)):
            runner = web.AppRunner(target_app(role, session), access_log=None)
            await runner.setup()
            stack.push_async_callback(runner.cleanup)
            port = int(os.environ.get(f"{role.upper()}_BRIDGE_PORT", port))
            await web.TCPSite(runner, "127.0.0.1", port).start()
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
