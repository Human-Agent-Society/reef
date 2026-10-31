from __future__ import annotations

import asyncio
from urllib.parse import quote

from aiohttp import web

from reef.core.records_types import RequestType

from reef.service.auth import page_query
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object

#: The most a record import takes in one request. A one-million-token trajectory is about 5 MB of JSON, so one
#: imported record or batch can hold it.
MAX_RECORD_IMPORT_BYTES = 10 * 1024 * 1024


def register_record_routes(app: web.Application, *, request_service: RequestService) -> None:
    async def import_record(request: web.Request) -> web.Response:
        body = await read_object(request.clone(client_max_size=MAX_RECORD_IMPORT_BYTES))
        item = await asyncio.to_thread(request_service.import_record, request.headers, body)
        return web.json_response(
            {
                "agent_record_id": item.agent_record_id,
                "scenario": item.scenario,
                "request_type": item.request_type.value,
            }
        )

    async def import_records(request: web.Request) -> web.Response:
        body = await read_object(request.clone(client_max_size=MAX_RECORD_IMPORT_BYTES))
        items = await asyncio.to_thread(request_service.import_records, request.headers, body)
        return web.json_response(
            {
                "records": [
                    {
                        "agent_record_id": item.agent_record_id,
                        "scenario": item.scenario,
                        "request_type": item.request_type.value,
                    }
                    for item in items
                ]
            }
        )

    def accept_typed(request_type: RequestType):
        async def accept(request: web.Request) -> web.Response:
            payload = await read_object(request)
            agent_record_id = None
            if isinstance(payload, dict) and "agent_record_id" in payload:
                # Client-supplied id (see reef-protocols.md): lets a
                # harness retry a report safely — identical resends
                # dedup, same id with different content conflicts.
                agent_record_id = payload.pop("agent_record_id")
                if not isinstance(agent_record_id, str) or not agent_record_id.strip():
                    raise ValueError("agent_record_id must be a non-empty string")
            item = await asyncio.to_thread(
                request_service.accept,
                request.headers,
                payload,
                request_type=request_type,
                agent_record_id=agent_record_id,
            )
            return web.json_response(
                {
                    "agent_record_id": item.agent_record_id,
                    "scenario": item.scenario,
                    "request_type": item.request_type.value,
                }
            )

        return accept

    async def accept_instruction(request: web.Request) -> web.Response:
        """Queue a training instruction (``RequestService.accept_instruction``); it is not a record."""
        payload = await read_object(request)
        request_id = None
        if isinstance(payload, dict) and "agent_record_id" in payload:
            # A client-supplied ID makes the request retry-safe: an identical resend returns the queued instruction.
            request_id = payload.pop("agent_record_id")
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("agent_record_id must be a non-empty string")
        scenario, instruction_id = await asyncio.to_thread(
            request_service.accept_instruction, request.headers, payload, request_id=request_id
        )
        # A training request has a page; its link carries a page key in place of the token.
        page = f"/reef/harness/requests/{quote(instruction_id, safe='')}/page?{page_query(request, scenario)}"
        return web.json_response(
            {"agent_record_id": instruction_id, "scenario": scenario, "request_type": "train", "page_path": page}
        )

    app.router.add_post("/reef/records", import_record)
    app.router.add_post("/reef/records/batch", import_records)
    app.router.add_post("/reef/report", accept_typed(RequestType.REPORT))
    app.router.add_post("/reef/train", accept_instruction)


__all__ = ["register_record_routes"]
