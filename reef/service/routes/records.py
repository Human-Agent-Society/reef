from __future__ import annotations

import asyncio

from aiohttp import web

from reef.core.records_types import RequestType
from reef.service.auth import page_key_of
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object


def register_record_routes(app: web.Application, *, request_service: RequestService) -> None:
    async def import_record(request: web.Request) -> web.Response:
        body = await read_object(request)
        item = await asyncio.to_thread(request_service.import_record, request.headers, body)
        return web.json_response(
            {
                "agent_record_id": item.agent_record_id,
                "scenario": item.scenario,
                "request_type": item.request_type.value,
            }
        )

    async def import_records(request: web.Request) -> web.Response:
        body = await read_object(request)
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
            answer = {
                "agent_record_id": item.agent_record_id,
                "scenario": item.scenario,
                "request_type": item.request_type.value,
            }
            # A training request has a page: the key its link carries in place of the token.
            page_key = page_key_of(request, item.scenario) if request_type is RequestType.TRAIN else None
            if page_key is not None:
                answer["page_key"] = page_key
            return web.json_response(answer)

        return accept

    app.router.add_post("/reef/records", import_record)
    app.router.add_post("/reef/records/batch", import_records)
    app.router.add_post("/reef/report", accept_typed(RequestType.REPORT))
    app.router.add_post("/reef/train", accept_typed(RequestType.TRAIN))


__all__ = ["register_record_routes"]
