from __future__ import annotations

import asyncio
import json

from aiohttp import web

from reef.service.artifact_evaluations import ArtifactEvaluations
from reef.service.routes.payload import read_object


def register_evaluation_reads(app: web.Application, *, evaluations: ArtifactEvaluations) -> None:
    """The reads of stored evaluations and their rollouts."""

    async def evaluation(request: web.Request) -> web.Response:
        evaluation_id = request.match_info["evaluation_id"]
        stored = await asyncio.to_thread(evaluations.get, request.match_info["scenario"], evaluation_id)
        if stored is None:
            raise web.HTTPNotFound(text=f"no evaluation {evaluation_id!r}")
        return web.json_response(stored.to_json(), headers={"Cache-Control": "no-store"})

    async def rollouts(request: web.Request) -> web.Response:
        scenario, evaluation_id = request.match_info["scenario"], request.match_info["evaluation_id"]
        views = await asyncio.to_thread(evaluations.rollouts_json, scenario, evaluation_id)
        if views is None:
            raise web.HTTPNotFound(text=f"no evaluation {evaluation_id!r}")
        return web.json_response(
            {"scenario": scenario, "evaluation_id": evaluation_id, "rollouts": views},
            headers={"Cache-Control": "no-store"},
        )

    async def calls(request: web.Request) -> web.Response:
        scenario, evaluation_id = request.match_info["scenario"], request.match_info["evaluation_id"]
        # A path number that is not an integer is a ValueError, which answers HTTP 400.
        task_index, repeat = int(request.match_info["task_index"]), int(request.match_info["repeat"])
        views = await asyncio.to_thread(evaluations.calls_json, scenario, evaluation_id, task_index, repeat)
        if views is None:
            raise web.HTTPNotFound(text=f"no evaluation {evaluation_id!r}")
        return web.json_response(
            {
                "scenario": scenario,
                "evaluation_id": evaluation_id,
                "task_index": task_index,
                "repeat": repeat,
                "calls": views,
            },
            headers={"Cache-Control": "no-store"},
        )

    async def call(request: web.Request) -> web.Response:
        scenario, evaluation_id = request.match_info["scenario"], request.match_info["evaluation_id"]
        key = tuple(int(request.match_info[name]) for name in ("task_index", "repeat", "attempt", "index"))
        view = await asyncio.to_thread(evaluations.call_json, scenario, evaluation_id, *key)
        if view is None:
            raise web.HTTPNotFound(text=f"evaluation {evaluation_id!r} keeps no call {'/'.join(map(str, key))}")
        return web.json_response(view, headers={"Cache-Control": "no-store"})

    async def trajectories(request: web.Request) -> web.Response:
        scenario, evaluation_id = request.match_info["scenario"], request.match_info["evaluation_id"]
        task_index, repeat = int(request.match_info["task_index"]), int(request.match_info["repeat"])
        views = await asyncio.to_thread(evaluations.trajectories_json, scenario, evaluation_id, task_index, repeat)
        if views is None:
            raise web.HTTPNotFound(text=f"no evaluation {evaluation_id!r}")
        return web.json_response(
            {
                "scenario": scenario,
                "evaluation_id": evaluation_id,
                "task_index": task_index,
                "repeat": repeat,
                "trajectories": views,
            },
            headers={"Cache-Control": "no-store"},
        )

    async def export(request: web.Request) -> web.StreamResponse:
        scenario, evaluation_id = request.match_info["scenario"], request.match_info["evaluation_id"]
        attempts = request.query.get("attempts")
        if attempts not in (None, "all"):
            raise ValueError("attempts is 'all', or absent for the scored attempt of each rollout")
        items = await asyncio.to_thread(evaluations.exported_rollouts, scenario, evaluation_id)
        if items is None:
            raise web.HTTPNotFound(text=f"no evaluation {evaluation_id!r}")
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-store"})
        await response.prepare(request)
        # One rollout's calls at a time, so the memory of an export does not grow with the evaluation.
        for item in items:
            lines = await asyncio.to_thread(
                evaluations.rollout_trajectories, scenario, evaluation_id, item, every_attempt=attempts == "all"
            )
            for trajectory in lines:
                await response.write(json.dumps(trajectory, ensure_ascii=False).encode() + b"\n")
        await response.write_eof()
        return response

    async def newest(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        # A limit that is not a positive integer is a ValueError, which answers HTTP 400.
        listed = await asyncio.to_thread(evaluations.newest, scenario, limit=int(request.query.get("limit", "50")))
        return web.json_response(
            {"scenario": scenario, "evaluations": [stored.to_json() for stored in listed]},
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_get("/reef/scenarios/{scenario}/evaluations", newest)
    app.router.add_get("/reef/scenarios/{scenario}/evaluations/{evaluation_id}", evaluation)
    app.router.add_get("/reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts", rollouts)
    app.router.add_get(
        "/reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls", calls
    )
    app.router.add_get(
        "/reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls/{attempt}/{index}",
        call,
    )
    app.router.add_get(
        "/reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/trajectories",
        trajectories,
    )
    app.router.add_get("/reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories", export)


def register_evaluation_writes(app: web.Application, *, evaluations: ArtifactEvaluations) -> None:
    """The start of an evaluation, which control accepts."""

    async def start(request: web.Request) -> web.Response:
        body = await read_object(request)
        stored = await asyncio.to_thread(evaluations.start, request.match_info["scenario"], body)
        return web.json_response(stored.to_json(), status=202)

    app.router.add_post("/reef/scenarios/{scenario}/evaluations", start)


__all__ = ["register_evaluation_reads", "register_evaluation_writes"]
