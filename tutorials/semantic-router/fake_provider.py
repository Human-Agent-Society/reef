"""A deterministic CPU stand-in for OpenAI-compatible Chat Completions services.

It runs no model and makes no network calls: every answer is ``"<model> answered."``, so a response shows
which target Reef selected. It serves ``GET /v1/models`` and ``POST /v1/chat/completions``, as JSON or, with
``"stream": true``, as SSE ending in ``data: [DONE]``, on ``127.0.0.1`` only. Use it to try routing, not to
measure anything.
"""

from __future__ import annotations

import argparse
import json

from aiohttp import web


def chat_completion_app(models: tuple[str, ...]) -> web.Application:
    """An application that answers chat completions for ``models`` and refuses any other model with 404."""

    async def list_models(request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": model, "object": "model"} for model in models]})

    async def chat_completions(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        model = payload.get("model")
        if model not in models:
            return web.json_response({"error": {"message": f"model {model!r} is not served here"}}, status=404)
        content = f"{model} answered."
        identity = {"id": "chatcmpl-cpu-provider", "created": 0, "model": model}
        usage = {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}
        if payload.get("stream") is not True:
            choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            return web.json_response({**identity, "object": "chat.completion", "choices": [choice], "usage": usage})
        chunks: list[dict[str, object]] = [
            {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None},
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for choice in chunks:
            event = {**identity, "object": "chat.completion.chunk", "choices": [choice]}
            await response.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        stream_options = payload.get("stream_options")
        if isinstance(stream_options, dict) and stream_options.get("include_usage") is True:
            event = {**identity, "object": "chat.completion.chunk", "choices": [], "usage": usage}
            await response.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help="Port on 127.0.0.1.")
    parser.add_argument("--model", action="append", required=True, help="A model ID to serve; repeat for more.")
    arguments = parser.parse_args()
    app = chat_completion_app(tuple(arguments.model))
    web.run_app(app, host="127.0.0.1", port=arguments.port, print=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
