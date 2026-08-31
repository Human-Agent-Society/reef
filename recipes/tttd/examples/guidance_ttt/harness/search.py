"""Chat request construction and response parsing for the guidance policy.

Guidance-TTT samples the trainable policy through Reef's OpenAI-compatible
chat route. Prompt rendering and exact token/log-prob capture belong to the
configured inference backend; the harness supplies chat semantics only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def openai_action(response: Mapping[str, Any]) -> str:
    """Extract assistant text from an OpenAI chat-completions response."""
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("response choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("response has no assistant content")
    return message["content"]


def guidance_chat_request(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Describe one guidance sample through Reef's chat route.

    The request never names an adapter: Reef's weight surface addresses it to
    whatever the deployment serves, so no caller can sample the frozen base by
    forgetting to name it.
    """
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_completion_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": False,
    }


__all__ = ["guidance_chat_request", "openai_action"]
