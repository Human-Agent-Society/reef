"""The model bindings Reef hands to harness-evolution methods and runs.

A :class:`ModelBinding` is one model endpoint as a value: where it is, which
model to request, how to authenticate, and which API dialect it speaks. The
recipe derives the served model's binding from its inference runtime, so the
URL and credential come from the deployment config, never from the method's
own environment reads; a method's auxiliary models (a stronger proposer, a
judge) are declared in the recipe config and resolved the same way.

:class:`ModelBindings` is the set a method receives: ``served`` is the model
under test - what the HTTP service proxies to and what evaluation runs
run against - and the named ones are the method's own. Methods call
:meth:`ModelBinding.chat`; ``render_tree`` merges
:meth:`ModelBinding.config_overlays` into the config files of each evaluation
run and install. A served
binding points at this Reef's own evaluation route for the scenario when the
service is known, so a run samples the release the scenario serves;
such calls are served without being recorded, so none of this traffic
becomes a scenario record either way.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from http.client import IncompleteRead
from typing import Any

from reef.core.errors import ReefError
from reef.core.model_metadata import ModelMetadata
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.inference.provider_protocol import (
    fold_anthropic_stream,
    fold_chat_stream,
    fold_responses_stream,
    provider_headers,
    stream_events,
    usage_of,
)
from reef.runtime.interfaces import InferenceRuntime

#: The API dialects a binding can speak. ``openai`` is Chat Completions,
#: ``responses`` is OpenAI Responses, and ``anthropic`` is Messages.
MODEL_APIS = ("openai", "responses", "anthropic")

#: What a run's rendered config carries where the key goes when the
#: endpoint needs none (a local vllm or ollama, the tutorial's case). Agent
#: binaries read the rendered key to decide a provider is usable and refuse
#: to start on an empty one - pi exits 1 with "No API key found for the
#: selected model" before it ever reaches the endpoint, so every run
#: fails and no gate can publish. Endpoints without auth ignore the value.
NO_KEY_PLACEHOLDER = "no-key"

ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


class ModelBindingError(ReefError):
    """A model call failed; ``status`` carries the HTTP status when there was one."""

    def __init__(self, message: str, *, status: int | None = None, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


@lru_cache(maxsize=128)
def provider_model_metadata(base_url: str, model: str, api_key: str | None) -> ModelMetadata | None:
    """Read the selected model's OpenRouter-compatible /models metadata, cached per endpoint and key.

    Standard OpenAI model lists have no capabilities; those leave metadata unknown.
    HTTP and parse failures are handled by the caller and are not cached.
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url}/v1/models", headers=headers)
    with urllib.request.urlopen(request, timeout=5.0) as response:
        catalog = json.load(response)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
        raise ValueError("model list must contain a data array")
    for item in catalog["data"]:
        if not isinstance(item, dict) or item.get("id") != model:
            continue
        window = item.get("context_length")
        parameters = item.get("supported_parameters")
        if window is None or parameters is None:
            return None
        if not isinstance(parameters, list) or not all(isinstance(parameter, str) for parameter in parameters):
            raise ValueError("model supported_parameters must be a list of strings")
        return ModelMetadata(context_window=window, reasoning="reasoning" in parameters)
    return None


@dataclass(frozen=True)
class ModelBinding:
    """One model endpoint plus the model name to request from it.

    ``base_url`` carries no ``/v1`` suffix; the request paths add it.
    """

    base_url: str
    model: str
    api_key: str | None = None
    api: str = "openai"
    timeout_s: float = 600.0
    #: The reply budget a harness bound to this endpoint asks for, in tokens. A reasoning model spends it on its
    #: reasoning first and answers with no text when it runs out, which reads to the harness as a turn that ended:
    #: the default leaves room for both. A harness that sets none of its own picks a smaller one (pi takes 16384).
    max_output_tokens: int = 32000
    metadata: ModelMetadata | None = None

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("model binding requires a base_url")
        if not self.model:
            raise ValueError("model binding requires a model name")
        if self.api not in MODEL_APIS:
            raise ValueError(f"model binding api must be one of {MODEL_APIS}, got {self.api!r}")
        if self.timeout_s <= 0:
            raise ValueError("model binding timeout_s must be positive")
        if isinstance(self.max_output_tokens, bool) or self.max_output_tokens <= 0:
            raise ValueError("model binding max_output_tokens must be a positive number of tokens")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @classmethod
    def from_runtime(cls, runtime: InferenceRuntime, *, model: str | None = None) -> ModelBinding:
        """Bind to ``runtime``'s endpoint; ``model`` overrides its model path."""

        name = model or runtime.model_path
        if not name:
            raise ValueError("model binding requires a model name: set reef.upstream_model or the recipe's model.path")
        return cls(
            base_url=runtime.base_url,
            model=name,
            api_key=runtime.provider_key,
            api=runtime.api,
            timeout_s=float(runtime.inference_timeout_s),
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        environ: Mapping[str, str],
        *,
        where: str = "model",
    ) -> ModelBinding:
        """A binding from a config section: ``url``, ``model``, optional
        ``api`` (default ``openai``), ``timeout_s``, and the credential as a
        literal ``api_key`` or the name of an environment variable in
        ``api_key_env`` (so the key itself stays out of the file)."""

        def text(key: str, *, required: bool = True) -> str | None:
            value = config.get(key)
            if value is None and not required:
                return None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{where}.{key} must be a non-empty string")
            return value.strip()

        api_key = text("api_key", required=False)
        env_name = text("api_key_env", required=False)
        if api_key is None and env_name is not None:
            api_key = environ.get(env_name, "").strip() or None
        timeout_raw = config.get("timeout_s", 600.0)
        try:
            timeout_s = float(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}.timeout_s must be a number") from exc
        try:
            return cls(
                base_url=text("url") or "",
                model=text("model") or "",
                api_key=api_key,
                api=text("api", required=False) or "openai",
                timeout_s=timeout_s,
                metadata=ModelMetadata.from_config(config["metadata"]) if "metadata" in config else None,
            )
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc

    def with_metadata(self, metadata: ModelMetadata | None = None) -> ModelBinding:
        """Resolve capabilities before replacing the provider URL with Reef's evaluation route."""
        selected = metadata or self.metadata
        if selected is None:
            try:
                selected = provider_model_metadata(self.base_url, self.model, self.api_key)
            except (IncompleteRead, OSError, ValueError):
                logging.getLogger(__name__).warning(
                    "Could not read metadata for model %r; configure evolution.model_metadata for Codex", self.model
                )
        return replace(self, metadata=selected)

    # -- Method-side calls ---------------------------------------------------

    def chat(self, messages: Sequence[Mapping[str, Any]], *, timeout_s: float | None = None, **params: Any) -> str:
        """One chat completion; returns the assistant text.

        ``messages`` use the OpenAI role shape (``system`` / ``user`` /
        ``assistant``) whatever the binding's API; the request is translated
        for the dialect. ``params`` are request fields (``temperature``,
        ``max_tokens``, ``stream``...) passed through. ``timeout_s`` overrides
        the binding's timeout for this call.
        """

        if self.api == "responses":
            responses_params = dict(params)
            if "max_tokens" in responses_params:
                if "max_output_tokens" in responses_params:
                    raise ValueError("responses chat cannot set both max_tokens and max_output_tokens")
                responses_params["max_output_tokens"] = responses_params.pop("max_tokens")
            response = self.complete(
                {"input": [dict(message) for message in messages], **responses_params}, timeout_s=timeout_s
            )
            pieces = [
                content.get("text", "")
                for item in response.get("output", ())
                if isinstance(item, Mapping) and item.get("type") == "message" and item.get("role") == "assistant"
                for content in item.get("content", ())
                if isinstance(content, Mapping) and content.get("type") == "output_text"
            ]
            if not all(isinstance(piece, str) for piece in pieces) or not pieces:
                raise ModelBindingError(f"model endpoint returned no completion: {response!r}"[:600])
            return "".join(pieces)

        if self.api == "anthropic":
            system = "\n\n".join(
                str(message.get("content", "")) for message in messages if message.get("role") == "system"
            )
            body: dict[str, Any] = {
                "max_tokens": ANTHROPIC_DEFAULT_MAX_TOKENS,
                "messages": [dict(message) for message in messages if message.get("role") != "system"],
                **params,
            }
            if system:
                body["system"] = system
            response = self.complete(body, timeout_s=timeout_s)
            try:
                return "".join(
                    str(block.get("text", "")) for block in response["content"] if block.get("type") == "text"
                )
            except (KeyError, TypeError, AttributeError) as exc:
                raise ModelBindingError(f"model endpoint returned no completion: {response!r}"[:600]) from exc

        response = self.complete({"messages": [dict(message) for message in messages], **params}, timeout_s=timeout_s)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelBindingError(f"model endpoint returned no completion: {response!r}"[:600]) from exc
        if not isinstance(content, str):
            raise ModelBindingError("model endpoint returned non-text content")
        return content

    def last_usage(self) -> dict[str, int] | None:
        """The tokens the latest ``complete`` reported (``usage_of`` of its response), or None."""
        return getattr(self, "_last_usage", None)

    def last_response(self) -> dict[str, Any] | None:
        """The latest ``complete`` response, including provider reasoning when returned.

        ``chat`` still returns only assistant text. A recording wrapper can
        retain this response without making a second call or losing native
        provider fields. None before a call or after a failed request.
        """
        return getattr(self, "_last_response", None)

    def complete(self, body: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        """POST one request in the binding's native dialect and return the
        response object: Chat Completions for ``openai``, Responses for
        ``responses``, and Messages for ``anthropic``. ``model`` defaults to
        this binding's. A streaming request is read to the end and folded into
        the non-streaming response shape, so callers see one contract either
        way. Prefer :meth:`chat` unless the method needs the raw response.
        """

        object.__setattr__(self, "_last_response", None)
        object.__setattr__(self, "_last_usage", None)
        request_body = {"model": self.model, **body}
        if self.api == "anthropic":
            path = "/v1/messages"
        else:
            path = "/v1/responses" if self.api == "responses" else "/v1/chat/completions"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"content-type": "application/json", **provider_headers(path, self.api_key)},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                if not request_body.get("stream"):
                    result = json.load(response)
                else:
                    # Read no further than the event that ends the reply: a proxy may keep the connection open.
                    events = stream_events(response)
                    if self.api == "anthropic":
                        result = fold_anthropic_stream(events)
                    elif self.api == "responses":
                        result = fold_responses_stream(events)
                    else:
                        result = fold_chat_stream(events)
            # The tokens this call reported, for a wrapper of ``chat`` that never sees the response body.
            object.__setattr__(self, "_last_usage", usage_of(result))
            object.__setattr__(self, "_last_response", result)
            return result
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(errors="replace")
            except Exception:
                detail = ""
            raise ModelBindingError(
                f"model endpoint answered {exc.code}: {detail[:400]}",
                status=exc.code,
                detail=detail,
            ) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ModelBindingError(f"model endpoint unreachable: {exc}") from exc

    # -- Run-side rendering ----------------------------------------------

    def config_overlays(
        self, descriptor: AdapterDescriptor, *, models: Sequence[str] = ()
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        """``(config target, data)`` pairs that point ``descriptor``'s harness at this binding, in template order.

        ``render_tree`` merges each ``data`` into its target after the tree's
        own config, so the binding wins every key it writes. ``models`` are
        further model names the harness may pick from: every template entry
        that names ``{model}`` (a mapping key or a list item) is repeated once
        per model, this binding's first, while a plain ``{model}`` elsewhere,
        the default, stays this binding's.
        """

        templates = descriptor.model_binding.get(self.api)
        if not templates:
            declared = sorted(descriptor.model_binding)
            known = ", ".join(declared) or "none"
            # The dialect is the upstream's: reef serve takes it as --inference.upstream-api.
            hint = (
                f": serve with --inference.upstream-api {declared[0]} on an upstream that speaks it"
                if declared
                else ""
            )
            raise ModelBindingError(
                f"adapter {descriptor.name!r} declares no model_binding for the {self.api!r} api "
                f"(declared: {known}); runs cannot reach a model{hint}"
            )
        metadata = {}
        if self.metadata is not None:
            metadata[self.model] = {
                "context_window": self.metadata.context_window,
                "reasoning": self.metadata.reasoning,
            }
        values: dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": self.api_key or NO_KEY_PLACEHOLDER,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "model_metadata": metadata,
        }
        choices = [self.model]
        for name in models:
            if isinstance(name, str) and name and name not in choices:
                choices.append(name)
        return tuple(
            (str(template["target"]), _substitute(template["data"], values, choices)) for template in templates
        )


@dataclass(frozen=True)
class ModelBindings(Mapping[str, ModelBinding]):
    """The models a method may call: ``served`` plus the method's named ones.

    ``served`` is the model under test - the one the HTTP service proxies to
    and evaluation tasks run against. The named bindings come from the
    recipe's ``evolution.models`` section; ``models["teacher"]`` looks one up
    and raises :class:`KeyError` naming what is declared when it is missing.
    """

    served: ModelBinding
    named: Mapping[str, ModelBinding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "served" in self.named:
            raise ValueError("'served' is reserved for the model under test; name the extra model differently")

    def __getitem__(self, name: str) -> ModelBinding:
        if name == "served":
            return self.served
        try:
            return self.named[name]
        except KeyError:
            declared = ", ".join(sorted(self.named)) or "none"
            raise KeyError(f"no model named {name!r}; declared under evolution.models: {declared}") from None

    def __iter__(self) -> Iterator[str]:
        yield "served"
        yield from self.named

    def __len__(self) -> int:
        return 1 + len(self.named)


def _mentions_model(value: Any) -> bool:
    if isinstance(value, str):
        return "{model}" in value
    if isinstance(value, Mapping):
        return any(_mentions_model(key) or _mentions_model(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_mentions_model(item) for item in value)
    return False


def _substitute(value: Any, values: Mapping[str, Any], models: Sequence[str] = ()) -> Any:
    """Fill ``{placeholders}``; with ``models``, a mapping key or list item naming ``{model}`` is repeated per model.

    A string that is one placeholder and nothing else becomes the value itself, so a number stays a number in the
    rendered config; a placeholder among other text is filled as written.
    """
    if isinstance(value, str):
        whole = values.get(value[1:-1]) if value[:1] == "{" and value[-1:] == "}" else None
        if whole is not None and not isinstance(whole, str):
            return whole
        for key, replacement in values.items():
            value = value.replace("{" + key + "}", str(replacement))
        return value
    if isinstance(value, Mapping):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if len(models) > 1 and isinstance(key, str) and "{model}" in key:
                for model in models:
                    each = {**values, "model": model}
                    out[_substitute(key, each)] = _substitute(item, each)
            else:
                out[_substitute(key, values, models)] = _substitute(item, values, models)
        return out
    if isinstance(value, list):
        out_list: list[Any] = []
        for item in value:
            if len(models) > 1 and _mentions_model(item):
                out_list.extend(_substitute(item, {**values, "model": model}) for model in models)
            else:
                out_list.append(_substitute(item, values, models))
        return out_list
    return value


__all__ = ["MODEL_APIS", "ModelBinding", "ModelBindingError", "ModelBindings"]
