"""Settings and HTTP routes shared by the remote inference component and its clients."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlsplit

from reef.core.values import JsonValue

#: Every control route is under this path. Any other path is an inference request for the engines.
CONTROL_PATH = "/reef/control/v1"
#: The control request header with the sender's serving-owner sequence, a positive decimal integer.
OWNER_SEQUENCE_HEADER = "x-reef-owner-sequence"
DEFAULT_CONTROL_TOKEN_ENV = "REEF_REMOTE_CONTROL_TOKEN"
DEFAULT_INFERENCE_TOKEN_ENV = "REEF_REMOTE_INFERENCE_TOKEN"

#: The options of each built-in engine that keep KV outside the engine: a storage backend, or a KV connector.
#: The pause before a change of weights clears the engine's own prefix cache only, which includes SGLang's
#: host-memory tier (``enable_hierarchical_cache``).
EXTERNAL_KV_CACHE_OPTIONS: dict[str, frozenset[str]] = {
    "sglang": frozenset({"hicache_storage_backend", "enable_lmcache"}),
    "vllm": frozenset({"kv_transfer_config", "kv_offloading_size", "kv_offloading_backend"}),
}


def require_no_external_kv_cache(engine: str, config: Mapping[str, Any]) -> None:
    """Refuse engine input whose KV cache can outlive a change of weights.

    ``engine`` is "sglang" or "vllm". ``config`` is the input of that engine's inference service; its ``options``
    hold the engine's native options, whose names may use dashes or underscores. Raise ``ValueError`` when an
    ``EXTERNAL_KV_CACHE_OPTIONS`` name of ``engine`` has a value other than None and False, which are the values
    of an option that is off.

    A component that keeps each request on its weights applies this before it starts engines, and so does the
    runtime that asks for that protection. Rollout engines and plain endpoints claim no binding to published
    weights, so they do not apply it.
    """
    names = {
        name.replace("-", "_")
        for name, value in config.get("options", {}).items()
        if value is not None and value is not False
    }
    if external := sorted(names & EXTERNAL_KV_CACHE_OPTIONS[engine]):
        raise ValueError(
            f"no pause clears a KV cache outside the {engine} engines, so engines that keep each request on its "
            f"weights cannot use these options: {', '.join(external)}"
        )


def component_url(value: object, setting: str) -> str:
    """The base URL ``value`` of an inference component, without trailing slashes.

    Raise ``ValueError``, which names ``setting``, unless ``value`` is an HTTP(S) URL string with a host and
    without credentials, query or fragment.
    """
    if not isinstance(value, str):
        raise ValueError(f"{setting} must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{setting} must be an HTTP(S) URL without credentials, query or fragment")
    return value.rstrip("/")


@dataclass(frozen=True)
class RemoteInferenceSettings:
    """How the control plane reaches a deployment's remote inference components.

    - ``url`` is the first component's HTTP(S) base URL, without credentials, query or fragment and without
      trailing slashes.
    - ``slot_urls`` holds the base URLs of further components, each under the rules of ``url`` and different from
      ``url`` and from the others. It is empty by default; a list is read as a tuple. Each URL names a separate
      component that runs the same model, engine and options as the first one, on its own GPUs
      (``inference.num-gpus`` counts those of one component). Only production serving of full-weight checkpoints
      uses them: each component serves one checkpoint at a time, so a deployment serves as many checkpoints at
      once as it has URLs. A component's engines stay allocated while it serves no checkpoint.
    - ``control_token_env`` and ``inference_token_env`` name two different
      environment variables. They hold the bearer tokens for control routes
      and for inference requests, which every component accepts.
    - ``startup_timeout_s`` is positive and bounds one engine start, including
      the model download on the component.
    - ``model_path`` is the model that each component loads: a Hugging Face
      repository ID or a path on the component's host, or None before
      deployment resolution. Resolution fills it with ``inference.model-path``
      as the file names it, before this host downloads anything, because a
      local snapshot path means nothing on the component's host.

    The values come from the ``inference.remote`` section, whose presence runs
    the engines that ``inference.backend`` names on the components. Service
    deployment validates them; the remote inference service reads them.
    """

    url: str
    control_token_env: str = DEFAULT_CONTROL_TOKEN_ENV
    inference_token_env: str = DEFAULT_INFERENCE_TOKEN_ENV
    startup_timeout_s: float = 3600.0
    model_path: str | None = None
    slot_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", component_url(self.url, "inference.remote.url"))
        # A string is a sequence of characters, not a list of URLs.
        if not isinstance(self.slot_urls, (list, tuple)):
            raise ValueError("inference.remote.slot-urls must be a list of URLs")
        slot_urls = tuple(
            component_url(url, f"inference.remote.slot-urls[{index}]") for index, url in enumerate(self.slot_urls)
        )
        repeated = sorted({url for url in slot_urls if url == self.url or slot_urls.count(url) > 1})
        if repeated:
            raise ValueError(
                "inference.remote.slot-urls must name components other than inference.remote.url and each other; "
                f"it repeats {', '.join(repeated)}"
            )
        object.__setattr__(self, "slot_urls", slot_urls)
        for name, value in (
            ("control-token-env", self.control_token_env),
            ("inference-token-env", self.inference_token_env),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"inference.remote.{name} must be a non-empty string")
        if self.control_token_env == self.inference_token_env:
            raise ValueError("inference.remote control and inference tokens must use different variables")
        if (
            isinstance(self.startup_timeout_s, bool)
            or not isinstance(self.startup_timeout_s, (int, float))
            or not math.isfinite(self.startup_timeout_s)
            or self.startup_timeout_s <= 0
        ):
            raise ValueError("inference.remote.startup-timeout-s must be a positive number")
        if self.model_path is not None and (not isinstance(self.model_path, str) or not self.model_path.strip()):
            raise ValueError("inference.remote.model-path must be a non-empty string")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RemoteInferenceSettings:
        """Read the public section; keys may use dashes or underscores."""
        if not isinstance(value, Mapping):
            raise ValueError("inference.remote must be an object")
        values = {str(key).replace("-", "_"): item for key, item in value.items()}
        unknown = sorted(set(values) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"unknown inference.remote fields: {', '.join(unknown)}")
        if "url" not in values:
            raise ValueError("inference.remote requires a url")
        return cls(**values)

    def to_dict(self) -> dict[str, JsonValue]:
        """The settings as JSON values under their field names; ``slot_urls`` is a list.

        ``from_mapping`` reads the mapping back as equal settings, also after a JSON or YAML round trip. The mapping
        names the token variables and holds no token.
        """
        slot_urls: list[JsonValue] = list(self.slot_urls)
        return {
            "url": self.url,
            "control_token_env": self.control_token_env,
            "inference_token_env": self.inference_token_env,
            "startup_timeout_s": self.startup_timeout_s,
            "model_path": self.model_path,
            "slot_urls": slot_urls,
        }

    def token(self, variable: str, environ: Mapping[str, str]) -> str:
        """The bearer token in ``variable``; a missing or empty value is an error."""
        token = environ.get(variable, "").strip()
        if not token:
            raise ValueError(f"the remote inference component requires the environment variable {variable}")
        return token
