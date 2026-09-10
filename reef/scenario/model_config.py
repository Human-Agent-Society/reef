"""Private model settings supplied when a scenario is created or updated."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from reef.runtime.adapters.inference_proxy import PROVIDER_APIS, InferenceProxyRuntime


class ScenarioModelConfig:
    """One replaceable endpoint value; readers retain their existing runtime snapshot."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._runtime: InferenceProxyRuntime | None = None
        if path is not None and path.exists():
            # Invalid persisted settings fail recovery instead of selecting another model.
            self._runtime = self._parse(json.loads(path.read_text()))

    @property
    def runtime(self) -> InferenceProxyRuntime | None:
        return self._runtime

    @staticmethod
    def _parse(value: object) -> InferenceProxyRuntime | None:
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) - {"url", "model", "api", "api_key"}:
            raise ValueError("model must be null or an object with url, model, api and api_key")
        for name in ("url", "model"):
            item = value.get(name)
            if not isinstance(item, str) or not item.strip() or any(ord(c) < 32 for c in item):
                raise ValueError(f"model.{name} must be a non-empty string without control characters")
        parsed = urlparse(value["url"])
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("model.url must be an HTTP(S) URL without credentials, query or fragment")
        api = value.get("api", "openai")
        if api not in PROVIDER_APIS:
            raise ValueError("model.api must be openai, responses or anthropic")
        key = value.get("api_key")
        if key is not None and (not isinstance(key, str) or any(ord(c) < 32 for c in key)):
            raise ValueError("model.api_key must be a string without control characters")
        return InferenceProxyRuntime(
            base_url=value["url"].strip(), model_path=value["model"].strip(), api=api, api_key=key
        )

    def save(self, value: object) -> None:
        runtime = self._parse(value)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self._path.parent, prefix=".model-")
            try:
                with os.fdopen(descriptor, "w") as output:
                    json.dump(value, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self._path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        self._runtime = runtime

    def view(self) -> dict[str, object] | None:
        """The applied configuration for API acknowledgements, without its credential."""
        runtime = self._runtime
        if runtime is None:
            return None
        return {
            "url": runtime.base_url,
            "model": runtime.model_path,
            "api": runtime.api,
            "has_api_key": bool(runtime.api_key),
        }
