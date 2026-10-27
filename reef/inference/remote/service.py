"""The control plane's view of a remote inference component: its service, backend and control client."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, BinaryIO
from urllib.parse import quote, urlencode

from reef.inference.remote.config import CONTROL_PATH, CONTROL_PROTOCOL, RemoteInferenceSettings
from reef.runtime.deployment import DeploymentResources, InferenceConnection, InferenceService
from reef.runtime.interfaces import InferenceBackend, WeightFiles

#: Bound on one control call other than an engine start; a full-weight upload legitimately takes hours.
CONTROL_TIMEOUT_S = 14_400
HEALTH_TIMEOUT_S = 30


class RemoteControl:
    """Blocking calls to one component's control routes.

    ``url`` is the component's base URL and ``token`` its control token. Each
    call sends one request under ``CONTROL_PATH`` and returns the decoded JSON
    answer; an HTTP error or an unreachable component raises ``RuntimeError``
    with the component's message. Values hold only strings, so they cross
    process boundaries to Reef's serving runtime.
    """

    def __init__(self, url: str, token: str) -> None:
        if not url or not token:
            raise ValueError("a remote control client needs the component URL and its control token")
        self.url = url.rstrip("/")
        self.token = token

    def call(
        self,
        method: str,
        route: str,
        *,
        payload: Mapping[str, Any] | None = None,
        upload: BinaryIO | None = None,
        upload_bytes: int = 0,
        query: Mapping[str, str] | None = None,
        timeout_s: float = CONTROL_TIMEOUT_S,
    ) -> Any:
        """Send one control request with a JSON ``payload`` or a streamed ``upload`` of ``upload_bytes``."""
        headers = {"Authorization": f"Bearer {self.token}"}
        body: bytes | BinaryIO | None = None
        if upload is not None:
            body = upload
            headers.update({"Content-Type": "application/x-tar", "Content-Length": str(upload_bytes)})
        elif payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        suffix = f"?{urlencode(query)}" if query else ""
        request = urllib.request.Request(
            f"{self.url}{CONTROL_PATH}/{route}{suffix}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"remote inference {method} {route} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"remote inference component at {self.url} is unreachable: {exc.reason}") from exc


class RemoteInferenceBackend(InferenceBackend):
    """Receiver operations of a remote component, called over its control routes.

    ``load_weight_files`` packs the directory into a tar file on the control
    plane's node and uploads it; the component unpacks and loads it.
    ``recover`` also covers a component that restarted and so lost the
    engines of ``service``: it starts them again and initializes them with
    the last runtime load ID that ``initialize_version`` set. ``pause`` then
    has no engine to pause and returns, because the restarted engines start
    paused.
    """

    def __init__(self, control: RemoteControl, service: RemoteInferenceService) -> None:
        self.control = control
        self.service = service
        self.initial_version: str | None = None

    def initialize_version(self, runtime_load_id: str) -> None:
        self.control.call("POST", "initialize-version", payload={"runtime_load_id": runtime_load_id})
        self.initial_version = runtime_load_id

    def inference_url(self) -> str:
        """The component itself: it forwards inference requests to its engines."""
        return self.control.url

    def runtime_load_ids(self) -> Sequence[str]:
        answer = self.control.call("GET", "runtime-load-ids")
        versions = answer.get("runtime_load_ids") if isinstance(answer, Mapping) else None
        if not isinstance(versions, list) or not all(isinstance(version, str) for version in versions):
            raise RuntimeError(f"remote inference returned invalid runtime load IDs: {answer!r}")
        return versions

    def pause(self) -> None:
        # A component that lost the engines runs none that generate; ``recover`` starts them paused.
        if self.service.engines_lost():
            return
        self.control.call("POST", "pause")

    def resume(self) -> None:
        self.control.call("POST", "resume")

    def recover(self) -> int:
        if self.service.engines_lost():
            engines = self.service.restart_engines()
            if self.initial_version is not None:
                self.initialize_version(self.initial_version)
            return engines
        answer = self.control.call("POST", "recover")
        replaced = answer.get("replaced") if isinstance(answer, Mapping) else None
        if isinstance(replaced, bool) or not isinstance(replaced, int) or replaced < 0:
            raise RuntimeError(f"remote inference returned an invalid replaced-engine count: {answer!r}")
        return replaced

    def abort(self) -> None:
        self.control.call("POST", "abort")

    def unload_adapter(self, name: str) -> None:
        self.control.call("DELETE", f"adapters/{quote(name, safe='')}")

    def load_weight_files(self, files: WeightFiles, *, adapter_name: str | None, runtime_load_id: str | None) -> None:
        if (adapter_name is None) != (files.layout == "full"):
            raise ValueError("an adapter name is given exactly for adapter weight files")
        if not files.path.is_dir():
            raise RuntimeError(f"weight files directory does not exist: {files.path}")
        query: dict[str, str] = {"layout": files.layout}
        if adapter_name is not None:
            query["adapter_name"] = adapter_name
        if runtime_load_id is not None:
            query["runtime_load_id"] = runtime_load_id
        with tempfile.TemporaryFile() as archive:
            with tarfile.open(fileobj=archive, mode="w") as tar:
                for entry in sorted(files.path.iterdir()):
                    tar.add(entry, arcname=entry.name)
            size = archive.tell()
            archive.seek(0)
            self.control.call("PUT", "weights", upload=archive, upload_bytes=size, query=query)


class RemoteInferenceService(InferenceService):
    """Start engines on a remote component and control them over HTTP.

    ``start`` asks the component to start ``engine_config`` with the backend
    that ``engine`` names (``inference.backend``) on ``num_gpus`` GPUs, replacing any engines
    it runs, and returns a connection without a control executor. Weight
    files reach the engines through :class:`RemoteInferenceBackend`.
    ``instance`` is the component process that runs the engines; the engines
    are lost when the component reports another process or no engines
    (``engines_lost``), and ``restart_engines`` starts them again.
    ``close`` stops the engines.
    """

    connection_protocol = CONTROL_PROTOCOL

    def __init__(
        self,
        settings: RemoteInferenceSettings,
        *,
        engine: str,
        engine_config: Mapping[str, Any],
        num_gpus: int,
        control_token: str,
    ) -> None:
        if not engine:
            raise ValueError("remote inference needs the engine backend that the component starts")
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus <= 0:
            raise ValueError("remote inference needs a positive num_gpus")
        self.settings = settings
        self.engine = engine
        self.engine_config = dict(engine_config)
        self.num_gpus = num_gpus
        self.control = RemoteControl(settings.url, control_token)
        self.started = False
        self.closed = False
        self.instance: str | None = None

    def start(self, resources: DeploymentResources | None = None) -> InferenceConnection:
        """Start the engines; ``resources`` are unused, since the component reserves its own GPUs."""
        if self.started or self.closed:
            raise RuntimeError("inference service can only be started once")
        self.started = True
        self.restart_engines()
        return InferenceConnection(self.connection_protocol, None)

    def restart_engines(self) -> int:
        """Start the engines on the component, replacing any it runs; return the number of engines."""
        answer = self.control.call(
            "PUT",
            "engines",
            payload={"backend": self.engine, "config": self.engine_config, "num_gpus": self.num_gpus},
            timeout_s=self.settings.startup_timeout_s,
        )
        instance = answer.get("instance") if isinstance(answer, Mapping) else None
        if not isinstance(instance, str) or not instance:
            raise RuntimeError(f"remote inference started engines without naming its process: {answer!r}")
        self.instance = instance
        return max(1, self.num_gpus // int(self.engine_config.get("gpus_per_engine") or self.num_gpus))

    def engines_lost(self) -> bool:
        """Whether the component restarted or stopped the engines; raise for engines that failed."""
        if not self.started or self.closed:
            raise RuntimeError("remote inference service is not running")
        answer = self.control.call("GET", "health", timeout_s=HEALTH_TIMEOUT_S)
        if not isinstance(answer, Mapping) or not isinstance(answer.get("engines"), bool):
            raise RuntimeError(f"remote inference returned an invalid health report: {answer!r}")
        return answer.get("instance") != self.instance or not answer["engines"]

    def backend(self, connection: InferenceConnection) -> InferenceBackend:
        self.require_connection(connection)
        return RemoteInferenceBackend(self.control, self)

    def check_health(self) -> None:
        if self.engines_lost():
            raise RuntimeError("the remote inference component restarted and runs none of this service's engines")

    def poll(self) -> None:
        self.check_health()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.started:
            self.control.call("DELETE", "engines", timeout_s=HEALTH_TIMEOUT_S)

    def require_connection(self, connection: InferenceConnection) -> None:
        if connection.protocol != self.connection_protocol:
            raise ValueError(f"incompatible remote inference connection: {connection.protocol!r}")


def create_inference(config: Mapping[str, Any]) -> RemoteInferenceService:
    """Build the service from a training deployment's inference input.

    ``config`` has ``remote`` (the ``inference.remote`` section), ``engine``
    (the ``inference.backend`` name), ``engine_config`` (that backend's own
    input) and ``num_gpus``. The control token is read from this process's
    environment.
    """
    settings = RemoteInferenceSettings.from_mapping(config["remote"])
    return RemoteInferenceService(
        settings,
        engine=config["engine"],
        engine_config=config["engine_config"],
        num_gpus=config["num_gpus"],
        control_token=settings.token(settings.control_token_env, os.environ),
    )
