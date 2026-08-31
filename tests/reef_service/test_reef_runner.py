from __future__ import annotations

import inspect
import json
import urllib.request

import pytest

from reef_client import ReefClient
from reef.service.deploy import build_parser


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return b'{"ok": true}'


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("inference", "/v1/chat/completions", {"model": "reef", "messages": []}),
        ("report", "/reef/report", {"score": 1.0, "references": ["r1"]}),
    ],
)
def test_client_methods_send_scenario_headers(monkeypatch, method: str, path: str, payload: dict) -> None:
    captured: list[urllib.request.Request] = []

    def fake_urlopen(request, timeout):
        assert timeout == 3.0
        captured.append(request)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = ReefClient("http://service", token="secret", timeout_s=3.0)

    if method == "inference":
        result = client.inference(
            "math",
            path,
            payload,
            recipe="tttd",
            extra_headers={"x-reef-release-id": "checkpoint-v42"},
        )
    else:
        result = getattr(client, method)(
            "math",
            payload,
            recipe="tttd",
            extra_headers={"x-reef-release-id": "checkpoint-v42"},
        )

    request = captured[0]
    assert result == {"ok": True}
    assert request.full_url == f"http://service{path}"
    assert request.headers["X-reef-scenario"] == "math"
    assert "X-reef-request-type" not in request.headers
    assert request.headers["X-reef-recipe"] == "tttd"
    assert request.headers["X-reef-release-id"] == "checkpoint-v42"
    assert request.headers["Authorization"] == "Bearer secret"
    assert json.loads(request.data) == payload


@pytest.mark.unit
def test_client_has_no_control_method() -> None:
    client = ReefClient("http://service")

    assert not hasattr(client, "control")


@pytest.mark.unit
@pytest.mark.parametrize(
    "method",
    ["inference", "inference_with_record", "report"],
)
def test_client_recipe_stays_optional(method: str) -> None:
    parameter = inspect.signature(getattr(ReefClient, method)).parameters["recipe"]

    assert parameter.default is None


@pytest.mark.unit
def test_client_has_no_lifecycle_registration_methods() -> None:
    client = ReefClient("http://service")

    assert not hasattr(client, "start_session")
    assert not hasattr(client, "start_attempt")
    assert not hasattr(client, "local_session")


@pytest.mark.unit
def test_service_cli_has_no_repository_argument() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert set(vars(args)) == {"config"}
    assert not hasattr(args, "artifact_repo")
    assert not hasattr(args, "initial_artifact")
    assert not hasattr(args, "recipe_config")


@pytest.mark.unit
@pytest.mark.parametrize(
    "removed_option",
    [
        ["--serve-only"],
        ["--default-recipe", "recipe"],
        ["--recipe", "openclawrl"],
        ["--port", "9000"],
    ],
)
def test_service_cli_rejects_removed_process_options(removed_option: list[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(removed_option)
