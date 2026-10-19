"""Training admission and queued requests share the same payload contract."""

from dataclasses import replace

import pytest

from reef.core import RequestType
from reef.core.training_request import TrainingRequest, normalize_training_request_payload
from reef.service.request_service import normalize_request_payload


def test_request_normalization_preserves_text_and_cleans_optional_fields() -> None:
    payload = {
        "text": "  Improve the harness\n",
        "session": "session-1",
        "release_id": "release-1",
        "id": "untrusted-id",
        "requires": [{"name": "TOKEN", "kind": "env", "prompt": "  Set the token  ", "extra": True}],
        "client": {
            "platform": "  darwin  ",
            "release": "not; a release",
            "commands": {"git": True, "node": False, "bad name": True, "python3": "yes"},
            "hostname": "not requested",
        },
    }
    expected = {
        "text": "  Improve the harness\n",
        "session": "session-1",
        "release_id": "release-1",
        "requires": [{"name": "TOKEN", "kind": "env", "prompt": "Set the token"}],
        "client": {"platform": "darwin", "commands": {"git": True, "node": False}},
    }
    assert normalize_training_request_payload(payload) == expected
    assert normalize_request_payload(RequestType.TRAIN, payload) == (expected, ())
    request = TrainingRequest.from_dict(payload, request_id="record-1")
    assert request.id == "record-1"
    assert request.to_dict() == expected
    assert TrainingRequest.from_dict(payload).id == ""
    assert replace(request, id="record-2").to_dict() == expected
    assert payload["requires"][0]["prompt"] == "  Set the token  "


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("text", None, "text must be a string"),
        ("text", " \n ", "text must be a non-empty string"),
        ("text", "x" * 4001, "text must not exceed 4000 characters"),
        ("session", None, "session must be a string"),
        ("release_id", 1, "release_id must be a string"),
        ("requires", "TOKEN", "requires must be a list"),
    ],
)
def test_admission_and_queued_requests_reject_invalid_fields(field: str, value: object, message: str) -> None:
    payload = {"text": "Improve the harness", "session": "s", "release_id": "r", field: value}
    with pytest.raises(ValueError, match=message):
        normalize_training_request_payload(payload)
    with pytest.raises(ValueError, match=message):
        TrainingRequest.from_dict(payload)


@pytest.mark.parametrize("client", [None, "darwin", {}, {"commands": {"bad name": True}}])
def test_absent_requirements_and_invalid_client_keep_the_existing_wire_shape(client: object) -> None:
    payload = {"text": "x" * 4000, "session": "", "release_id": "", "requires": None, "client": client}
    expected = {"text": "x" * 4000, "session": "", "release_id": "", "requires": []}
    assert normalize_training_request_payload(payload) == expected
    assert TrainingRequest.from_dict(payload).to_dict() == expected


@pytest.mark.parametrize("field", ["text", "session", "release_id"])
def test_required_fields_cannot_be_omitted(field: str) -> None:
    payload = {"text": "Improve the harness", "session": "s", "release_id": "r"}
    payload.pop(field)
    with pytest.raises(ValueError, match=f"{field} must be a string"):
        normalize_training_request_payload(payload)
    with pytest.raises(ValueError, match=f"{field} must be a string"):
        TrainingRequest.from_dict(payload)


def test_client_command_report_is_bounded() -> None:
    commands = {f"tool-{index}": True for index in range(65)}
    normalized = normalize_training_request_payload(
        {"text": "Improve the harness", "session": "s", "release_id": "r", "client": {"commands": commands}}
    )
    assert normalized["client"] == {"commands": {f"tool-{index}": True for index in range(64)}}
