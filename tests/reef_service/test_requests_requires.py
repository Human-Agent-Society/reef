"""The tutorial proposer's ``requires``: the request prompt names the shape, a ``{"requires": [...]}`` object in
the reply lands on the request mapping the backend handed over, and a malformed item is dropped and recorded."""

from __future__ import annotations

import json
from types import MappingProxyType, ModuleType

import pytest
from reef_service.test_harness_example import NODES, REQUEST, _method, canned, request_reply

REQUIRES_OBJECT = {
    "requires": [
        {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"},
        {"name": "notify", "kind": "permission", "check": "osascript -e 'display notification \"x\"'"},
        {"name": "twilio", "kind": "service"},
    ]
}
RULES = {"id": "brief", "name": "rules", "config": {"text": "Be brief."}}


@pytest.fixture
def evolution(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _method(monkeypatch, "evolution")


def test_the_request_prompt_says_how_to_name_what_the_change_needs(evolution) -> None:
    model = canned(request_reply(RULES))
    evolution.propose(NODES, (), model, requests=(dict(REQUEST),))
    prompt = model.prompt
    # One worked example per kind, each with a prompt; an env item is the variable name alone, its value never
    # written into the tree.
    assert '{"requires": [...]}' in prompt
    for kind in ("permission", "env", "service"):
        assert f'"kind": "{kind}"' in prompt
    assert prompt.count('"prompt": "') == 3 and "Each item carries a prompt" in prompt
    assert (
        "the value is never written into the tree" in prompt
        and "Omit the object when the change needs nothing" in prompt
    )


def test_the_requires_object_beside_the_entries_is_appended_to_the_request_mapping(evolution) -> None:
    request = {**REQUEST, "requires": [{"name": "existing", "kind": "env"}]}
    code = "export default function (pi) {}\n"
    reply = request_reply(
        {"id": "notify", "name": "code_extension", "config": {"name": "notify", "code": code}}, REQUIRES_OBJECT
    )
    (mutation,) = evolution.propose(NODES, (), canned(reply), requests=(request,)).mutations
    assert (mutation.op, mutation.id) == ("create", "notify")
    assert request["requires"] == [{"name": "existing", "kind": "env"}, *REQUIRES_OBJECT["requires"]]
    # No object: nothing is added; a reply with no usable entry is a proposal without mutations, the reason in
    # its notes, and adds nothing either.
    request = dict(REQUEST)
    evolution.propose(NODES, (), canned(request_reply(RULES)), requests=(request,))
    assert "requires" not in request
    request = dict(REQUEST)
    nothing = evolution.propose(NODES, (), canned(json.dumps([REQUIRES_OBJECT])), requests=(request,))
    assert nothing.mutations == () and nothing.notes == {"failure": "the reply holds no usable entry"}
    assert "requires" not in request


def test_a_malformed_requires_item_is_dropped_and_a_read_only_mapping_is_left_alone(evolution) -> None:
    malformed = {
        "requires": [
            {"name": "ok", "kind": "service", "check": "true"},
            {"name": "x", "kind": "secret"},
            {"name": "../escape", "kind": "env"},
            {"name": "empty-check", "kind": "env", "check": ""},
            {"name": "no-kind"},
            "not an object",
        ]
    }
    reply = request_reply(RULES, malformed, {"requires": "not a list"})
    request = dict(REQUEST)
    proposal = evolution.propose(NODES, (), canned(reply), requests=(request,))
    assert request["requires"] == [{"name": "ok", "kind": "service", "check": "true"}]
    # The dropped items are on record with the reason each was refused, so the step page can show them.
    refused = proposal.notes["refused_requires"]
    assert [entry["item"] for entry in refused] == malformed["requires"][1:]
    assert all(entry["reason"] for entry in refused)
    frozen = MappingProxyType(dict(REQUEST))
    (mutation,) = evolution.propose(NODES, (), canned(reply), requests=(frozen,)).mutations
    assert mutation.id == "brief" and "requires" not in frozen
