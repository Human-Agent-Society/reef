"""Validate scripted pi trials before handing them to the sandbox's SDK driver."""

from __future__ import annotations


def trial_script(value: object) -> dict[str, object]:
    """Accept a bounded sequence of prompts, fresh sessions and observable assertions."""
    if not isinstance(value, dict) or set(value) - {"steps", "fixture_tools"}:
        raise ValueError("script must contain steps and optional fixture_tools")
    fixtures = value.get("fixture_tools", [])
    if not isinstance(fixtures, list) or any(not isinstance(name, str) or not name.strip() for name in fixtures):
        raise ValueError("fixture_tools must be an array of tool names")
    if len(fixtures) > 30 or len(set(fixtures)) != len(fixtures):
        raise ValueError("fixture_tools must contain at most 30 distinct names")
    if set(fixtures) & {"read", "write", "edit", "bash", "grep", "find", "ls"}:
        raise ValueError("fixture_tools cannot replace built-in tools")
    steps = value.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 30:
        raise ValueError("script needs between 1 and 30 steps")
    assertions = 0
    list_checks = {
        "tools",
        "forbidden_tools",
        "system_contains",
        "system_excludes",
        "messages_contains",
        "messages_excludes",
        "executed_tools",
        "tool_errors",
    }
    for step in steps:
        if not isinstance(step, dict) or set(step) - {"prompt", "new_session", "tool_call", "expect"}:
            raise ValueError("each step accepts prompt, new_session, tool_call and expect")
        if step.get("new_session") is True:
            if set(step) != {"new_session"}:
                raise ValueError("a new_session step cannot contain other fields")
            continue
        prompt = step.get("prompt")
        if "new_session" in step or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("each prompt step needs nonempty prompt text")
        call = step.get("tool_call")
        if call is not None:
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise ValueError("tool_call needs name and arguments")
            if (
                not isinstance(call["name"], str)
                or not call["name"].strip()
                or not isinstance(call["arguments"], dict)
            ):
                raise ValueError("tool_call needs a nonempty name and an arguments object")
        expect = step.get("expect", {})
        if not isinstance(expect, dict) or set(expect) - (list_checks | {"model_called"}):
            raise ValueError("unknown trial assertion")
        for name, expected in expect.items():
            if name == "model_called":
                if not isinstance(expected, bool):
                    raise ValueError("model_called must be boolean")
            elif not isinstance(expected, list) or any(not isinstance(item, str) for item in expected):
                raise ValueError(f"{name} must be an array of strings")
            if name == "executed_tools" and not set(expected) <= set(fixtures):
                raise ValueError("executed_tools can only assert fixture tool execution")
            assertions += 1
    if not assertions:
        raise ValueError("a scripted trial needs at least one assertion")
    return {"steps": steps, "fixture_tools": fixtures}
