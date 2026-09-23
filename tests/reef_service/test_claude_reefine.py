"""Reef's shipped entries for the Claude Code adapter: the ``/reefine`` command, the harness reference skill
and the update notice, seeded by the reefine profile's defaults, rendered where Claude Code reads them, and
the text proposer's surface for that adapter.

Claude Code loads no code from a harness tree, so the three entries are a
command file, a skill and a settings hook, and a request for that harness is
answered with the kinds Claude Code reads (skill, rules, agent_command, config)
by a prompt that never mentions pi's extension API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reef_service.test_harness_recipe import batch, make_binary

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.requests import REQUESTS_ENTRY_ID, request_entries, requests_skill_id
from reef.harness.episodes.version_check import CLAUDE_NOTICE_HOOK, VERSION_CHECK_ENTRY_ID, version_check_entry
from reef.harness.tree.nodes import RESERVED_ENTRY_IDS
from reef.harness.tree.render import render_composition
from reef.recipe import build_recipe
from reef.recipe.reefine import ReefineRecipe, evolution
from reef.train.cordis_backend import CordisBackend, Mutation
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

CLAUDE = get_adapter("claude")


def test_the_claude_requests_entries_are_a_command_file_and_a_reference_skill() -> None:
    command, skill = request_entries("claude")
    assert command["id"] == REQUESTS_ENTRY_ID and command["name"] == "agent_command"
    assert command["config"]["name"] == "reefine"
    text = command["config"]["text"]
    assert text.startswith("---\n") and "description:" in text.split("---")[1]
    # The command runs the wrapper by name; run_agent puts the install root on PATH for that.
    assert "reef-claude evolve --wait" in text and "reef-claude install --release" in text
    assert skill["id"] == "reef-claude-harness-api" == requests_skill_id("claude") and skill["name"] == "skill"
    assert skill["id"] in RESERVED_ENTRY_IDS
    assert skill["config"]["text"].startswith("---\nname: reef-claude-harness-api\n")
    assert requests_skill_id("opencode") is None


def test_the_claude_notice_is_a_session_start_hook_that_runs_only_through_the_wrapper() -> None:
    entry = version_check_entry("claude")
    assert entry["id"] == VERSION_CHECK_ENTRY_ID and entry["name"] == "config"
    (group,) = entry["config"]["data"]["hooks"]["SessionStart"]
    assert group["hooks"] == [{"type": "command", "command": CLAUDE_NOTICE_HOOK}]
    # The session runs the wrapper without a permission prompt, so /reefine installs on the person's yes.
    assert entry["config"]["data"]["permissions"] == {"allow": ["Bash(reef-claude *)"]}
    # Without the wrapper in the environment (an episode, a tree run by hand) the hook does nothing.
    assert CLAUDE_NOTICE_HOOK.startswith('if [ -n "$REEF_HARNESS_WRAPPER" ]')
    assert '"$REEF_HARNESS_WRAPPER" notice --hook claude' in CLAUDE_NOTICE_HOOK


def test_the_entries_render_where_claude_code_reads_them_and_the_hook_survives_a_second_config() -> None:
    seed = (version_check_entry("claude"), *request_entries("claude"))
    nodes = [(str(options["name"]), options["config"]) for options in seed]
    files = render_composition(nodes, CLAUDE)
    assert files["claude/commands/reefine.md"] == request_entries("claude")[0]["config"]["text"]
    assert files["claude/skills/reef-claude-harness-api/SKILL.md"] == request_entries("claude")[1]["config"]["text"]
    settings = json.loads(files["claude/settings.json"])
    assert settings["includeCoAuthoredBy"] is False  # the descriptor's defaults stay
    assert settings["hooks"]["SessionStart"][0]["hooks"][0]["command"] == CLAUDE_NOTICE_HOOK
    # An evolved config entry adding its own hooks joins the lists instead of replacing reef's notice.
    evolved = (
        "config",
        {
            "target": "primary",
            "data": {
                "hooks": {
                    "SessionStart": [{"hooks": [{"type": "command", "command": "echo hi"}]}],
                    "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "exit 2"}]}],
                },
                "permissions": {"deny": ["WebFetch"]},
            },
        },
    )
    merged = json.loads(render_composition([*nodes, evolved], CLAUDE)["claude/settings.json"])
    assert [group["hooks"][0]["command"] for group in merged["hooks"]["SessionStart"]] == [
        CLAUDE_NOTICE_HOOK,
        "echo hi",
    ]
    assert merged["hooks"]["PreToolUse"][0]["matcher"] == "Bash" and merged["permissions"]["deny"] == ["WebFetch"]
    assert merged["permissions"]["allow"] == ["Bash(reef-claude *)"]


def test_the_reefine_profile_boots_for_claude_with_its_defaults(tmp_path: Path) -> None:
    """``requests`` and ``version_check`` default to true, and the claude adapter ships both, so a profile that
    names the adapter and nothing else boots; a settings change waits for review like an extension."""
    config = {"evolution": {"adapter": "claude", "tasks": ["[health] Run `echo reef-ok` and reply with its output."]}}
    built = build_recipe("reef.recipe.reefine:ReefineRecipe", {}, config)
    assert isinstance(built, ReefineRecipe) and built.adapter == "claude"
    assert [options["id"] for options in built.seed] == [
        VERSION_CHECK_ENTRY_ID,
        REQUESTS_ENTRY_ID,
        "reef-claude-harness-api",
    ]
    assert built.review_kinds == ("code_extension", "config")
    backend = CordisBackend(
        descriptor=CLAUDE,
        propose=resolve_proposer(lambda nodes, samples, models: None),
        score_episode=resolve_episode_scorer(lambda task, result: 0.0),
        tasks=("probe",),
        # Claude Code speaks the Anthropic messages API, the one dialect the adapter binds.
        models=ModelBinding(base_url="http://localhost:8000", model="qwen3", api_key="dummy", api="anthropic"),
        seed=tuple(built.seed),
        binary=str(make_binary(tmp_path)),
    )
    entries = [dict(entry) for entry in built.seed]
    assert "claude/commands/reefine.md" in backend._render_for_episode(entries)
    result = backend.prepare_step(batch(), {"steps": 1, "entries": entries}, 0)
    assert result.outcome == "skip" and result.metrics["skipped"] == "no proposal"


def test_the_claude_surface_tells_the_proposer_claude_codes_kinds_and_nothing_of_pi() -> None:
    surface = evolution.surface_of("claude")
    prompt = evolution.REQUEST_PROMPT.format(
        request="x",
        machine="",
        failures="",
        entries="[]",
        kinds=surface.kinds,
        guidance=surface.guidance,
        env_read=surface.env_read,
        setup=surface.setup,
        reserved="",
        plan="",
        api="",
    )
    for pi_word in ("pi.registerCommand", "pi extension", "PI_OFFLINE", "reef-pi", "process.env", "code_extension"):
        assert pi_word not in prompt
    for claude_word in ("- config:", "settings.json", "PreToolUse", "$ARGUMENTS", "CLAUDE.md", "reef-claude setup"):
        assert claude_word in prompt
    assert surface.request_kinds == ("skill", "rules", "agent_command", "config")
    assert surface.api_skill == requests_skill_id("claude")
    # pi keeps the prompt it had, and an adapter without a surface of its own gets pi's.
    pi = evolution.surface_of("pi")
    assert "pi.registerCommand" in pi.guidance and pi.request_kinds[-1] == "code_extension"
    assert evolution.surface_of(None) is pi and evolution.surface_of("opencode") is pi
    review = evolution.REVIEW_PROMPT.format(
        request="x", design="d", entries="[]", session_env="CLAUDE_CONFIG_DIR", command_check=surface.command_check
    )
    assert "pi.registerCommand" not in review and "hooks write its" in review


def test_a_config_entry_in_a_reply_parses_with_its_object_body_and_its_target() -> None:
    reply = json.dumps(
        [
            {"design": "a hook"},
            {
                "id": "chat-guard",
                "name": "config",
                "config": {"data": {"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "x"}]}]}}},
            },
            {"name": "config", "config": {"target": "primary", "data": {"permissions": {"deny": ["WebFetch"]}}}},
            {"id": "empty", "name": "config", "config": {"data": {}}},
            {"id": "text-body", "name": "config", "config": {"data": "not an object"}},
        ]
    )
    parsed = evolution._parse_proposal(reply, kinds=("config",))
    assert parsed is not None and len(parsed) == 2
    (named, unnamed) = parsed
    assert named[0] == "chat-guard" and named[1] == "config"
    assert named[2] == {
        "data": {"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "x"}]}]}},
        "target": "primary",
    }
    assert unnamed[0].startswith("config-") and unnamed[2]["data"] == {"permissions": {"deny": ["WebFetch"]}}
    # pi's kinds never take a config entry, whatever the reply carries.
    assert evolution._parse_proposal(reply, kinds=evolution.surface_of("pi").request_kinds) is None


def test_undeclared_env_reads_the_hook_commands_of_a_config_entry() -> None:
    mutations = [
        Mutation(
            "create",
            "chat-guard",
            {
                "name": "config",
                "config": {
                    "target": "primary",
                    "data": {
                        "hooks": {
                            "PreToolUse": [
                                {
                                    "hooks": [
                                        {
                                            "type": "command",
                                            "command": 'test -f "$CLAUDE_CONFIG_DIR/.chat" && say "$CHAT_VOICE" $HOME $REEF_TOKEN',
                                        }
                                    ]
                                }
                            ]
                        }
                    },
                },
            },
        ),
        Mutation("create", "notes", {"name": "rules", "config": {"text": "$NOT_A_HOOK"}}),
    ]
    claude = evolution.surface_of("claude")
    assert evolution._undeclared_env(mutations, [], claude.session_env) == ["CHAT_VOICE"]
    declared = [{"name": "CHAT_VOICE", "kind": "env"}]
    assert evolution._undeclared_env(mutations, declared, claude.session_env) == []


def test_a_flattened_command_file_is_put_back_together_from_its_frontmatter_keys() -> None:
    reply = json.dumps(
        [
            {
                "id": "chat",
                "name": "agent_command",
                "config": {
                    "description": "Enter chat mode",
                    "argument-hint": "off",
                    "allowed-tools": "Bash(touch *)",
                    "disable-model-invocation": True,
                    "body": '!`touch "$CLAUDE_CONFIG_DIR/.chat"`\nYou are in chat mode.',
                },
            },
            {"id": "plain", "name": "agent_command", "config": {"text": "just a prompt"}},
            {"id": "nothing", "name": "agent_command", "config": {"description": "no body at all"}},
        ]
    )
    parsed = evolution._parse_proposal(reply, kinds=("agent_command",))
    assert parsed is not None and [entry_id for entry_id, _, _ in parsed] == ["chat", "plain"]
    text = parsed[0][2]["text"]
    assert text.startswith("---\ndescription: Enter chat mode\nargument-hint: off\nallowed-tools: Bash(touch *)\n")
    assert "disable-model-invocation: true\n---\n!`touch" in text and text.endswith("You are in chat mode.")
    assert parsed[1][2] == {"name": "plain", "text": "just a prompt"}


def test_an_answer_with_a_settings_entry_stands_when_the_review_calls_it_undelivered(monkeypatch) -> None:
    """A hook or a permission rule enforces behavior, so a review that says such an answer only imitates it is
    read as partial: the points stay uncovered, the answer is kept, and pi's surface keeps the old reading."""
    from reef.harness.episodes.model_binding import ModelBinding, ModelBindings

    answer = json.dumps(
        [
            {"design": "a hook. How to use: /chat"},
            {"id": "chat", "name": "agent_command", "config": {"text": "---\ndescription: chat\n---\nhi"}},
            {
                "id": "chat-hooks",
                "name": "config",
                "config": {"data": {"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "exit 2"}]}]}}},
            },
        ]
    )
    review = json.dumps({"result": "partial", "delivers": False, "covered": [], "uncovered": ["one point"]})
    replies = iter(["[]", answer, review, answer, review, answer, review])
    monkeypatch.setattr(evolution, "_ask", lambda models, prompt, **kw: (next(replies), None))
    models = ModelBindings(served=ModelBinding(base_url="http://127.0.0.1:9", model="m"))
    proposal = evolution.propose((), (), models, requests=[{"text": "add /chat"}], entries=(), adapter="claude")
    assert [m.id for m in proposal.mutations] == ["chat", "chat-hooks"]
    assert proposal.notes["review"]["delivers"] is True and proposal.notes["review"]["uncovered"] == ["one point"]
    assert proposal.notes["attempts"] == 3 and "failure" not in proposal.notes
    # The same reviews of a pi answer carrying an extension still end the step with the reason.
    pi_answer = json.dumps(
        [{"design": "x"}, {"id": "chat", "name": "code_extension", "config": {"code": "export default () => {}"}}]
    )
    replies = iter(["[]", pi_answer, review, pi_answer, review, pi_answer, review])
    proposal = evolution.propose((), (), models, requests=[{"text": "add /chat"}], entries=(), adapter="pi")
    assert proposal.mutations == () and proposal.notes["failure"].startswith("the change does not deliver")


def test_a_replys_doubled_frontmatter_and_over_escaped_hook_quotes_are_put_right() -> None:
    reply = json.dumps(
        [
            {
                "id": "chat",
                "name": "agent_command",
                "config": {"text": "---\ndescription: Enter\n---\n---\ndescription: Enter\n---\nEntering chat mode."},
            },
            {
                "id": "kept",
                "name": "agent_command",
                "config": {"text": "---\ndescription: One\n---\n---\ndescription: Other\n---\nbody"},
            },
            {
                "id": "hooks",
                "name": "config",
                "config": {
                    "data": {
                        "hooks": {
                            "UserPromptExpansion": [
                                {
                                    "matcher": "^chat$",
                                    "hooks": [
                                        {"type": "command", "command": 'touch \\"$CLAUDE_CONFIG_DIR/.chat-mode\\"'}
                                    ],
                                }
                            ]
                        }
                    }
                },
            },
        ]
    )
    parsed = evolution._parse_proposal(reply, kinds=("agent_command", "config"))
    assert parsed is not None
    by_id = {entry_id: config for entry_id, _, config in parsed}
    assert by_id["chat"]["text"] == "---\ndescription: Enter\n---\nEntering chat mode."
    assert by_id["kept"]["text"].count("---\n") == 4  # two different blocks are the model's to keep
    command = by_id["hooks"]["data"]["hooks"]["UserPromptExpansion"][0]["hooks"][0]["command"]
    assert command == 'touch "$CLAUDE_CONFIG_DIR/.chat-mode"'


def test_a_reply_with_the_design_beside_the_array_and_a_stray_brace_still_parses() -> None:
    entry = {"id": "hooks", "name": "config", "config": {"data": {"hooks": {"Stop": [{"hooks": []}]}}}}
    reply = 'Here it is.\n{"design": "the plan. How to use: /x"} ' + json.dumps([entry])[:-1] + "}]\nDone."
    items = evolution._items_in(reply)
    assert [next(iter(item)) for item in items] == ["design", "id"]
    assert evolution._parse_design(reply) == "the plan. How to use: /x"
    parsed = evolution._parse_proposal(reply, kinds=("config",))
    assert parsed is not None and parsed[0][0] == "hooks" and parsed[0][2]["data"] == entry["config"]["data"]
    # Prose brackets around a value are skipped, and a reply without JSON gives nothing.
    assert evolution._items_in("see [1] and {2} then " + json.dumps({"design": "d"})) == [{"design": "d"}]
    assert evolution._items_in("nothing here [") == []


def test_a_numeric_setting_whose_name_ends_in_tokens_is_no_credential() -> None:
    """Claude Code's CLAUDE_CODE_MAX_OUTPUT_TOKENS and CLAUDE_CODE_MAX_CONTEXT_TOKENS hold limits, and a settings
    seed built from a person's ~/.claude carries them; the secret screen still refuses a token-named key with text."""
    from reef.harness.tree.nodes import config_node

    config_node(
        None,
        {"data": {"env": {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1000000"}}},
    )
    with pytest.raises(ValueError, match="inline credential"):
        config_node(None, {"data": {"env": {"GITHUB_TOKEN": "ghp_not_a_number"}}})


def test_the_service_takes_a_request_body_larger_than_aiohttps_default(tmp_path) -> None:
    """A Claude Code session with many MCP servers and skills posts several MiB on its first call; aiohttp's 1 MiB
    default answered 413, which Claude Code reported as a request too large."""
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer
    from reef_service.test_harness_requests import _growing_recipe, _seeded

    from reef.service.app import MAX_REQUEST_BYTES, create_app

    assert MAX_REQUEST_BYTES >= 16 * 1024 * 1024
    dispatcher = _seeded(tmp_path, _growing_recipe(tmp_path, lambda nodes, samples, models: None))

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            big = {"text": "x" * (3 * 1024 * 1024), "session": "s", "release_id": "r"}
            response = await client.post("/reef/train", json=big, headers={"x-reef-scenario": "agents"})
            # Whatever the route says about the body, it is not the size that stops it.
            assert response.status != 413, await response.text()
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
