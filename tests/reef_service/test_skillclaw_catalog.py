"""SkillCatalogModule: the pool-serving skills layer and its catalog injection.

Ported from the sealed campaign's test_skill_catalog.py (branch #223, commit
0c81580e); the module now lives in the skillclaw method package and
reads the pool from the rendered pi tree's ``skills/`` directory. The
expected strings below are the OpenClaw ``system-prompt.ts`` catalog format
(``buildSkillsSection`` wrapping ``formatSkillsForPrompt`` or
``formatSkillsCompact``), byte-pinned: they are what the SkillClaw proxy
injects.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

from reef.core.errors import ReefError

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "skillclaw"

_ALPHA = "---\nname: alpha\ndescription: Count things\n---\n\nCount carefully.\n"
_BETA = "---\nname: beta\ndescription: Sort things\n---\n\nSort carefully.\n"

_POOL = {"skills/alpha/SKILL.md": _ALPHA, "skills/beta/SKILL.md": _BETA}

_FULL_SECTION = """## Skills (mandatory)
Before replying: scan <available_skills> <description> entries.
- If exactly one skill clearly applies: read its SKILL.md at <location> with `read`, then follow it.
- If multiple could apply: choose the most specific one, then read/follow it.
- If none clearly apply: do not read any SKILL.md.
Constraints: never read more than one skill up front; only read after selecting.
- When a skill drives external API writes, assume rate limits: prefer fewer larger writes, avoid tight one-item loops, serialize bursts when possible, and respect 429/Retry-After.
The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.

<available_skills>
  <skill>
    <name>alpha</name>
    <description>Count things</description>
    <location>/root/skills/alpha/SKILL.md</location>
  </skill>
  <skill>
    <name>beta</name>
    <description>Sort things</description>
    <location>/root/skills/beta/SKILL.md</location>
  </skill>
</available_skills>
"""


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    # Every example names its package ``harness``: drop any sibling
    # example's cached import so this file's syspath entry wins.
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.catalog")


@pytest.fixture
def module_cls(catalog: ModuleType) -> type:
    return catalog.SkillCatalogModule


def _module(module_cls: type, **overrides):
    config = {"public_root": "/root/skills"}
    config.update(overrides)
    return module_cls(**config)


def test_served_text_is_the_openclaw_skills_section(module_cls) -> None:
    assert _module(module_cls).served_text(_POOL) == _FULL_SECTION


def test_catalog_falls_back_to_the_compact_form_beyond_max_chars(module_cls) -> None:
    text = _module(module_cls, max_chars=200).served_text(_POOL)
    assert text is not None
    assert "<location>/root/skills/alpha/SKILL.md</location>" in text
    assert "<description>Count things</description>" not in text
    assert "matches its name" in text


def test_catalog_lists_only_skills_with_frontmatter_name_and_description(module_cls) -> None:
    text = _module(module_cls).served_text(
        {
            **_POOL,
            "skills/bare/SKILL.md": "No frontmatter at all.",
            "skills/nameless/SKILL.md": "---\ndescription: Only a description\n---\n\nBody.\n",
            "skills/silent/SKILL.md": (
                "---\nname: silent\ndescription: Hidden\ndisable-model-invocation: true\n---\n\nBody.\n"
            ),
        }
    )
    assert text == _FULL_SECTION


def test_catalog_ignores_the_layers_non_skill_files(module_cls) -> None:
    """The rendered pi tree carries the adapter's own files beside skills/."""
    text = _module(module_cls).served_text({**_POOL, "settings.json": "{}", "models.json": "{}"})
    assert text == _FULL_SECTION


def test_catalog_escapes_xml_in_names_and_descriptions(module_cls) -> None:
    text = _module(module_cls).served_text(
        {"skills/amp/SKILL.md": '---\nname: a&b\ndescription: Uses <tags> & "quotes"\n---\n\nBody.\n'}
    )
    assert text is not None
    assert "<name>a&amp;b</name>" in text
    assert "<description>Uses &lt;tags&gt; &amp; &quot;quotes&quot;</description>" in text


def test_prepare_request_appends_after_existing_system_content(module_cls) -> None:
    payload = {"messages": [{"role": "system", "content": "Base."}, {"role": "user", "content": "q"}]}
    prepared = _module(module_cls).prepare_request(_POOL, "/v1/chat/completions", payload)
    content = prepared["messages"][0]["content"]
    assert content.startswith("Base.\n\n## Skills (mandatory)")
    assert prepared["messages"][1] == {"role": "user", "content": "q"}
    assert payload["messages"][0]["content"] == "Base."


def test_prepare_request_flattens_a_parts_list_system_message(module_cls) -> None:
    payload = {"messages": [{"role": "system", "content": [{"type": "text", "text": "Base."}]}]}
    prepared = _module(module_cls).prepare_request(_POOL, "/v1/chat/completions", payload)
    assert prepared["messages"][0]["content"].startswith("Base.\n\n## Skills (mandatory)")


def test_prepare_request_inserts_a_system_message_when_none_exists(module_cls) -> None:
    payload = {"messages": [{"role": "user", "content": "q"}]}
    prepared = _module(module_cls).prepare_request(_POOL, "/v1/chat/completions", payload)
    assert prepared["messages"][0] == {"role": "system", "content": _FULL_SECTION}


def test_prepare_request_appends_to_the_anthropic_system_shapes(module_cls) -> None:
    module = _module(module_cls)
    text = module.served_text(_POOL)
    assert module.prepare_request(_POOL, "/v1/messages", {"messages": []})["system"] == text
    assert module.prepare_request(_POOL, "/v1/messages", {"system": "Base.", "messages": []})["system"] == (
        f"Base.\n\n{text}"
    )
    blocks = module.prepare_request(
        _POOL, "/v1/messages", {"system": [{"type": "text", "text": "Base."}], "messages": []}
    )["system"]
    assert blocks == [{"type": "text", "text": "Base."}, {"type": "text", "text": text}]


def test_prepare_request_passes_an_empty_pool_through(module_cls) -> None:
    payload = {"messages": [{"role": "user", "content": "q"}]}
    assert _module(module_cls).prepare_request({}, "/v1/chat/completions", payload) is payload


def test_catalog_names_recovers_the_listed_skills_from_a_recorded_request(module_cls) -> None:
    prepared = _module(module_cls).prepare_request(
        _POOL, "/v1/chat/completions", {"messages": [{"role": "user", "content": "q"}]}
    )
    assert module_cls.catalog_names(prepared) == ("alpha", "beta")
    assert module_cls.catalog_names({"messages": []}) == ()


def test_catalog_names_unescapes_xml(module_cls) -> None:
    prepared = _module(module_cls).prepare_request(
        {"skills/amp/SKILL.md": "---\nname: a&b\ndescription: d\n---\n\nBody.\n"},
        "/v1/chat/completions",
        {"messages": []},
    )
    assert module_cls.catalog_names(prepared) == ("a&b",)


def test_validate_requires_skill_directories_with_a_skill_md(module_cls) -> None:
    module = _module(module_cls)
    module.validate(_POOL)
    module.validate({**_POOL, "skills/alpha/notes.txt": "aux data", "settings.json": "{}"})
    with pytest.raises(ReefError, match="is not under a skill directory"):
        module.validate({"skills/SKILL.md": "top level"})
    with pytest.raises(ReefError, match=r"must contain SKILL\.md"):
        module.validate({"skills/alpha/notes.txt": "aux data"})


def test_constructor_rejects_a_blank_public_root_and_a_non_positive_budget(module_cls) -> None:
    with pytest.raises(ValueError, match="public_root"):
        module_cls(public_root="  ")
    with pytest.raises(ValueError, match="max_chars"):
        module_cls(public_root="/root/skills", max_chars=0)
