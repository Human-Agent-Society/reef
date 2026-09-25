"""dsh adapter quirks: the patch layers, the credential file, skill frontmatter, and the boot scaffold.

dsh composes its plugin tree from bundle layers plus one user patch layer
per profile, ``profiles/<profile>/cordis.patch.yml``: a YAML list of entries
addressed by plugin id. Config nodes write each layer as a JSON object keyed
by id, so two nodes touching one plugin deep merge, and ``finalize_render``
emits the list for the headless profile an episode runs and the web profile
``reef-dsh web`` boots: one entry per id (a string starting with ``!!js ``
becomes a js expression, the form dsh's own bundles use), then one
``insert`` entry per rendered code extension so the loader boots it from its
path relative to the profile, the web profile loading the headless
profile's module. The ``env`` config target becomes ``.env``, the lowest
trust layer of dsh's launch environment, which is how the model binding's
key reaches its ``apiKeyEnv`` route. dsh ignores a SKILL.md without YAML
frontmatter, so a skill node whose text has none gets ``name`` and
``description`` synthesized, and an agent_command renders under the second
skill root as a user invocable skill (``/name``), the only command surface
dsh has. A command is always ``disable-model-invocation: true``, with no
``user-invocable`` key and none of the camelCase keys dsh refuses
(``userInvocable``, ``disableModelInvocation``, ``modelInvocable``).
Frontmatter the command text carries is read the way dsh reads it: between
two ``---`` lines, either one allowed a trailing carriage return, as YAML
1.2, where ``Yes`` and ``1:30`` are strings and so is a value tagged ``!``
or ``!!str``. It keeps its other keys; a ``name`` or ``description`` dsh
would not accept (absent, empty, not a string, or a name that is not a
skill name) is written the way a missing one is, and an empty or null block
counts as a mapping with no keys; frontmatter that does not parse, nests
too deeply to read, holds any other tag, or is any other value than a
mapping is refused at render. Every header is written so that YAML 1.2
reads each value back with its type: a string that YAML 1.1 or 1.2 would
read as a number, a boolean or a null is quoted.

The traps a mutated patch could reopen, in either profile: the session log
must stay plain JSONL (the reader cannot parse zstd, and a compressed
profile refuses a sessions root that holds plain logs), and the session
telemetry and the LLM title call stay disabled; the web profile's manifest
keeps ``patchReload: startup``. A composition that flips any of them is
rejected at render, the same check that rejects an invalid node.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

import yaml

from reef.harness.tree.render import RenderError

#: Each profile's patch layer and the directory its inserts name an extension under, relative to the profile.
PROFILE_PATCHES = {
    "dsh/profiles/headless/cordis.patch.yml": "./extensions/",
    "dsh/profiles/web/cordis.patch.yml": "../headless/extensions/",
}
#: The web profile's manifest: with dsh's default for a new web profile, live patch reload, ``dsh web`` exits at start.
WEB_MANIFEST = "dsh/profiles/web/package.json"
_ENV = "dsh/.env"
_EXTENSIONS = "dsh/profiles/headless/extensions/"
_SKILLS = "dsh/skills/"
_COMMANDS = "dsh-agents/skills/"
_JS = "!!js "
#: dsh lists a skill only under a name of lowercase letters and digits, in words joined by single hyphens.
SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: The invocation keys a command's own frontmatter loses: a command is user invocable by leaving the first out, and
#: dsh ignores a skill that holds any of the camelCase ones.
COMMAND_DROPPED_KEYS = ("user-invocable", "userInvocable", "disableModelInvocation", "modelInvocable")
#: How the YAML 1.2 core schema of dsh's reader types a plain value: each tag, its pattern and the characters the
#: value can start with. PyYAML follows YAML 1.1 instead, where Yes, on and 1:30 are not strings and 09, 0o17 and
#: 1e3 are.
YAML_1_2_RESOLVERS = (
    ("tag:yaml.org,2002:null", r"^(?:~|[Nn]ull|NULL|)$", ["~", "n", "N", ""]),
    ("tag:yaml.org,2002:bool", r"^(?:[Tt]rue|TRUE|[Ff]alse|FALSE)$", list("tTfF")),
    ("tag:yaml.org,2002:int", r"^(?:0o[0-7]+|[-+]?[0-9]+|0x[0-9a-fA-F]+)$", list("-+0123456789")),
    (
        "tag:yaml.org,2002:float",
        r"^(?:[-+]?\.(?:inf|Inf|INF)|\.nan|\.NaN|\.NAN"
        r"|[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)[eE][-+]?[0-9]+|[-+]?(?:\.[0-9]+|[0-9]+\.[0-9]*))$",
        list("-+.0123456789"),
    ),
)

# dsh's boot scaffolds each profile beside the rendered patch: the empty root
# entry list, node_modules symlinks into the installation, and the module
# fallback links, plus a package manifest and the pnpm workspace file for a
# profile without a manifest (the headless one; the web one is rendered).
# Episode state, not residue.
cleanup_whitelist = (
    "dsh/profiles/headless/package.json",
    "dsh/profiles/headless/cordis.yml",
    "dsh/profiles/headless/pnpm-workspace.yaml",
    "dsh/profiles/headless/node_modules/**",
    "dsh/profiles/headless/.dsh-module-fallback/**",
    "dsh/profiles/web/cordis.yml",
    "dsh/profiles/web/node_modules/**",
    "dsh/profiles/web/.dsh-module-fallback/**",
    "dsh/profiles/node_modules/**",
)


class _Js(str):
    """A js expression scalar, dumped with the ``!!js`` tag."""


class _Dumper(yaml.SafeDumper):
    pass


class FrontmatterLoader(yaml.SafeLoader):
    """``yaml.safe_load`` as dsh reads frontmatter: YAML 1.2 types for plain values; no tag but ``!`` and ``!!str``.

    A scalar tagged ``!`` is a string, as it is to dsh. Any other tag is refused because dsh types it where PyYAML
    may type it another way or fail.
    """

    yaml_implicit_resolvers: ClassVar[dict[str, list[tuple[str, re.Pattern[str]]]]] = {}

    def compose_node(self, parent: yaml.Node | None, index: object) -> yaml.Node | None:
        event = self.peek_event()
        event_tag = event.tag if isinstance(event, (yaml.ScalarEvent, yaml.CollectionStartEvent)) else None
        if event_tag not in (None, "!", "tag:yaml.org,2002:str"):
            raise yaml.MarkedYAMLError(problem=f"a value has the tag {event_tag!r}", problem_mark=event.start_mark)
        if event_tag == "!" and isinstance(event, yaml.ScalarEvent):
            # PyYAML resolves this scalar as a plain one, so ! true would be a boolean and ! "true\n" would fail.
            event.tag = "tag:yaml.org,2002:str"
        return super().compose_node(parent, index)

    def construct_yaml_int(self, node: yaml.ScalarNode) -> int:
        # YAML 1.2 reads 0777 in base ten, where PyYAML reads it in base eight and fails on 09.
        value = self.construct_scalar(node)
        if value.startswith(("0o", "0x")):
            return int(value[2:], 8 if value[1] == "o" else 16)
        return int(value)


_Dumper.add_representer(_Js, lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:js", str(value)))
FrontmatterLoader.add_constructor("tag:yaml.org,2002:int", FrontmatterLoader.construct_yaml_int)
for resolver_tag, resolver_pattern, first_characters in YAML_1_2_RESOLVERS:
    FrontmatterLoader.add_implicit_resolver(resolver_tag, re.compile(resolver_pattern), first_characters)
    # The dumper keeps YAML 1.1's patterns and adds these, so it quotes a string either version reads as another type.
    _Dumper.add_implicit_resolver(resolver_tag, re.compile(resolver_pattern), first_characters)


def _tagged(value: Any) -> Any:
    if isinstance(value, str):
        return _Js(value[len(_JS) :]) if value.startswith(_JS) else value
    if isinstance(value, dict):
        return {key: _tagged(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tagged(item) for item in value]
    return value


def _patch(entries: dict[str, Any], extensions: list[str], directory: str) -> str:
    rows: list[dict[str, Any]] = [{"id": plugin, **_tagged(entry)} for plugin, entry in sorted(entries.items())]
    if extensions:
        rows.append({"insert": [{"id": f"extension-{name}", "name": f"{directory}{name}.mjs"} for name in extensions]})
    return yaml.dump(rows, Dumper=_Dumper, sort_keys=True, default_flow_style=False, allow_unicode=True)


def _with_frontmatter(path: str, text: str, user_only: bool) -> str:
    # dsh reads frontmatter from a first line that is ---, less one trailing carriage return, to the next such line.
    lines = text.split("\n")
    opened = len(lines) > 1 and lines[0].removesuffix("\r") == "---"
    if opened and not user_only:
        return text
    header: dict[str, Any] = {}
    body = text
    if opened:
        # A command's own frontmatter is written again with the command's invocation.
        close = next((index for index in range(1, len(lines)) if lines[index].removesuffix("\r") == "---"), 0)
        if not close:
            raise RenderError(f"dsh command {path} opens its frontmatter with --- but never closes it")
        try:
            own = yaml.load("\n".join(lines[1:close]), Loader=FrontmatterLoader)
        except yaml.YAMLError as exc:
            raise RenderError(f"dsh command {path} has frontmatter that is not valid YAML: {exc}") from exc
        except RecursionError as exc:
            raise RenderError(f"dsh command {path} has frontmatter nested too deeply to read") from exc
        except Exception as exc:
            # The model writes this text: any other failure to read it (an integer past Python's digit limit, say)
            # is a refusal of the proposal, never a crash of the step.
            raise RenderError(f"dsh command {path} has frontmatter Reef cannot read: {exc}") from exc
        if own is not None and not isinstance(own, dict):
            raise RenderError(f"dsh command {path} has frontmatter that is not a YAML mapping")
        header, body = own or {}, "\n".join(lines[close + 1 :])
    name = path.split("/")[-2]
    first = next((line.strip().lstrip("#").strip() for line in body.splitlines() if line.strip()), "")
    description = first[:200] or name
    header = {"name": name, "description": description, **header}
    # dsh ignores a skill whose name is not a skill name or whose description is empty or not a string, so either is
    # written the way a missing one is.
    if not isinstance(header["name"], str) or not SKILL_NAME.fullmatch(header["name"]):
        header["name"] = name
    if not isinstance(header["description"], str) or not header["description"]:
        header["description"] = description
    if user_only:
        # Only the person types a command: never the model, and never hidden from the / menu.
        header["disable-model-invocation"] = True
        for key in COMMAND_DROPPED_KEYS:
            header.pop(key, None)
    try:
        dumped = yaml.dump(header, Dumper=_Dumper, sort_keys=False, default_flow_style=False, allow_unicode=True)
    except (RecursionError, ValueError, yaml.YAMLError) as exc:
        # A header that loaded can still nest too deeply to write again (a duplicate key keeps its first place but
        # takes its last, deeper value).
        raise RenderError(f"dsh command {path} has frontmatter Reef cannot write again: {exc}") from exc
    return "---\n" + dumped + "---\n" + body


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    extensions = sorted(
        path[len(_EXTENSIONS) : -len(".mjs")]
        for path in files
        if path.startswith(_EXTENSIONS) and path.endswith(".mjs") and "/" not in path[len(_EXTENSIONS) :]
    )
    for patch_path, directory in PROFILE_PATCHES.items():
        entries = json.loads(files[patch_path])
        for plugin, entry in entries.items():
            if not isinstance(entry, dict):
                raise RenderError(f"dsh patch entry {plugin!r} must be an object holding config, disabled, or inject")
        if entries.get("session-persistence-jsonl", {}).get("config", {}).get("compression") != "none":
            raise RenderError(
                f"dsh composition must keep the session log uncompressed (compression: none) in {patch_path}: "
                "Reef reads it, and the profiles share one sessions root"
            )
        for plugin in ("session-telemetry-otel", "session-title-llm"):
            if entries.get(plugin, {}).get("disabled") is not True:
                raise RenderError(f"dsh composition must keep {plugin} disabled in {patch_path}")
        files[patch_path] = _patch(entries, extensions, directory)
    if json.loads(files[WEB_MANIFEST]).get("dsh", {}).get("profile", {}).get("patchReload") != "startup":
        raise RenderError(f"dsh composition must keep dsh.profile.patchReload startup in {WEB_MANIFEST}")
    files[_ENV] = "".join(f"{key}={value}\n" for key, value in sorted(json.loads(files[_ENV]).items()))
    for path, text in list(files.items()):
        for root, user_only in ((_SKILLS, False), (_COMMANDS, True)):
            if path.startswith(root) and path.endswith("/SKILL.md"):
                files[path] = _with_frontmatter(path, text, user_only)
    return files
