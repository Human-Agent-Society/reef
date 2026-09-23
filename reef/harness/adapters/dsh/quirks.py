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
dsh has. A command is always ``disable-model-invocation: true`` with no
``user-invocable`` key: frontmatter the node text carries keeps its other
keys and gets the missing ``name`` and ``description``, and frontmatter that
does not parse is refused at render.

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
from typing import Any

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


_Dumper.add_representer(_Js, lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:js", str(value)))


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
    if text.startswith("---\n") and not user_only:
        return text
    header: dict[str, Any] = {}
    body = text
    if text.startswith("---\n"):
        # A command's own frontmatter is read the way dsh reads it, up to the first line that is exactly ---,
        # and written again with the command's invocation.
        match = re.match(r"---\n(.*?)^---$\n?", text, re.DOTALL | re.MULTILINE)
        if match is None:
            raise RenderError(f"dsh command {path} opens its frontmatter with --- but never closes it")
        try:
            own = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise RenderError(f"dsh command {path} has frontmatter that is not valid YAML: {exc}") from exc
        if not isinstance(own, dict):
            raise RenderError(f"dsh command {path} has frontmatter that is not a YAML mapping")
        header, body = own, text[match.end() :]
    name = path.split("/")[-2]
    first = next((line.strip().lstrip("#").strip() for line in body.splitlines() if line.strip()), "")
    header = {"name": name, "description": first[:200] or name, **header}
    if user_only:
        # Only the person types a command: never the model, and never hidden from the / menu.
        header["disable-model-invocation"] = True
        header.pop("user-invocable", None)
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + body


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
