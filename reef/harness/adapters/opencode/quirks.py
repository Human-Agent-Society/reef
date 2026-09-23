"""opencode adapter quirks: boot mutations and enforced config invariants.

opencode's boot mutates its own composition directories: it injects a
``$schema`` key into config files that lack one (an in-place edit of a
rendered file, invisible to the residue scan), writes a ``.gitignore``, and
npm-installs ``@opencode-ai/plugin`` with a ``node_modules`` tree and
lockfiles. The whitelist below names exactly those artifacts so the episode
inverse tolerates them and nothing else.

``finalize_render`` enforces the traps a mutated config node could reopen: a
benchmark episode must never autoupdate the binary mid-campaign or upload a
share link, so a composition that overrides either is rejected at render -
the same gate that rejects an invalid node.

It also keeps the model Reef's. The model binding Reef appends at render is
the only writer of ``provider`` and ``model``: it renders after the tree and
wins every key it writes, so a tree may hold those two keys only in the
binding's shape, and a provider, a model choice (``small_model``, an agent's
or a command's ``model``) or a provider list anywhere else is a composition
pointing opencode at another endpoint. A command's ``agent`` must name an
agent the tree defines or one opencode builds in, since opencode fails the
command at run time with an opaque server error otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import yaml

from reef.harness.tree.render import RenderError

_CONFIG_PATH = "opencode/opencode.json"
COMMAND_DIR = "opencode/command/"

#: What the model binding writes: one ``reef`` provider holding these keys, its options, and empty model entries.
BINDING_PROVIDER = "reef"
BINDING_PROVIDER_KEYS = frozenset({"models", "npm", "options"})
BINDING_OPTION_KEYS = frozenset({"apiKey", "baseURL"})
#: Top-level keys the binding never writes that choose which model or which providers a run uses.
MODEL_CHOICE_KEYS = ("small_model", "enabled_providers", "disabled_providers")
#: The agents opencode documents as built in. Its hidden title, summary and compaction agents also run,
#: but they are the prompts of opencode's own calls, not modes a command picks.
BUILTIN_AGENTS = frozenset({"build", "plan", "general", "explore"})

cleanup_whitelist = (
    "opencode/.gitignore",
    "opencode/package.json",
    "opencode/package-lock.json",
    "opencode/bun.lock",
    "opencode/node_modules/**",
)


def check_binding_shape(config: Mapping[str, object]) -> None:
    """``provider`` and ``model`` appear together in the binding's shape, or not at all."""
    provider = config.get("provider")
    model = config.get("model")
    if provider is None and model is None:
        return
    if provider is None or model is None:
        key = "model" if provider is None else "provider"
        raise RenderError(f"opencode composition must not set {key}: Reef's model binding writes provider and model")
    refusal = (
        f"opencode composition must not set provider: Reef's model binding writes the only one, {BINDING_PROVIDER!r}"
    )
    if not isinstance(provider, dict) or set(provider) != {BINDING_PROVIDER}:
        raise RenderError(refusal)
    reef = provider[BINDING_PROVIDER]
    if not isinstance(reef, dict) or set(reef) != BINDING_PROVIDER_KEYS:
        raise RenderError(refusal)
    options, models = reef["options"], reef["models"]
    if not isinstance(options, dict) or set(options) != BINDING_OPTION_KEYS or not isinstance(models, dict):
        raise RenderError(refusal)
    if any(entry != {} for entry in models.values()):
        raise RenderError(refusal)
    if not isinstance(model, str) or model.removeprefix(f"{BINDING_PROVIDER}/") not in models:
        raise RenderError("opencode composition must not set model: Reef's model binding chooses it")


def check_command(where: str, command: Mapping[str, object], agents: frozenset[str]) -> None:
    """A command chooses no model, and its agent is one the run has."""
    if "model" in command:
        raise RenderError(f"opencode {where} must not choose a model: Reef's model binding chooses it")
    agent = command.get("agent")
    if agent is not None and agent not in agents:
        known = ", ".join(sorted(agents))
        raise RenderError(f"opencode {where} names agent {agent!r}, which the tree does not define (agents: {known})")


def command_frontmatter(text: str) -> Mapping[str, object]:
    """A command file's YAML frontmatter, empty when it has none; opencode skips a file whose frontmatter does not parse."""
    lines = text.split("\n")
    if lines[0].rstrip() != "---":
        return {}
    closing = next((index for index, line in enumerate(lines[1:], start=1) if line.startswith("---")), None)
    if closing is None:
        return {}
    try:
        data = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    config = json.loads(files[_CONFIG_PATH])
    if config.get("autoupdate") is not False:
        raise RenderError("opencode composition must keep autoupdate false for benchmark episodes")
    if config.get("share") != "disabled":
        raise RenderError("opencode composition must keep share disabled for benchmark episodes")
    check_binding_shape(config)
    for key in MODEL_CHOICE_KEYS:
        if key in config:
            raise RenderError(f"opencode composition must not set {key}: Reef's model binding chooses the model")
    agents = set(BUILTIN_AGENTS)
    # ``mode`` is opencode's deprecated name for ``agent``; both define agents.
    for section in ("agent", "mode"):
        entries = config.get(section)
        for name, agent in entries.items() if isinstance(entries, dict) else ():
            if not isinstance(agent, dict):
                continue
            if "model" in agent:
                raise RenderError(
                    f"opencode {section} {name!r} must not choose a model: Reef's model binding chooses it"
                )
            if agent.get("disable") is True:
                agents.discard(name)
            else:
                agents.add(name)
    defined = frozenset(agents)
    commands = config.get("command")
    for name, command in commands.items() if isinstance(commands, dict) else ():
        if isinstance(command, dict):
            check_command(f"command {name!r} in opencode.json", command, defined)
    for path, text in files.items():
        if path.startswith(COMMAND_DIR) and path.endswith(".md"):
            check_command(f"command {path[len(COMMAND_DIR) : -len('.md')]!r}", command_frontmatter(text), defined)
    return files
