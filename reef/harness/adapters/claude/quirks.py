"""Claude Code adapter quirks: boot artifacts and enforced config invariants.

Claude Code's boot writes local state beside the rendered config under
CLAUDE_CONFIG_DIR: a top-level ``.claude.json`` state file, a ``statsig/``
feature-gate cache, per-session ``todos/`` lists, and ``shell-snapshots/``.
The descriptor whitelists those so the episode inverse tolerates them and
reports anything else as residue.

``finalize_render`` enforces the traps a mutated ``settings.json`` could
reopen. The descriptor keeps a benchmark episode hermetic through
environment variables (auto-update, telemetry, and non-essential traffic all
off); a composition that sets ``settings.env`` to turn any of them back on,
or that flips ``includeCoAuthoredBy`` on, is rejected at render — the same
gate that rejects an invalid node.

It also keeps every model call on Reef's model binding. The binding writes
``ANTHROPIC_BASE_URL``, ``ANTHROPIC_AUTH_TOKEN`` and ``ANTHROPIC_MODEL`` into
``settings.env``, renders after the tree and wins every name it writes. Its
token is a credential, which a tree cannot hold because admission refuses an
inline credential, so the three names pass only beside the binding's token.
Any other env name Claude Code reads to choose the endpoint (another
provider such as Bedrock or Vertex, a proxy), the credential or the model,
a settings key that chooses the model, a credential helper or a login
method, and a ``model`` in the frontmatter of a command or a skill are the
tree choosing where calls go, and are refused.
"""

from __future__ import annotations

import json
import re

import yaml

from reef.harness.tree.render import RenderError

_CONFIG_PATH = "claude/settings.json"

# The env switches the descriptor relies on to keep episodes hermetic. A
# rendered settings.env that sets any of these to a falsey value would undo
# the descriptor's own env and let the binary phone home mid-campaign.
_HERMETIC_ENV = ("DISABLE_AUTOUPDATER", "DISABLE_TELEMETRY", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")
_FALSEY = {"0", "false", "off", "no", ""}

#: The env names Reef's model binding writes, and among them its credential.
BINDING_ENV = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL")
BINDING_CREDENTIAL = "ANTHROPIC_AUTH_TOKEN"
#: The env names Claude Code 2.1.257 reads to choose the endpoint, the provider, the credential or the model of a
#: model call: every ANTHROPIC_ name, the cloud providers and their credentials, the provider switches, proxies,
#: endpoints, sockets, credential helpers and files, and model names. Matched without case, as Windows reads env.
MODEL_ROUTE_ENV = re.compile(
    r"^(?:ANTHROPIC_|AWS_|AZURE_|GOOGLE_|GCLOUD_|GCE_|CLOUDSDK_|CLOUD_ML_|VERTEX_)"
    r"|^CLAUDE_CODE_(?:USE_(?:ANTHROPIC_|BEDROCK|FOUNDRY|GATEWAY|MANTLE|VERTEX)|SKIP_\w*_AUTH$|PROVIDER_)"
    r"|_MODEL$|_MODEL_(?:CATALOG|OPTION|FORCE)"
    r"|PROXY|BASE_URL|ENDPOINT|SOCKET|API_KEY|AUTH_|CREDS|CREDENTIAL|HELPER|FILE_DESCRIPTOR",
    re.IGNORECASE,
)
#: settings.json keys that choose the model, a credential helper or a login method.
MODEL_ROUTE_SETTINGS = (
    "advisorModel",
    "apiKeyHelper",
    "availableModels",
    "awsAuthRefresh",
    "awsCredentialExport",
    "enforceAvailableModels",
    "fallbackModel",
    "forceLoginGatewayUrl",
    "forceLoginMethod",
    "gcpAuthRefresh",
    "model",
    "modelOverrides",
    "modelPicker",
    "proxyAuthHelper",
    "switchModelsOnFlag",
)
#: Where commands and skills render; Claude Code reads a ``model`` in their frontmatter.
MARKDOWN_ROOTS = ("claude/commands/", "claude/skills/")
#: Claude Code's frontmatter block: after a byte order mark, an opening --- line, then the text up to the next ---.
FRONTMATTER_BLOCK = re.compile(r"---\s*\n([\s\S]*?)---\s*\n?")
#: A plain ``key: value`` line and the value characters Claude Code quotes when its YAML reader refuses the block.
PLAIN_LINE = re.compile(r"^([a-zA-Z_-]+):\s+(.+)$")
YAML_SYNTAX = re.compile(r"[{}[\]*&#!|>%@`]|: ")

cleanup_whitelist = (
    "claude/.claude.json",
    "claude/statsig",
    "claude/todos",
    "claude/shell-snapshots",
)


def quoted_values(block: str) -> str:
    """``block`` as Claude Code reads it again when its YAML reader refuses it: plain values holding YAML quoted."""
    lines = []
    for line in block.split("\n"):
        match = PLAIN_LINE.match(line)
        if match is not None:
            value = match.group(2)
            quoted = value[0] in "\"'" and value[-1] == value[0]
            if not quoted and YAML_SYNTAX.search(value):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                line = f'{match.group(1)}: "{escaped}"'
        lines.append(line)
    return re.sub(r"(?m)^\t+", lambda tabs: "  " * len(tabs.group(0)), "\n".join(lines))


def top_level_keys(block: str) -> set[str] | None:
    """The top-level keys of a YAML block, merged keys included; None when PyYAML cannot read it."""
    loader = yaml.SafeLoader(block)
    try:
        node = loader.get_single_node()
        if not isinstance(node, yaml.MappingNode):
            return set()
        loader.flatten_mapping(node)
    except (yaml.YAMLError, RecursionError):
        return None
    finally:
        loader.dispose()
    return {key.value for key, _ in node.value if isinstance(key, yaml.ScalarNode)}


def frontmatter_chooses_model(text: str) -> bool:
    """Whether Claude Code can read a ``model`` key in the frontmatter of a command or a skill.

    A block PyYAML cannot read as Claude Code does counts when it holds the word ``model`` or an escape that could
    spell it, since Bun's YAML reader may still read it.
    """
    match = FRONTMATTER_BLOCK.match(text.removeprefix("\ufeff"))
    if match is None:
        return False
    block = match.group(1)
    for candidate in (block, quoted_values(block)):
        keys = top_level_keys(candidate)
        if keys is not None:
            return "model" in keys
    return "model" in block or "\\" in block


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    config = json.loads(files[_CONFIG_PATH])
    if config.get("includeCoAuthoredBy") is True:
        raise RenderError("claude composition must keep includeCoAuthoredBy false for benchmark episodes")
    env = config.get("env")
    if isinstance(env, dict):
        for key in _HERMETIC_ENV:
            if key in env and str(env[key]).strip().lower() in _FALSEY:
                raise RenderError(f"claude composition must not re-enable {key} for benchmark episodes")
        credential = env.get(BINDING_CREDENTIAL)
        bound = isinstance(credential, str) and bool(credential.strip())
        for name in env:
            if bound and name in BINDING_ENV:
                continue
            if MODEL_ROUTE_ENV.search(name):
                raise RenderError(
                    f"claude composition must not set env {name}: Reef's model binding chooses the endpoint, "
                    "the credential and the model"
                )
    for key in MODEL_ROUTE_SETTINGS:
        if key in config:
            raise RenderError(
                f"claude composition must not set {key}: Reef's model binding chooses the model and its credential"
            )
    for path, text in files.items():
        if path.startswith(MARKDOWN_ROOTS) and path.endswith(".md") and frontmatter_chooses_model(text):
            raise RenderError(
                f"claude {path} must not set model in its frontmatter: Reef's model binding chooses the model"
            )
    return files
