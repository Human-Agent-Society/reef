"""pi adapter quirks: what the declarative descriptor cannot state.

pi is the friendlier of the two bundled harnesses - one environment variable
relocates its whole configuration and it never mutates rendered files at boot -
so the quirks reduce to the files it may create beside the rendered files:
``trust.json`` (trust decisions) and ``auth.json`` (provider auth caches) can
appear in the agent directory even for an offline run.

pi follows the Agent Skills spec: a ``SKILL.md`` needs ``name`` and
``description`` frontmatter or the skill is reported as a conflict at startup
and never offered. ``process_skill`` gives a skill node whose
text carries none both, the description being the text's first line.

pi 0.84.2 rewrites a few old settings keys when it loads ``settings.json`` and
writes the new form back on its next save: ``queueMode`` becomes
``steeringMode``, a boolean ``websockets`` becomes ``transport``, and a
``skills`` object becomes the array of its ``customDirectories`` (its
``enableSkillCommands`` moves to the top level). The render writes the new
form itself, so the file an install writes is the one pi keeps, and the check
of the keys pi may not change never refuses pi's own rewrite.
"""

from __future__ import annotations

from typing import Any

from reef.harness.adapters.common.frontmatter import with_skill_frontmatter
from reef.harness.adapters.descriptor import AdapterRenderer

cleanup_whitelist = (
    "pi-agent/trust.json",
    "pi-agent/auth.json",
)

SETTINGS_PATH = "pi-agent/settings.json"


def migrated_settings(settings: dict[str, object]) -> dict[str, object]:
    """``settings`` with the old keys pi 0.84.2 rewrites on load in their new form, as its migrateSettings does."""
    if "queueMode" in settings and "steeringMode" not in settings:
        settings["steeringMode"] = settings.pop("queueMode")
    if "transport" not in settings and isinstance(settings.get("websockets"), bool):
        settings["transport"] = "websocket" if settings.pop("websockets") else "sse"
    skills = settings.get("skills")
    if isinstance(skills, dict):
        if "enableSkillCommands" in skills and "enableSkillCommands" not in settings:
            settings["enableSkillCommands"] = skills["enableSkillCommands"]
        directories = skills.get("customDirectories")
        if isinstance(directories, list) and directories:
            settings["skills"] = directories
        else:
            settings.pop("skills")
    return settings


#: The extension a pi layer loads: pi reads models.json only from its own directory, so this registers the
#: providers of the layer's, with the defaults pi gives a model there.
PROVIDER_EXTENSION = """\
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export default function (pi: any) {
  const path = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "models.json");
  const { providers } = JSON.parse(readFileSync(path, "utf8"));
  for (const [name, provider] of Object.entries<any>(providers)) {
    const models = (provider.models ?? []).map((model: any) => ({
      name: model.id,
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 16384,
      ...model,
    }));
    pi.registerProvider(name, { ...provider, models });
  }
}
"""


class PiAdapterRenderer(AdapterRenderer):
    @staticmethod
    def layer_files(files: dict[str, str]) -> dict[str, str]:
        # --append-system-prompt reads a file only when it exists, so the rules file is always there. The extension
        # sits where no part renders, in a directory -e loads, so the layer always holds one and pi loads it as a
        # package rather than as one extension.
        return {"pi-agent/AGENTS.md": "", **files, "pi-agent/extensions/reef/index.ts": PROVIDER_EXTENSION}

    @staticmethod
    def process_config(path: str, config: dict[str, Any]) -> dict[str, Any]:
        # Write pi's settings in the form pi keeps.
        return migrated_settings(config) if path == SETTINGS_PATH else config

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        # Give every skill the frontmatter pi requires when its text has none.
        return with_skill_frontmatter(path, text)
