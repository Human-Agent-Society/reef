"""pi adapter render steps: skill frontmatter, and the layer's files and arguments.

pi follows the Agent Skills spec: a ``SKILL.md`` needs ``name`` and
``description`` frontmatter or the skill is reported as a conflict at startup
and never offered. ``process_skill`` gives a skill node whose
text carries none both, the description being the text's first line.

On a person's machine, pi reads ``settings.json`` and ``models.json`` only
from the person's own directory. ``layer_files`` adds an extension that
registers the release's providers, and ``layer_args`` passes the settings the
adapter's checks admit as pi's flags: the thinking level as ``--thinking``,
and an empty ``defaultTools`` as ``--no-builtin-tools``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from reef.harness.adapters.common.frontmatter import with_skill_frontmatter
from reef.harness.adapters.descriptor import AdapterRenderer

SETTINGS_PATH = "pi-agent/settings.json"


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
    def layer_args(files: Mapping[str, str]) -> tuple[tuple[str, ...], ...]:
        settings = json.loads(files[SETTINGS_PATH])
        level = settings.get("defaultThinkingLevel")
        thinking = () if level is None else (("--thinking", str(level)),)
        return (*thinking, ("--no-builtin-tools",)) if settings.get("defaultTools") == [] else thinking

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        # Give every skill the frontmatter pi requires when its text has none.
        return with_skill_frontmatter(path, text)
