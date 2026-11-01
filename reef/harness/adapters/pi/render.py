"""pi adapter render steps: skill frontmatter, Reef's extension in the package, and the layer's files.

pi follows the Agent Skills spec: a ``SKILL.md`` needs ``name`` and
``description`` frontmatter or the skill is reported as a conflict at startup
and never offered. ``process_skill`` gives a skill node whose
text carries none both, the description being the text's first line.

A release is a pi package in ``plugin/``. A package carries skills, prompts and
extensions, but no rules file and no settings, so ``finalize_render`` adds
Reef's extension, which adds ``plugin/rules.md`` to the system prompt and
applies ``plugin/settings.json``. The package's manifest loads that extension
before the tree's: pi runs ``before_agent_start`` handlers in load order, so a
tree's extension that replaces the system prompt replaces the rules too, as it
replaces pi's own context files. pi reads ``models.json`` only from its own
directory, so ``layer_files`` adds an extension that registers the layer's
provider on a person's machine.
"""

from __future__ import annotations

from reef.harness.adapters.common.frontmatter import with_skill_frontmatter
from reef.harness.adapters.descriptor import AdapterRenderer

#: Reef's extension in every package: the release's rules after pi's system prompt, unless the prompt already
#: holds them, and the release's defaultTools, which turns off pi's built-in tools and keeps the extensions' own.
REEF_EXTENSION = """\
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const plugin = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const read = (name: string) => (existsSync(join(plugin, name)) ? readFileSync(join(plugin, name), "utf8") : "");

export default function (pi: any) {
  pi.on("before_agent_start", async (event: any) => {
    const rules = read("rules.md").trim();
    // Another extension may have put the rules in already, in either order.
    if (!rules || event.systemPrompt.includes(rules)) return undefined;
    return { systemPrompt: `${event.systemPrompt}\\n\\n${rules}` };
  });
  pi.on("session_start", async () => {
    const settings = JSON.parse(read("settings.json") || "{}");
    if (!Array.isArray(settings.defaultTools)) return;
    const builtin = new Set(
      pi.getAllTools().filter((tool: any) => tool.sourceInfo?.source === "builtin").map((tool: any) => tool.name),
    );
    pi.setActiveTools(pi.getActiveTools().filter((name: string) => !builtin.has(name)));
  });
}
"""

#: The extension a layer adds: pi reads models.json only from its own directory, so this registers the
#: providers of the layer's, with the defaults pi gives a model there.
PROVIDER_EXTENSION = """\
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export default function (pi: any) {
  const path = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "pi-agent", "models.json");
  if (!existsSync(path)) return;
  const { providers } = JSON.parse(readFileSync(path, "utf8"));
  for (const [name, provider] of Object.entries<any>(providers ?? {})) {
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


#: The package's manifest. Without one, pi loads a package's extensions in directory listing order, which differs
#: between file systems. pi loads a path once, so the extensions directory adds the tree's after Reef's.
PACKAGE_MANIFEST = """\
{
  "name": "reef-harness",
  "private": true,
  "pi": {
    "extensions": ["extensions/reef/index.ts", "extensions"],
    "skills": ["skills"],
    "prompts": ["prompts"]
  }
}
"""


class PiAdapterRenderer(AdapterRenderer):
    @staticmethod
    def finalize_render(files: dict[str, str]) -> dict[str, str]:
        # A directory, not a file, so no extension part of the tree takes its path.
        return {
            **files,
            "plugin/extensions/reef/index.ts": REEF_EXTENSION,
            "plugin/package.json": PACKAGE_MANIFEST,
        }

    @staticmethod
    def layer_files(files: dict[str, str]) -> dict[str, str]:
        return {**files, "plugin/extensions/reef-provider/index.ts": PROVIDER_EXTENSION}

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        # Give every skill the frontmatter pi requires when its text has none.
        return with_skill_frontmatter(path, text)
