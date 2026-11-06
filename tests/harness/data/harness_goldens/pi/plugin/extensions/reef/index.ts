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
    return { systemPrompt: `${event.systemPrompt}\n\n${rules}` };
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
