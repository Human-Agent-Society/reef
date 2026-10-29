import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// The package's rules.md, beside this file's extensions/ directory.
const RULES = join(dirname(fileURLToPath(import.meta.url)), "..", "rules.md");

export default function reefQuickstartSystem(pi) {
  pi.on("before_agent_start", async () => {
    const rules = readFileSync(RULES, "utf8").replace(/\n+$/, "");
    return { systemPrompt: rules };
  });
  pi.on("before_provider_request", async ({ payload }) => {
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.messages)) return payload;
    const messages = payload.messages.map((message) => {
      const content = message?.content;
      if (Array.isArray(content) && content.length === 1 && content[0]?.type === "text") {
        return { ...message, content: content[0].text };
      }
      return message;
    });
    return { ...payload, messages };
  });
}
