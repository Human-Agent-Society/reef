import { readFileSync } from "node:fs";
import { join } from "node:path";

export default function reefQuickstartSystem(pi) {
  pi.on("before_agent_start", async () => {
    const root = process.env.PI_CODING_AGENT_DIR;
    const rules = readFileSync(join(root, "AGENTS.md"), "utf8").replace(/\n+$/, "");
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
