// The proposer agent's own tools: its workspace through Reef's admission, and the candidate harness run for real.
// Both go to the gateway at REEF_PROPOSER_URL, which holds every credential; this file holds none.
import { Type } from "typebox";

async function gateway(route, body, signal) {
  const base = process.env.REEF_PROPOSER_URL;
  if (!base) throw new Error("REEF_PROPOSER_URL is not set: these tools run only inside a Reef proposer run");
  const response = await fetch(`${base}${route}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`${route} answered ${response.status}: ${text.slice(0, 2000)}`);
  return { content: [{ type: "text", text }], details: {} };
}

export default function (pi) {
  if (!process.env.REEF_PROPOSER_URL) return;

  pi.registerTool({
    name: "harness_check",
    label: "Check the harness",
    description:
      "Read workspace/harness back into entries and run Reef's admission on them, exactly as the evolve step will: " +
      "the mutations it would apply, or why it refuses them (a bad name, a reserved id, a credential-shaped text, " +
      "an extension that does not load). Takes no arguments.",
    promptSnippet: "Check workspace/harness against Reef's admission",
    promptGuidelines: ["Run harness_check after every change to workspace/harness and fix what it refuses."],
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, signal) {
      return gateway("/check", {}, signal);
    },
  });

  pi.registerTool({
    name: "harness_trial",
    label: "Try the harness",
    description:
      "Run the candidate harness in workspace/harness for real: a fresh pi session with the changed tree, online, " +
      "given `task` as its prompt. Answers with the session's final text, the tools it called, every image, " +
      "speech, embedding or decision call it made (status and the provider's error when one failed) and the end of " +
      "its stderr. Nothing the trial does reaches the user.",
    promptSnippet: "Run the changed harness on a task and see what it really does",
    promptGuidelines: [
      "Use harness_trial to prove a change works before you finish; a change never tried is not done.",
      "When harness_trial shows a failed provider call, read its error and fix the model, parameters or format.",
    ],
    parameters: Type.Object({
      task: Type.String({ description: "the prompt the trial session gets, as a user would type it" }),
    }),
    async execute(_toolCallId, params, signal) {
      return gateway("/trial", { task: params.task }, signal);
    },
  });
}
