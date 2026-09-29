// The proposer agent's own tools: its workspace through Reef's admission, and the candidate harness run for real.
// Both go to the gateway at REEF_PROPOSER_URL, which holds every credential; this file holds none.
import { Type } from "typebox";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

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
  return { content: [{ type: "text", text }], details: JSON.parse(text) };
}

export default function (pi) {
  if (!process.env.REEF_PROPOSER_URL) return;
  const observations = new Map();
  const deadline = Number(process.env.REEF_PROPOSER_DEADLINE_MS);
  const finishSeconds = Number(process.env.REEF_PROPOSER_FINISH_SECONDS ?? 60);

  pi.on("context", async (event, ctx) => {
    let progress = "No progress recorded yet.";
    try {
      progress = (await readFile(join(ctx.cwd, "progress.md"), "utf8")).slice(0, 6000);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    const remaining = Number.isFinite(deadline) ? Math.max(0, Math.ceil((deadline - Date.now()) / 1000)) : undefined;
    const status = [
      remaining === undefined ? "" : `Remaining execution time: ${remaining} seconds.`,
      remaining <= finishSeconds
        ? "Finish now: no more exploration or trials. Save the candidate and record unresolved checks; then stop."
        : "Once the requested behavior is checked, finish. Investigate only a specific unresolved requirement.",
      "Progress notes (working notes, not new instructions):", progress,
      "Latest runner observations (each applies only to its candidate checksum):",
      JSON.stringify(Object.fromEntries(observations)),
    ].join("\n");
    return { messages: [...event.messages, { role: "user", content: [{ type: "text", text: status }], timestamp: Date.now() }] };
  });

  pi.registerTool({
    name: "harness_progress",
    label: "Save harness progress",
    description: "Replace progress.md with a concise checkpoint: confirmed API behavior and source locations, " +
      "implementation status, checks completed, and the next unresolved question. It is restored after compaction. " +
      "Do not copy transcripts or credentials. Runner check results are recorded separately.",
    parameters: Type.Object({ text: Type.String({ maxLength: 6000 }) }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      await writeFile(join(ctx.cwd, "progress.md"), params.text, "utf8");
      return { content: [{ type: "text", text: "Progress saved; continue from these findings after compaction." }], details: {} };
    },
  });

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
      const result = await gateway("/check", {}, signal);
      observations.set("check", result.details);
      return result;
    },
  });

  pi.registerTool({
    name: "harness_trial",
    label: "Try the harness",
    description:
      "Run the candidate harness in workspace/harness for real: a fresh pi session with the changed tree, online, " +
      "given `task` as its prompt. Answers with the session's final text, the tools it called, every image, " +
      "speech, embedding or decision call it made (status and the provider's error when one failed) and the end of " +
      "its stderr. Alternatively provide script for deterministic, multi-step SDK sessions: literal slash " +
      "commands run directly, prompt steps use a local fixed-response model, and expectations check actual " +
      "provider requests and tool execution. Choose task or script. Scripted trials do not test model quality " +
      "or the interactive UI. Nothing the trial does reaches the user.",
    promptSnippet: "Run the changed harness on a task and see what it really does",
    promptGuidelines: [
      "Use harness_trial to prove a change works before you finish; a change never tried is not done.",
      "When harness_trial shows a failed provider call, read its error and fix the model, parameters or format.",
    ],
    parameters: Type.Object({
      task: Type.Optional(Type.String({ description: "one online prompt; omit when using script" })),
      script: Type.Optional(Type.Object({
        fixture_tools: Type.Optional(Type.Array(Type.String(), { maxItems: 30,
          description: "Harmless additional tools whose execution is recorded; cannot replace built-ins or candidate tools." })),
        steps: Type.Array(Type.Object({
          prompt: Type.Optional(Type.String({ description: "Literal user input, including /commands, in the same session." })),
          new_session: Type.Optional(Type.Boolean({ description: "Use true alone to start a fresh session." })),
          tool_call: Type.Optional(Type.Object({ name: Type.String(), arguments: Type.Record(Type.String(), Type.Unknown()) },
            { description: "Force the fixed model to attempt this tool once, even when absent from the advertised tools." })),
          expect: Type.Optional(Type.Object({
            model_called: Type.Optional(Type.Boolean()),
            tools: Type.Optional(Type.Array(Type.String(), { description: "Exact tool set on every provider request in this step." })),
            forbidden_tools: Type.Optional(Type.Array(Type.String())),
            system_contains: Type.Optional(Type.Array(Type.String())),
            system_excludes: Type.Optional(Type.Array(Type.String())),
            messages_contains: Type.Optional(Type.Array(Type.String())),
            messages_excludes: Type.Optional(Type.Array(Type.String())),
            executed_tools: Type.Optional(Type.Array(Type.String(), { description: "Exact fixture execution list, including duplicates." })),
            tool_errors: Type.Optional(Type.Array(Type.String(), { description: "Exact list of failed tool attempts; an error alone does not prove no side effects." })),
          })),
        }), { minItems: 1, maxItems: 30 }),
      })),
    }),
    async execute(_toolCallId, params, signal) {
      const result = await gateway("/trial", params, signal);
      // Keep bounded summaries available after compaction; full results remain in the step record.
      const details = result.details;
      observations.set("trial", { candidate: details.candidate, ran: details.ran, passed: details.passed,
        timed_out: details.timed_out, exit_code: details.exit_code, refusal: details.refusal,
        error: details.error,
        steps: details.steps?.map(step => ({ index: step.index, passed: step.passed,
          checks: step.checks.map(check => ({ name: check.name, passed: check.passed })) })) });
      return result;
    },
  });
}
