---
name: reef-pi-extension-api
description: The pi 0.84.2 extension API in brief. Read before writing or changing a code_extension entry for reef-pi. Covers the file shape, tools with typebox parameters, commands, events, ctx.ui, messages, exec, and the rules a reef tree entry must keep.
---
# pi extension API (0.84.2)

An extension is one module at pi-agent/extensions/<name>.ts. pi loads it with jiti, so plain JavaScript in a .ts file runs as is; type annotations are allowed but not needed. A tree entry has no npm install: import only node: modules, typebox, @earendil-works/pi-coding-agent, @earendil-works/pi-ai and @earendil-works/pi-tui.

## File shape

```ts
import { Type } from "typebox";

export default function (pi) {
  pi.on("session_start", async (_event, ctx) => { ... });
  pi.registerTool({ ... });
  pi.registerCommand("name", { ... });
}
```

The default export is a factory that receives the extension API. It may be async; pi awaits it before session_start. Do not start processes, sockets, watchers or timers in the factory: start them in session_start or in the tool or command that needs them, and stop them in a session_shutdown handler.

## Tools: pi.registerTool

```ts
pi.registerTool({
  name: "word_count",
  label: "Word count",
  description: "Count the words in a file (shown to the model)",
  promptSnippet: "Count words in a file",
  promptGuidelines: ["Use word_count instead of wc when the user asks for a word count."],
  parameters: Type.Object({
    path: Type.String({ description: "file to count" }),
    unit: Type.Optional(Type.Unsafe({ type: "string", enum: ["words", "lines"] })),
  }),
  async execute(toolCallId, params, signal, onUpdate, ctx) {
    const result = await pi.exec("wc", ["-w", params.path], { signal });
    if (result.code !== 0) throw new Error(result.stderr.trim());
    return { content: [{ type: "text", text: result.stdout.trim() }], details: {} };
  },
});
```

- parameters is a typebox schema. Type.Object, Type.String, Type.Number, Type.Boolean, Type.Array, Type.Optional. For a string choice use StringEnum from @earendil-works/pi-ai; Type.Union of literals breaks on some providers.
- execute(toolCallId, params, signal, onUpdate, ctx) returns { content: [{ type: "text", text }], details? }. content goes to the model; details is for rendering and state.
- Throw an Error to report a failure; a returned value is never an error.
- onUpdate?.({ content: [...] }) streams progress. Check signal?.aborted for cancellation and pass signal to fetch and pi.exec.
- promptSnippet puts one line in the system prompt's tool list; each promptGuidelines bullet must name the tool.

## Commands: pi.registerCommand

```ts
pi.registerCommand("standup", {
  description: "Summarize today's work",
  handler: async (args, ctx) => {
    if (!args.trim()) { ctx.ui.notify("Usage: /standup <since>", "warning"); return; }
    pi.sendUserMessage(`Summarize the work since ${args}`);
  },
});
```

The handler gets the text after /standup as args. A command runs no model call by itself; send a user message to start a turn. The reef-harness and reef-versions commands and the reef_ask_user and reef_file_request tools belong to reef: register nothing under those names.

## Events: pi.on(name, handler)

Every handler receives (event, ctx). The ones that matter:

| event | when | return |
|-------|------|--------|
| session_start | a session starts, resumes or reloads; event.reason | nothing |
| agent_start | a run begins after the user's prompt | nothing |
| agent_end | that run ends; event.messages | nothing |
| tool_call | before a tool runs; event.toolName, event.input (mutable) | { block: true, reason } to stop it |
| tool_result | after a tool ran; event.toolName, event.content, event.isError | { content } to replace the result |
| turn_end | one model response and its tool calls are done; event.turnIndex, event.message, event.toolResults | nothing |

Also: before_agent_start (return { systemPrompt } to add instructions for the turn), session_shutdown (clean up), input (event.text; return { action: "handled" } to answer without the model).

## ctx

- ctx.hasUI: true in the TUI and RPC modes, false under -p and --mode json. Guard every dialog with it.
- ctx.ui.notify(text, "info" | "warning" | "error"): a line that does not block.
- await ctx.ui.confirm(title, message): boolean.
- await ctx.ui.select(title, options): the chosen string or undefined. Undefined is Escape: treat it as the person backing out, not as a skipped question.
- await ctx.ui.input(title, placeholder): a string or undefined.
- Every dialog takes an options argument, { signal, timeout }: pass ctx.signal (or the tool's own) so an aborted turn dismisses it.
- ctx.ui.setStatus(key, text): a footer status until cleared; pass undefined to clear.
- ctx.ui.setWidget(key, lines): an array of strings shown above the input box until cleared with undefined. This is where a long job's progress belongs, so the session's own output stays the person's.
- ctx.cwd, ctx.model, ctx.signal (the turn's abort signal), ctx.isIdle().
- ctx.sessionManager.getSessionId(), getSessionFile(), getEntries(), getBranch().

## Keys

- pi.registerShortcut("ctrl+shift+r", { description, handler: async (ctx) => {} }): a key the person presses. There is no click target for a widget, so a key is how a person opens what a widget shows.
- pi binds most ctrl+letter keys itself, among them ctrl+a, ctrl+c, ctrl+d, ctrl+g, ctrl+l, ctrl+n, ctrl+o, ctrl+p, ctrl+r, ctrl+s, ctrl+t, ctrl+u, ctrl+v, ctrl+x and ctrl+z. Registering one of those makes pi warn at startup about the clash, so add shift: ctrl+shift+<letter> is free.

## Messages

- pi.sendUserMessage(text): a user message that starts a turn. While the agent streams pass { deliverAs: "steer" } or { deliverAs: "followUp" }; without one it throws.
- pi.sendMessage({ customType, content, display: true }, { triggerTurn: true }): a custom message in the model's context.
- pi.appendEntry(customType, data): persisted, not in the model's context.
- pi.sendUserMessage("/name args", { expandPromptTemplates: true }) runs your own command /name instead of starting a turn. An event handler reaches what only a command's ctx has this way.

## Sessions

- A session is one saved conversation: a .jsonl file in ctx.sessionManager.getSessionDir(), one directory per project. pi starts a new one on every launch; event.reason on session_start is "startup" for that launch.
- ctx.switchSession(path, { withSession }), ctx.newSession({ withSession }) and ctx.fork(entryId, { withSession }) exist only on a command's ctx. They replace the session the person sees, history included, and leave the old ctx stale: do follow-up work in withSession, with the ctx it receives.
- To open another session at launch, register a command that calls ctx.switchSession, and from session_start with reason "startup" run it with pi.sendUserMessage as above. Never paste an old transcript into the new session's context in its place: the person would still see an empty session.

## Running commands: pi.exec

```ts
const result = await pi.exec("git", ["status", "--short"], { signal, timeout: 5000 });
// result.stdout, result.stderr, result.code, result.killed
```

Pass values as arguments, never as shell source. The directory of this harness's own pi binary is first on PATH, so pi.exec("pi", ["-p", instruction]) starts a second session with the same models and extensions.

## Network

fetch is global. Pass signal. Reef's own routes take the headers { "x-reef-scenario": process.env.REEF_SCENARIO } and, when set, { authorization: `Bearer ${process.env.REEF_TOKEN}` }; the service is at process.env.REEF_SERVICE_URL.

Models beyond the session's chat model are Reef routes too, when the Reef recipe configures a multimodal provider: POST a JSON body in that provider's own format (OpenRouter's by default) to process.env.REEF_SERVICE_URL + one of the routes below, with Reef's headers above and { "content-type": "application/json" }. Reef adds the provider's key; the extension holds none.

- /v1/images: generate an image from a prompt.
- /v1/embeddings: embed text.
- /v1/audio/speech: text to speech; the response body is the audio bytes.
- /v1/decisions: a fast structured choice (routing, classification, a risk or completion check) from a decision model such as ~typesafe/jev-latest, where the provider serves one. The body carries a state and typed questions (noul, choice, score); the answer is a value with probabilities, never text.

Name the model in the body. These routes do not stream, and answer 501 when the Reef recipe configures no multimodal provider or its provider serves no such route.

## Rules for a reef tree entry

- Return before registering anything when process.env.PI_OFFLINE is set: evaluation episodes are hermetic and must see no network calls, prompts or timers.
- Credentials come from process.env at run time, never from the file: admission refuses a credential shaped literal, and the tree persists every version.
- Keep state in tool result details, not in module variables, so a resumed or forked session rebuilds it.
- Never throw out of an event handler for an expected condition: log with ctx.ui.notify or return nothing.
- One file, no dependencies, ASCII text.

## A complete example

```ts
import { Type } from "typebox";

export default function (pi) {
  if (process.env.PI_OFFLINE) return;

  pi.registerTool({
    name: "note",
    label: "Note",
    description: "Append one line to NOTES.md in the working directory",
    parameters: Type.Object({ line: Type.String() }),
    async execute(_id, params, signal, _onUpdate, ctx) {
      const result = await pi.exec("sh", ["-c", 'printf "%s\\n" "$1" >> NOTES.md', "note", params.line], { signal });
      if (result.code !== 0) throw new Error(result.stderr.trim());
      return { content: [{ type: "text", text: `noted in ${ctx.cwd}/NOTES.md` }], details: {} };
    },
  });

  pi.registerCommand("notes", {
    description: "Show NOTES.md",
    handler: async (_args, ctx) => {
      const result = await pi.exec("cat", ["NOTES.md"]);
      ctx.ui.notify(result.code === 0 ? result.stdout : "no NOTES.md yet", "info");
    },
  });

  pi.on("tool_call", async (event, ctx) => {
    if (event.toolName === "bash" && /\brm -rf\b/.test(event.input.command || "")) {
      if (!ctx.hasUI) return { block: true, reason: "rm -rf needs a person to confirm" };
      const ok = await ctx.ui.confirm("Dangerous command", event.input.command);
      if (!ok) return { block: true, reason: "declined" };
    }
  });
}
```
