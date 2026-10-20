// A real pi SDK session with a local, deterministic OpenAI-compatible model.
// Requests are observed after pi's provider hooks; fixture tools record actual execution.
import { createServer } from "node:http";
import { readFile, realpath, writeFile, access } from "node:fs/promises";
import { constants } from "node:fs";
import { dirname, delimiter, join } from "node:path";
import { pathToFileURL } from "node:url";

const root = process.env.HOME;
const plan = JSON.parse(await readFile(join(root, "trial-script.json"), "utf8"));
const report = { passed: false, steps: [], errors: [] };
let requests = [];
let executed = [];
let results = [];
let currentStep;
let session;

// Resolve the installation on this executor, including a remote E2B template.
let binary;
for (const directory of process.env.PATH.split(delimiter)) {
  const candidate = join(directory, "pi");
  try {
    await access(candidate, constants.X_OK);
    binary = await realpath(candidate);
    break;
  } catch (error) {
    if (!['ENOENT', 'EACCES', 'ENOTDIR'].includes(error.code)) throw error;
  }
}
if (!binary) throw new Error("pi is not on PATH");
const packageRoot = dirname(dirname(binary));
const manifest = JSON.parse(await readFile(join(packageRoot, "package.json"), "utf8"));
if (manifest.name !== "@earendil-works/pi-coding-agent") throw new Error("unexpected pi installation");
report.pi_version = manifest.version;
const { createAgentSession, DefaultResourceLoader, SessionManager, SettingsManager, ModelRuntime } =
  await import(pathToFileURL(join(packageRoot, "dist/index.js")).href);

const server = createServer(async (request, response) => {
  try {
    let body = "";
    for await (const chunk of request) body += chunk;
    const payload = JSON.parse(body);
    const first = requests.length === 0;
    const call = first ? currentStep?.tool_call : undefined;
    requests.push({
      tools: (payload.tools ?? []).map(tool => tool.function.name),
      system: payload.messages.filter(message => ["system", "developer"].includes(message.role))
        .map(message => typeof message.content === "string" ? message.content : JSON.stringify(message.content)).join("\n"),
      messages: JSON.stringify(payload.messages.filter(message => !["system", "developer"].includes(message.role))),
    });
    if (requests.length > 8) throw new Error("scripted prompt exceeded 8 model requests");
    const delta = call
      ? { role: "assistant", tool_calls: [{ index: 0, id: `trial-${report.steps.length}`, type: "function",
          function: { name: call.name, arguments: JSON.stringify(call.arguments) } }] }
      : { role: "assistant", content: "OK" };
    response.writeHead(200, { "content-type": "text/event-stream" });
    for (const choice of [
      { index: 0, delta, finish_reason: null },
      { index: 0, delta: {}, finish_reason: call ? "tool_calls" : "stop" },
    ]) {
      response.write(`data: ${JSON.stringify({ id: "trial", object: "chat.completion.chunk",
        created: 0, model: "trial", choices: [choice] })}\n\n`);
    }
    response.end("data: [DONE]\n\n");
  } catch (error) {
    report.errors.push(String(error));
    response.writeHead(500, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: { message: String(error) } }));
  }
});

function checkStep(step) {
  const checks = [];
  for (const [name, expected] of Object.entries(step.expect ?? {})) {
    let actual;
    let passed;
    if (name === "model_called") {
      actual = requests.length > 0;
      passed = actual === expected;
    } else if (name === "executed_tools") {
      actual = executed;
      passed = JSON.stringify([...actual].sort()) === JSON.stringify([...expected].sort());
    } else if (name === "tool_errors") {
      actual = results.filter(result => result.isError).map(result => result.toolName);
      passed = JSON.stringify([...actual].sort()) === JSON.stringify([...expected].sort());
    } else {
      // Absence of a model call cannot prove a claim about a model request.
      const textField = name.startsWith("system_") ? "system" : name.startsWith("messages_") ? "messages" : undefined;
      actual = requests.map(request => textField ? request[textField] : request.tools);
      passed = requests.length > 0 && requests.every(request => {
        if (name === "tools") {
          return JSON.stringify([...request.tools].sort()) === JSON.stringify([...expected].sort());
        }
        if (name === "forbidden_tools") return expected.every(tool => !request.tools.includes(tool));
        if (name.endsWith("_contains")) return expected.every(text => request[textField].includes(text));
        return expected.every(text => !request[textField].includes(text));
      });
      if (textField) actual = actual.map(text => ({ characters: text.length }));
    }
    checks.push({ name, expected, actual, passed });
  }
  return checks;
}

async function startSession(reason = "startup") {
  const agentDir = process.env.PI_CODING_AGENT_DIR;
  const cwd = process.cwd();
  const settings = SettingsManager.inMemory({ ...SettingsManager.create(cwd, agentDir).getGlobalSettings(),
    compaction: { enabled: false }, retry: { enabled: false } });
  const runtime = await ModelRuntime.create({ authPath: join(agentDir, "trial-auth.json"),
    modelsPath: null, refreshOnCreate: false });
  runtime.registerProvider("reef-trial", {
    baseUrl: `http://127.0.0.1:${server.address().port}/v1`, apiKey: "trial-local",
    api: "openai-completions",
    models: [{ id: "trial", name: "trial", reasoning: false, input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 1000 }],
  });
  const loader = new DefaultResourceLoader({ cwd, agentDir, settingsManager: settings,
    extensionFactories: [pi => {
      for (const name of plan.fixture_tools) {
        pi.registerTool({ name, label: name, description: `Trial fixture: ${name}`,
          parameters: { type: "object", properties: {}, additionalProperties: true },
          async execute() {
            executed.push(name);
            return { content: [{ type: "text", text: `Executed fixture ${name}` }], details: {} };
          },
        });
      }
    }],
  });
  await loader.reload();
  const errors = loader.getExtensions().errors;
  if (errors.length) throw new Error(`extension load failed: ${JSON.stringify(errors)}`);
  const configuredNames = loader.getExtensions().extensions.flatMap(extension => [...extension.tools.keys()]);
  for (const name of plan.fixture_tools) {
    if (configuredNames.filter(tool => tool === name).length !== 1) {
      throw new Error(`fixture tool collides with candidate tool: ${name}`);
    }
  }
  const created = await createAgentSession({ cwd, agentDir, modelRuntime: runtime,
    model: runtime.getModel("reef-trial", "trial"), resourceLoader: loader, settingsManager: settings,
    sessionManager: SessionManager.inMemory(cwd), thinkingLevel: "off",
    sessionStartEvent: { type: "session_start", reason } });
  session = created.session;
  session.subscribe(event => {
    if (event.type === "tool_execution_end") {
      results.push({ toolName: event.toolName, isError: event.isError, result: event.result });
    }
  });
  await session.bindExtensions({ onError: error => report.errors.push(`${error.event}: ${error.error}`) });
}

try {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  await startSession();
  for (const [index, step] of plan.steps.entries()) {
    requests = [];
    executed = [];
    results = [];
    currentStep = step;
    if (step.new_session) {
      await session.extensionRunner.emit({ type: "session_shutdown", reason: "new" });
      session.dispose();
      await startSession("new");
    } else {
      await session.prompt(step.prompt);
    }
    const last = session.messages.at(-1);
    if (last?.role === "assistant" && ["error", "aborted"].includes(last.stopReason)) {
      report.errors.push(last.errorMessage ?? last.stopReason);
    }
    const checks = checkStep(step);
    report.steps.push({ index, passed: checks.every(check => check.passed), checks,
      model_requests: requests.length, tools: requests.map(request => request.tools),
      executed_tools: executed, tool_results: results });
  }
  report.passed = report.errors.length === 0 && report.steps.every(step => step.passed);
} catch (error) {
  report.errors.push(String(error));
} finally {
  if (session) {
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
  }
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
  await writeFile(join(root, "sessions/trial-result.json"), JSON.stringify(report));
}
process.exitCode = report.passed ? 0 : 1;
