// Harness requests: the /reefine and /versions commands and the
// reef_ask_user and reef_file_request tools for reef-pi. The person asks in
// plain words. With a UI the command clarifies the request in the background:
// a loop beside the session calls the session's model with the request, the
// recent conversation and the two tools, asks what is unclear (reef_ask_user)
// and files it (reef_file_request). The chat keeps one collapsed entry for it,
// whose expanded view holds the whole clarification, so the session's own
// context and transcript stay free of it; with --direct, or headless, the
// command files it as is. A filed request goes to
// reef with this session's id and the installed release through native manual
// training, and every filing answers with a link to the request's page. The
// service proposer writes the change, and a watch here polls the catalog, shows
// in the footer whether the request is queued or its step is running and for
// how long, and reports the step's result in the session as a custom message
// the chat keeps, with why the proposer produced nothing when it did. The
// filed requests not yet reported are kept beside the release file, so a
// restarted pi reports their results at its next session start. /versions
// lists the release chain with each step's result and request, marks the step
// this tree runs as installed and the newest published one as current, and
// offers a step's page, which holds the design, the review and the numbers.
// A settled step offers its install once the session is between turns, so a
// win reaches the person who asked without them going looking; a busy session
// keeps the report's commands instead. /versions <step> install runs the
// same install on demand, promoting a release held back from the served head
// first, so installing is the one decision. Nothing here writes
// a mutation. Kept free of annotations on purpose: plain JavaScript in a .ts file, so plain
// node can parse it in CI and pi's TS loader accepts it unchanged; pi-tui, which draws the entry, is imported
// lazily from pi's own loader, so plain node loads the file without it. Evaluation
// episodes set PI_OFFLINE and this extension then registers nothing, so the
// evaluation never sees the commands or the tools.
import { createHash, createHmac } from "node:crypto";
import { accessSync, constants, existsSync, readFileSync, writeFileSync } from "node:fs";
import { release } from "node:os";
import { delimiter, join } from "node:path";

// The release file the install script and harness_pull write at the tree root.
const RELEASE_FILE = ".reef-harness-release";
// The wrapper the install runs through: run_agent exports its path, and a tree run directly has it beside the
// release file; without either the person hears the commands instead.
const WRAPPER_NAME = "reef-pi";
const INSTALL_LATER_TEXT = "reef: install it later with reef-pi update, then reef-pi setup";
const NO_WRAPPER_TEXT = "reef: no reef-pi wrapper found; install it with reef-pi update, then reef-pi setup";
// The installed marker in a version's detail dialog. The install channel writes the release file naming it.
const INSTALLED_MARK = "installed (this tree)";
// The wrapper's exit code for an update it refused because an item is unmet: the setup loop runs, then the
// update again.
const UPDATE_REFUSED_CODE = 3;
// The filed requests not yet reported, beside the release file: {id, text, filed_at} entries, the newest ten,
// none older than a day.
const REQUESTS_FILE = ".reef-harness-requests.json";
const REQUESTS_MAX = 10;
const REQUESTS_MAX_AGE_MS = 24 * 60 * 60 * 1000;
// The watch polls the catalog (and, until a step takes the request, its record) once per interval and gives up
// at the cap.
const WATCH_INTERVAL_MS = 5000;
const WATCH_CAP_MS = 30 * 60 * 1000;
// How many of the proposer's latest moves the opened spinner lists; the request page has them all.
const ACTIVITY_LINES = 4;
// The commands a request reports as on this machine's PATH or not, so the proposer builds for this machine rather
// than for the sandbox it tries the change in; the same list as reef.core.training_request.CLIENT_COMMANDS.
const CLIENT_COMMANDS = [
  "afplay",
  "say",
  "osascript",
  "open",
  "pbcopy",
  "terminal-notifier",
  "xdg-open",
  "notify-send",
  "paplay",
  "pw-play",
  "aplay",
  "wl-copy",
  "xclip",
  "powershell.exe",
  "wslview",
  "ffplay",
  "ffmpeg",
  "mpv",
  "mpg123",
  "sox",
  "espeak",
  "curl",
  "git",
  "gh",
  "python3",
  "node",
  "brew",
  "apt-get",
];
// Every request to reef gives up after this: a hung connection must not stall a command or the watch's ticks.
const FETCH_TIMEOUT_MS = 10000;
// The custom message type the report is appended to the session as; pi renders plain text content itself.
const REPORT_MESSAGE_TYPE = "reef-harness";
// The spinner above the input box while a step runs: its key, its frames and how often they turn. The widget
// sits above the editor, so the step is visible without taking the screen or the person's input.
const WIDGET_KEY = "reef-harness";
const SPINNER_FRAMES = ["|", "/", "-", "\\"];
const SPINNER_INTERVAL_MS = 250;
// A terminal hyperlink (OSC 8): pi's TUI measures around it, and a click opens the URL. A terminal without
// hyperlink support shows the label alone, so the widget's other ways in stay the ones that always work.
function link(url, label) {
  return `\x1b]8;;${url}\x1b\\${label}\x1b]8;;\x1b\\`;
}

// The key that opens the spinner's detail, which the spinner itself names so the person knows it is there.
// It must be a plain ctrl+letter that pi leaves free: a terminal without the Kitty keyboard protocol or xterm's
// modifyOtherKeys (Apple Terminal among them) sends ctrl+shift+<letter> as the bare control byte, so pi reads
// ctrl+shift+r as ctrl+r, its session rename. ctrl+q is the letter pi binds nowhere, and its control byte, the
// Kitty sequence and the modifyOtherKeys sequence all match it.
const WATCH_SHORTCUT = "ctrl+q";
// The service's phase for a running step, in the words the spinner and the panel show.
const PHASE_WORDS = {
  queued: "queued, waiting for a step",
  proposing: "writing the change",
  evaluating: "checking the harness",
  running: "running the step",
  settling: "saving the result",
};
// The service caps a request's text; the filed text stays within it.
const REQUEST_MAX_CHARS = 4000;
// The choice under every question that opens a free text answer, and the one that drops the request.
const OTHER = "Other (type an answer)";
const CANCEL = "Cancel this request";
// The mark on the option the model recommends, which is listed first; the answer filed is the option alone.
const RECOMMENDED = " (recommended)";
const NO_UI_TEXT = "no UI in this session: proceed with your best assumptions and list them in the request";
// What the model is told when the person backs out: it must not file, and it must not ask again.
const CANCELLED_TEXT =
  "the user cancelled this harness request: do not file it, do not ask again, and say it was cancelled";
// The background clarification: the entry type the chat keeps it as, the widget that shows it while it runs, the
// model calls it may take, and how much of the recent conversation it reads as background.
const CLARIFY_ENTRY_TYPE = "reef-harness-clarify";
const CLARIFY_WIDGET_KEY = "reef-harness-clarify";
const CLARIFY_MAX_TURNS = 8;
const CONTEXT_MESSAGES = 6;
const CONTEXT_MESSAGE_CHARS = 1200;

// Tool parameters as plain JSON schema: pi compiles them with typebox, which reads JSON schema as is, so the
// extension needs no static import beyond node.
const ASK_USER_PARAMETERS = {
  type: "object",
  properties: {
    questions: {
      type: "array",
      minItems: 1,
      maxItems: 4,
      items: {
        type: "object",
        properties: {
          question: { type: "string", description: "one open point, as a question" },
          options: { type: "array", items: { type: "string" }, minItems: 2, maxItems: 4 },
          recommended: {
            type: "string",
            description: "the option you would choose, word for word as it appears in options, when one is clearly better",
          },
        },
        required: ["question", "options"],
      },
    },
  },
  required: ["questions"],
};
const FILE_REQUEST_PARAMETERS = {
  type: "object",
  properties: {
    request: { type: "string", description: "the user's original words" },
    clarifications: {
      type: "array",
      items: {
        type: "object",
        properties: { question: { type: "string" }, answer: { type: "string" } },
        required: ["question", "answer"],
      },
    },
  },
  required: ["request"],
};
// The two tools as the model sees them, registered on the session and offered to the background clarification.
const ASK_USER_TOOL = {
  name: "reef_ask_user",
  label: "Ask the user",
  description:
    "Ask the user before filing a harness change with reef_file_request, only about a decision that changes " +
    "what gets built and that a reasonable default cannot settle: a clear request needs no question. Each " +
    "question is one decision with 2 to 4 concrete options that do not overlap; name the one you would choose " +
    "as recommended when one is clearly better. The user can always type an answer of their own, and can " +
    "cancel the whole request. Never ask for a setup value (a phone number, a credential, an account, a " +
    "permission): reef-pi setup collects those after the install.",
  parameters: ASK_USER_PARAMETERS,
};
const FILE_REQUEST_TOOL = {
  name: "reef_file_request",
  label: "File a harness request",
  description:
    "File a harness change with reef: the user's original request and the answers reef_ask_user collected. " +
    "Reef's service writes the change and reports here when the step settles.",
  parameters: FILE_REQUEST_PARAMETERS,
};

function readJson(path) {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
}

function message(error) {
  return error instanceof Error ? error.message : String(error);
}

// The first `limit` characters of a text, the cut marked.
function clip(text, limit) {
  return text.length > limit ? `${text.slice(0, limit - 3)}...` : text;
}

// The poll interval, from the environment so a test can shorten it; the cap stays.
function watchIntervalMs() {
  const configured = Number(process.env.REEF_HARNESS_WATCH_MS);
  return Number.isFinite(configured) && configured > 0 ? configured : WATCH_INTERVAL_MS;
}

// The fetch deadline, from the environment so a test can shorten it.
function fetchTimeoutMs() {
  const configured = Number(process.env.REEF_HARNESS_FETCH_MS);
  return Number.isFinite(configured) && configured > 0 ? configured : FETCH_TIMEOUT_MS;
}

// A request to reef with a deadline. The signal covers the connect, the headers and the body, so a hung
// service costs one timeout and no more; the body is read here, and json() parses it on demand as fetch's does.
async function fetchWithTimeout(url, init = {}) {
  const controller = new AbortController();
  const timeoutMs = fetchTimeoutMs();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const body = await response.text();
    return { ok: response.ok, status: response.status, text: async () => body, json: async () => JSON.parse(body) };
  } catch (error) {
    throw controller.signal.aborted ? new Error(`no answer within ${timeoutMs} ms`) : error;
  } finally {
    clearTimeout(timer);
  }
}

// What the clarification's model does with a request before it files it. It sees no file and runs no command, so
// the prompt says what reef-pi is: without it a model reads the name as an unrelated web app and asks about pages.
const CLARIFY_SYSTEM_PROMPT = [
  "You clarify a person's request for a change to their reef-pi harness and file it with reef.",
  "reef-pi is pi, a coding agent that runs in a terminal (not a web or browser app), started with the reef-pi " +
    "command in a project directory. Its harness is the files pi loads at startup: TypeScript extensions that " +
    "run inside pi's process (commands, tools, handlers for events such as session_start, dialogs and widgets " +
    "in the terminal UI), skills, AGENTS.md rules, prompt templates and settings.json. A session is one saved " +
    "pi conversation, a JSONL file per project directory; pi starts a new session on every launch.",
  "Word every question and default in those terms, in the language the request is written in. You cannot " +
    "read files or run commands, and you do not " +
    "write the change: reef's service writes it. Use only the reef_ask_user and reef_file_request tools, and " +
    "end by filing the request.",
].join("\n\n");

// The clarification's first message: the recent conversation as background, then the request, each as data in
// a fence.
function clarifyMessage(text, conversation) {
  const background = conversation
    ? ["The recent conversation in the session, as background for what the request refers to:", "", "```", conversation, "```", ""]
    : [];
  return [
    ...background,
    "The user asked for this harness change:",
    "",
    "```",
    text,
    "```",
    "",
    "Before filing it with reef_file_request, decide what would be built: when the behavior triggers, what it " +
      "does, what state it keeps and how it learns that state. Ask with reef_ask_user only about a decision that " +
      "changes what gets built, that the request leaves open, and that a reasonable default cannot settle: a " +
      "clear request needs no question, so file it at once. Rules for the questions:",
    "- as few as the request needs, often none; one decision per question, worded so the user can answer " +
      "without knowing how the harness works;",
    "- 2 to 4 options that are concrete, mutually exclusive and cover the likely answers; no two options that " +
      "mean the same thing; the user can always type their own;",
    "- when one option is clearly the better choice, name it as recommended: it is shown first, marked, and the " +
      "user can take it in one keystroke;",
    "- when the person's own machine can do what the request asks and a model the deployment pays for can do " +
      "it too, which one runs is a decision, not a setup detail: ask it, and let the options say what each " +
      "side gives up, the machine's being free, offline and only as good as what is installed, the model's " +
      "being billed for every use and better at it;",
    "- never ask for a value or a setup detail the user provides when the change is installed: a phone number, " +
      "a credential, an account, a permission, or which app or service to use when the request already names " +
      "one; reef-pi setup collects those once, after the install;",
    "- do not ask what a reasonable default settles; choose the default and say so in a clarification line.",
    "Then call reef_file_request with the user's original words as `request` and the answers as " +
      "`clarifications`, one line per answer and one per default you chose. Do not write the change yourself: " +
      "reef's service writes it.",
  ].join("\n");
}

// The filed text: the request verbatim, then the answers as question and answer pairs.
function filedText(request, clarifications) {
  const pairs = (Array.isArray(clarifications) ? clarifications : []).filter(
    (item) => item && typeof item.question === "string" && typeof item.answer === "string",
  );
  const lines = [request.trim()];
  if (pairs.length) lines.push("", "Clarifications:", ...pairs.map((item) => `- Q: ${item.question}\n  A: ${item.answer}`));
  return lines.join("\n").slice(0, REQUEST_MAX_CHARS);
}

// The metrics a catalog row carries, or nothing.
function metricsOf(row) {
  return row && row.metrics && typeof row.metrics === "object" ? row.metrics : {};
}

// The record id of the request a step consumed; the watch keys its row by it.
function requestIdOf(row) {
  const request = metricsOf(row).training_request;
  return request && typeof request.id === "string" ? request.id : null;
}

// What the review left uncovered, when the step recorded a review.
function uncoveredOf(row) {
  const notes = metricsOf(row).proposal_notes;
  const review = notes && typeof notes === "object" ? notes.review : null;
  const items = review && Array.isArray(review.uncovered) ? review.uncovered : [];
  return items.filter((item) => typeof item === "string" && item.trim()).map((item) => item.trim());
}

// Why the proposer produced nothing, when the step recorded it beside its notes.
function failureOf(row) {
  const notes = metricsOf(row).proposal_notes;
  return notes && typeof notes === "object" && typeof notes.failure === "string" ? notes.failure.trim() : "";
}

// An elapsed time as the footer shows it: minutes and two digit seconds.
function elapsedText(ms) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

// The text parts of a message's content, which is a string or a list of parts.
function textOfContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((part) => part && part.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("\n");
}

// The last few user and assistant messages on the session's branch, their text only and each clipped: what a
// request such as "the thing we just discussed" refers to. Tool calls and their output stay out.
function recentConversation(sessionManager) {
  const lines = [];
  for (const entry of sessionManager.getBranch()) {
    if (entry.type !== "message" || !["user", "assistant"].includes(entry.message.role)) continue;
    const text = textOfContent(entry.message.content).trim();
    if (text) lines.push(`${entry.message.role}: ${clip(text, CONTEXT_MESSAGE_CHARS)}`);
  }
  return lines.slice(-CONTEXT_MESSAGES).join("\n\n");
}

// What is wrong with a tool call's arguments, or null. pi validates the registered tools' arguments against
// their schema; the clarification calls the model itself, so it checks the fields the tools read.
function argumentsProblem(name, args) {
  if (name === ASK_USER_TOOL.name) {
    const questions = args && args.questions;
    const usable =
      Array.isArray(questions) &&
      questions.length > 0 &&
      questions.every((item) => item && typeof item.question === "string" && Array.isArray(item.options));
    return usable ? null : "questions must be a non-empty list of {question, options}";
  }
  if (name === FILE_REQUEST_TOOL.name) {
    return args && typeof args.request === "string" && args.request.trim() ? null : "request must be a non-empty string";
  }
  return `no tool named ${name}; use ${ASK_USER_TOOL.name} or ${FILE_REQUEST_TOOL.name}`;
}

export default function requests(pi) {
  if (process.env.PI_OFFLINE) return; // hermetic episodes never see the commands or the tools
  const agentDir = process.env.PI_CODING_AGENT_DIR;
  const serviceUrl = process.env.REEF_SERVICE_URL;
  const scenario = process.env.REEF_SCENARIO;
  if (!agentDir || !serviceUrl || !scenario) return;
  // pi-tui draws the clarification's entry; pi's loader resolves it, and without it the entry draws nothing.
  let tui = null;
  import("@earendil-works/pi-tui").then(
    (module) => {
      tui = module;
    },
    () => {},
  );
  // The wrapper relocates the agent into a temp copy and exports the true
  // install root; a tree run directly falls back to the release file beside it.
  const destDir = process.env.REEF_HARNESS_DEST || join(agentDir, "..");

  const reefHeaders = () => {
    const token = process.env.REEF_TOKEN;
    return { "x-reef-scenario": scenario, ...(token ? { authorization: `Bearer ${token}` } : {}) };
  };

  // A page a browser opens: the query carries the scenario and, in place of the token, its page key (reef's
  // reef/core/page_key.py), which opens this scenario's two pages alone. The model reads these links in tool
  // results and prompts, so they must not carry the token.
  const pageKey = (token) =>
    createHmac("sha256", createHash("sha256").update(token, "utf8").digest())
      .update(`reef-page\n${scenario}`, "utf8")
      .digest("hex");
  const pageLink = (path) => {
    const token = process.env.REEF_TOKEN;
    const query = `scenario=${encodeURIComponent(scenario)}${token ? `&key=${pageKey(token)}` : ""}`;
    return `${serviceUrl}${path}?${query}`;
  };
  const requestPageLink = (recordId) => pageLink(`/reef/harness/requests/${encodeURIComponent(recordId)}/page`);
  const stepPageLink = (step) => pageLink(`/reef/harness/releases/${step}/page`);

  const installedRelease = () => {
    const releaseInfo = readJson(join(destDir, RELEASE_FILE));
    return releaseInfo && typeof releaseInfo.release_id === "string" && releaseInfo.release_id ? releaseInfo.release_id : null;
  };

  const noReleaseText = () =>
    `no ${RELEASE_FILE} release file at ${destDir}: this tree did not come through reef's install channel, ` +
    "so a request cannot name the release it runs; nothing was sent";

  // This machine as a request reports it: the platform, and which of CLIENT_COMMANDS are on the PATH, read from the
  // PATH's directories without running anything.
  const clientReport = () => {
    const dirs = (process.env.PATH || "").split(delimiter).filter(Boolean);
    const onPath = (name) =>
      dirs.some((dir) => {
        try {
          accessSync(join(dir, name), constants.X_OK);
          return true;
        } catch {
          return false;
        }
      });
    return {
      platform: process.platform,
      arch: process.arch,
      release: release(),
      commands: Object.fromEntries(CLIENT_COMMANDS.map((name) => [name, onPath(name)])),
    };
  };

  // POST the request with this session and the installed release; the answer names the record the step's
  // catalog row carries. Throws with the message the notice shows.
  const fileRequest = async (text, ctx) => {
    const releaseId = installedRelease();
    if (!releaseId) throw new Error(noReleaseText());
    const body = { text, session: ctx.sessionManager.getSessionId(), release_id: releaseId, client: clientReport() };
    let response;
    try {
      // Not under the turn's abort signal: an Esc after the body went out would report a filed request as unreachable.
      response = await fetchWithTimeout(`${serviceUrl}/reef/train`, {
        method: "POST",
        headers: { ...reefHeaders(), "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (error) {
      throw new Error(`reef unreachable at ${serviceUrl}: ${message(error)}`);
    }
    if (!response.ok) throw new Error(`reef refused the request (HTTP ${response.status}): ${await response.text()}`);
    const answer = await response.json();
    return String(answer.agent_record_id);
  };

  // The catalog oldest first; a step is a row's position in it, the creation row being 0, which is the commit
  // step the service keys the page by (a rejected step publishes nothing, so only its position names it).
  const releases = async () => {
    let response;
    try {
      response = await fetchWithTimeout(`${serviceUrl}/reef/harness/releases`, { headers: reefHeaders() });
    } catch (error) {
      throw new Error(`reef unreachable at ${serviceUrl}: ${message(error)}`);
    }
    if (!response.ok) throw new Error(`reef refused the catalog read (HTTP ${response.status}): ${await response.text()}`);
    const rows = (await response.json()).releases;
    return Array.isArray(rows) ? rows : [];
  };

  // Where the step for a request stands: the service's own phase for it, which the record alone cannot tell.
  // An older service has no such route; a 404 there reads as no progress rather than a lost request.
  const requestProgress = async (recordId) => {
    const path = `/reef/harness/requests/${encodeURIComponent(recordId)}/progress`;
    const response = await fetchWithTimeout(`${serviceUrl}${path}`, { headers: reefHeaders() });
    if (!response.ok) return null;
    return await response.json();
  };

  // Read the record only to detect a request removed from storage.
  const requestRecord = async (recordId) => {
    const path = `/reef/scenarios/${encodeURIComponent(scenario)}/records/${encodeURIComponent(recordId)}`;
    const response = await fetchWithTimeout(`${serviceUrl}${path}`, { headers: reefHeaders() });
    if (response.status === 404) return null; // the service no longer knows the request: its scenario was reset
    if (!response.ok) throw new Error(`reef refused the record read (HTTP ${response.status})`);
    return await response.json();
  };

  // A request the service no longer knows is dropped and said once, instead of a watch that never settles.
  const goneText = (id8) =>
    `reef: request ${id8} is no longer on the service (its scenario was reset); ask again with /reefine`;

  // A promoted row stays pending in the catalog; the promote is a later row naming it, so with the rows given
  // the pending row reads "promoted at vN".
  const resultOf = (row, rows = []) => {
    if (row.pending) {
      const promoted = rows.findIndex(
        (other) => other.operation === "promote" && other.rollback_target_release_id === row.release_id,
      );
      return promoted >= 0 ? `promoted at v${promoted}` : "pending";
    }
    const metrics = metricsOf(row);
    if (typeof metrics.selected === "boolean") return metrics.selected ? "selected" : "rejected";
    if (metrics.skipped) return "skipped";
    return String(row.operation || "unknown");
  };

  // The one line a settled step earns, with the next action, quoting the request; the wrapper prints the same.
  // Every line names the step, whose page holds the details.
  const settledText = (step, rows, ask) => {
    const row = rows[step];
    const metrics = metricsOf(row);
    const release = String(row.release_id || "").slice(0, 8);
    const selectionResult = resultOf(row, rows);
    const details = ` Details: /versions v${step}.`;
    if (selectionResult === "selected") {
      return (
        `reef: '${ask}' is published as release ${release}. Install when ready with /versions v${step} install.` +
        details
      );
    }
    if (selectionResult === "pending") {
      return (
        `reef: '${ask}' is ready as release ${release}. This release changes an extension, so read it before ` +
        `it runs: /versions v${step} opens the page, /versions v${step} install serves it.`
      );
    }
    if (selectionResult === "rejected") {
      const reason = metrics.selection && metrics.selection.reason ? metrics.selection.reason : "no reason recorded";
      return (
        `reef: '${ask}' did not pass the checks (${reason}). Nothing changed; rephrase or split the request.` + details
      );
    }
    if (selectionResult === "skipped") {
      // The proposer's own reason, when the step recorded one: a failed model call, a reply with no entry.
      const failure = failureOf(row);
      const why = failure ? `${metrics.skipped}: ${failure}` : String(metrics.skipped);
      return `reef: '${ask}' produced no change (${why}). Nothing changed.${details}`;
    }
    return `reef: '${ask}' settled as ${selectionResult} (release ${release}); /versions v${step} shows it.`;
  };

  // The filed requests not yet reported, newest last, filed_at in seconds since the epoch as the service records
  // its times; an entry older than a day is dropped on read, and a write keeps the newest ten.
  const storedRequests = () => {
    const entries = readJson(join(destDir, REQUESTS_FILE));
    const oldest = (Date.now() - REQUESTS_MAX_AGE_MS) / 1000;
    return (Array.isArray(entries) ? entries : []).filter(
      (entry) =>
        entry && typeof entry.id === "string" && typeof entry.text === "string" && Number(entry.filed_at) > oldest,
    );
  };
  const writeStoredRequests = (entries) => {
    try {
      writeFileSync(join(destDir, REQUESTS_FILE), `${JSON.stringify(entries.slice(-REQUESTS_MAX), null, 2)}\n`);
    } catch {
      // An install root that cannot be written loses the restart safety only: this session's watch still reports.
    }
  };
  const rememberRequest = (recordId, text) => {
    const others = storedRequests().filter((entry) => entry.id !== recordId);
    writeStoredRequests([...others, { id: recordId, text, filed_at: Date.now() / 1000 }]);
  };
  const forgetRequest = (recordId) => writeStoredRequests(storedRequests().filter((entry) => entry.id !== recordId));

  // The report for a settled step: the result line and what the review left uncovered. It is appended to the
  // session as a custom message, which the chat renders and the session file keeps, and shown as a notice, the
  // one line a person sees at once but the next status line may overwrite.
  const deliverReport = (step, rows, text, ctx) => {
    const lines = [settledText(step, rows, clip(text.trim(), 60))];
    const uncovered = uncoveredOf(rows[step]);
    if (uncovered.length) lines.push(`Not covered: ${uncovered.join("; ")}`);
    const content = lines.join("\n");
    pi.sendMessage({ customType: REPORT_MESSAGE_TYPE, content, display: true }, { triggerTurn: false });
    ctx.ui.notify(content, "info");
  };

  // The wrapper the next steps run through: the one run_agent exported, else the one beside the release file.
  const wrapperPath = () => {
    const exported = process.env.REEF_HARNESS_WRAPPER;
    if (exported && existsSync(exported)) return exported;
    const beside = join(destDir, WRAPPER_NAME);
    return existsSync(beside) ? beside : null;
  };

  // One wrapper call; a wrapper that could not be started reads as a failed one.
  const runWrapper = async (wrapper, args) => {
    try {
      return await pi.exec(wrapper, args);
    } catch (error) {
      return { stdout: "", stderr: message(error), code: 1, killed: false };
    }
  };

  const textOf = (value, fallback) => (typeof value === "string" && value.trim() ? value : fallback);

  // The setup loop: what the release still needs from the person, asked here once and stored by the wrapper (an
  // env value in its env file, a check off for a check that passed). A declined item stays unmet and is named at
  // the end. Returns the names left unmet, or null when the wrapper could not list the items.
  const runSetup = async (wrapper, releaseId, ctx) => {
    const listed = await runWrapper(wrapper, ["setup", "--json", "--release", releaseId]);
    if (listed.code !== 0) {
      ctx.ui.notify(listed.stderr.trim() || `reef: reef-pi setup --json exited ${listed.code}`, "error");
      return null;
    }
    let items;
    try {
      items = JSON.parse(listed.stdout).items;
    } catch (error) {
      ctx.ui.notify(`reef: reef-pi setup --json printed no JSON: ${message(error)}`, "error");
      return null;
    }
    const unmet = [];
    for (const item of Array.isArray(items) ? items : []) {
      if (!item || typeof item.name !== "string" || item.met === true) continue;
      const name = item.name;
      let result = null;
      if (item.kind === "env") {
        const value = await ctx.ui.input(textOf(item.prompt, `Value for ${name}`), "");
        // The value is one argument to the wrapper, never shell source; the wrapper keeps it in its env file.
        if (value) result = await runWrapper(wrapper, ["setup", "--set", `${name}=${value}`, "--release", releaseId]);
      } else {
        const confirmed = await ctx.ui.confirm(textOf(item.prompt, "Run this check?"), textOf(item.check, ""));
        if (confirmed) result = await runWrapper(wrapper, ["setup", "--run", name, "--release", releaseId]);
      }
      if (result === null) {
        unmet.push(name);
        ctx.ui.notify(`reef: ${name} skipped`, "warning");
      } else if (result.code === 0) {
        ctx.ui.notify(`reef: ${name} ${item.kind === "env" ? "set" : "met"}`, "info");
      } else {
        unmet.push(name);
        ctx.ui.notify(`reef: ${name} not met (exit ${result.code})`, "warning");
      }
    }
    if (unmet.length) ctx.ui.notify(`reef: still to set up: ${unmet.join(", ")} (reef-pi setup)`, "warning");
    return unmet;
  };

  // The install through the wrapper: its update pinned to the release, the setup loop, and the one line that says
  // how to load it (pi's /reload re-runs session_start; only the person can type it). An update the wrapper
  // refused for unmet items runs the setup loop first and then the update again.
  const installRelease = async (releaseId, ctx) => {
    const wrapper = wrapperPath();
    if (!wrapper) {
      ctx.ui.notify(NO_WRAPPER_TEXT, "warning");
      return;
    }
    const update = () => runWrapper(wrapper, ["update", "--release", releaseId]);
    let updated = await update();
    if (updated.code === UPDATE_REFUSED_CODE) {
      await runSetup(wrapper, releaseId, ctx);
      updated = await update();
    } else if (updated.code === 0) {
      await runSetup(wrapper, releaseId, ctx);
    }
    if (updated.code !== 0) {
      const detail = updated.stderr.trim();
      ctx.ui.notify(`reef: reef-pi update failed (exit ${updated.code})${detail ? `: ${detail}` : ""}`, "error");
      return;
    }
    ctx.ui.notify(`Installed release ${releaseId.slice(0, 8)}. Type /reload to load it now.`, "info");
  };

  // The install after a confirmation that says why; a decline names the commands for later. A pending release
  // is promoted first: it is held back from the served head, and installing it is the person saying it may run.
  const offerInstall = async (releaseId, why, ctx, { pending = false } = {}) => {
    if (!ctx.hasUI) return;
    const confirmed = await ctx.ui.confirm(`Install release ${releaseId.slice(0, 8)} now?`, why);
    if (!confirmed) {
      ctx.ui.notify(INSTALL_LATER_TEXT, "info");
      return;
    }
    let install = releaseId;
    if (pending) {
      // A promote republishes the tree as a commit of its own, so the head it mints is what gets installed.
      const head = await promoteRelease(releaseId, ctx);
      if (!head) return; // the failure was notified; nothing was installed
      install = head;
    }
    await installRelease(install, ctx);
  };

  // The promote behind an install of a pending release: the served head moves to it, so later sessions are
  // offered the same version. A failure is notified here and answers null.
  const promoteRelease = async (releaseId, ctx) => {
    let response;
    try {
      response = await fetchWithTimeout(`${serviceUrl}/reef/scenarios/${encodeURIComponent(scenario)}/promote`, {
        method: "POST",
        headers: { ...reefHeaders(), "content-type": "application/json" },
        body: JSON.stringify({ release_id: releaseId }),
      });
    } catch (error) {
      ctx.ui.notify(`reef unreachable at ${serviceUrl}: ${message(error)}`, "error");
      return null;
    }
    if (!response.ok) {
      ctx.ui.notify(`reef refused the promote (HTTP ${response.status}): ${await response.text()}`, "error");
      return null;
    }
    const answer = await response.json();
    return typeof answer.release_id === "string" && answer.release_id ? answer.release_id : null;
  };

  // The install offered the moment a step settles, so a win reaches the session that asked for it without the
  // person going looking. Only a step that produced a tree has one to offer; the rest end at their report.
  const offerSettledInstall = async (step, rows, ctx) => {
    const row = rows[step];
    const selectionResult = resultOf(row, rows);
    if (selectionResult !== "selected" && selectionResult !== "pending") return;
    const why =
      selectionResult === "pending"
        ? `This release changes an extension, which runs in pi with your privileges. Read ${stepPageLink(step)} first.`
        : `Read the change first: ${stepPageLink(step)}`;
    await offerInstall(String(row.release_id), why, ctx, { pending: selectionResult === "pending" });
  };

  // The watch: one at a time, so a second filing replaces the first; session_shutdown clears it.
  let watch = null;

  const stopWatch = (ctx) => {
    if (!watch) return;
    clearInterval(watch.timer);
    if (watch.spinner) clearInterval(watch.spinner);
    watch = null;
    ctx.ui.setStatus("reef", undefined);
    ctx.ui.setWidget(WIDGET_KEY, undefined);
  };

  // What the panel and the widget both read: the phase in the person's words, and how long the step has run.
  const progressLines = (mine) => {
    const phase = PHASE_WORDS[mine.state] || mine.state;
    const since = mine.startedAt === null ? "" : ` ${elapsedText(Date.now() - mine.startedAt)}`;
    return { phase, since };
  };

  // What the expanded spinner shows under its first line: the step as it stands now, read from the watch, so
  // looking in costs no request and never blocks the session.
  const watchLines = () => {
    const lines = [`  asked: ${watch.ask}`, `  request: ${watch.recordId.slice(0, 8)}`];
    // The proposer's latest moves, newest first; the page lists the rest.
    for (const line of watch.activity.slice(-ACTIVITY_LINES).reverse()) {
      if (!line || typeof line.text !== "string") continue;
      lines.push(`  ${line.failed ? "x" : "-"} ${String(line.kind || "")}: ${clip(line.text, 100)}`);
    }
    if (watch.episodes !== null) lines.push(`  evaluation episodes: ${watch.episodes}`);
    if (watch.stepRecord) lines.push(`  step record: ${watch.stepRecord}`);
    lines.push(`  full detail: ${requestPageLink(watch.recordId)}`);
    lines.push("  the step runs in the background; your input stays yours");
    return lines;
  };

  // The spinner above the input box: one line while it is closed, the step's detail under it once opened.
  const redraw = (ctx) => {
    if (!watch) return;
    const { phase, since } = progressLines(watch);
    const frame = SPINNER_FRAMES[watch.frame % SPINNER_FRAMES.length];
    const page = link(requestPageLink(watch.recordId), "open the page");
    const head =
      `${frame} reef: ${phase}${since} - ${page}, ${WATCH_SHORTCUT} or /reefine to ` +
      `${watch.expanded ? "close" : "look in"}`;
    ctx.ui.setWidget(WIDGET_KEY, watch.expanded ? [head, ...watchLines()] : [head]);
  };

  const startWatch = (recordId, text, ctx) => {
    stopWatch(ctx);
    const ask = clip(text.trim(), 60);
    const id8 = recordId.slice(0, 8);
    const deadline = Date.now() + WATCH_CAP_MS;
    // startedAt is the first poll that saw a step holding the request; the indicators count from it. state is
    // the service's own phase for the step, "queued" until a step takes the request.
    const mine = {
      timer: null,
      spinner: null,
      polling: false,
      startedAt: null,
      status: null,
      missing: 0,
      state: "queued",
      recordId,
      ask,
      episodes: null,
      stepRecord: null,
      activity: [],
      frame: 0,
      expanded: false,
    };
    // The spinner redraws on its own clock, so the frames turn between polls.
    const draw = () => {
      if (watch !== mine) return;
      mine.frame++;
      redraw(ctx);
    };
    const show = (status) => {
      if (status === mine.status) return; // the footer is redrawn only when its text changes
      mine.status = status;
      ctx.ui.setStatus("reef", status);
    };
    const tick = async () => {
      let rows = [];
      try {
        rows = await releases();
      } catch {
        // A failed read, a timeout included, is one missed poll; the next tick reads again.
      }
      if (watch !== mine) return; // replaced or shut down while the read was out
      const step = rows.findIndex((row) => requestIdOf(row) === recordId);
      if (step >= 0) {
        stopWatch(ctx);
        forgetRequest(recordId);
        deliverReport(step, rows, text, ctx);
        // The install is offered here only between turns: a dialog mid turn would take the person's input away.
        // A busy session keeps the report's commands, and the next session start offers the same release.
        if (ctx.isIdle()) await offerSettledInstall(step, rows, ctx);
        await resumeStored(rows, ctx); // another filed request still waiting takes the watch over
        return;
      }
      if (Date.now() >= deadline) {
        // The request stays stored: the next session start reports the result once the catalog has it.
        stopWatch(ctx);
        ctx.ui.notify(`reef: no result yet for '${ask}'; /versions shows it when it settles`, "warning");
        return;
      }
      // The step's own phase, which the record alone cannot tell: proposing, evaluating, and the episode count.
      let progress;
      try {
        progress = await requestProgress(recordId);
      } catch {
        progress = undefined; // an older service, or a missed poll: the elapsed time still counts
      }
      if (watch !== mine) return;
      if (progress) {
        mine.state = String(progress.state || mine.state);
        mine.episodes = typeof progress.episodes_total === "number" ? progress.episodes_total : null;
        mine.stepRecord = typeof progress.step_record === "string" ? progress.step_record : null;
        // What the proposer has done so far, oldest first; an older service sends none.
        mine.activity = Array.isArray(progress.activity) ? progress.activity : [];
        // The step's own clock beats the watch's: a reconnecting session counts from when the step began.
        if (typeof progress.started_at === "number") mine.startedAt = progress.started_at * 1000;
        else if (progress.state && progress.state !== "queued" && mine.startedAt === null) mine.startedAt = Date.now();
      }
      if (mine.startedAt === null) {
        let record;
        try {
          record = await requestRecord(recordId);
        } catch {
          record = undefined; // a failed record read keeps the indicators as they were; the next tick reads again
        }
        if (watch !== mine) return;
        // Two polls in a row without the record: one 404 can be a record not written yet, two is a reset.
        mine.missing = record === null ? mine.missing + 1 : 0;
        if (mine.missing >= 2) {
          stopWatch(ctx);
          forgetRequest(recordId);
          ctx.ui.notify(goneText(id8), "warning");
          return;
        }
      }
      if (mine.startedAt !== null) {
        show(`reef: step for request ${id8} running for ${elapsedText(Date.now() - mine.startedAt)}`);
      }
      draw();
    };
    const poll = async () => {
      if (mine.polling) return; // a slow read never overlaps the next tick
      mine.polling = true;
      try {
        await tick();
      } finally {
        mine.polling = false;
      }
    };
    mine.timer = setInterval(poll, watchIntervalMs());
    // A headless session exits when its turn ends; the timers must not hold the process open for the result.
    if (typeof mine.timer.unref === "function") mine.timer.unref();
    if (ctx.hasUI) {
      mine.spinner = setInterval(draw, SPINNER_INTERVAL_MS);
      if (typeof mine.spinner.unref === "function") mine.spinner.unref();
    }
    watch = mine;
    show(`reef: request ${id8} queued`);
    if (ctx.hasUI) draw();
  };

  // A filing: the request is stored until its report is delivered, and the watch starts.
  const filed = (recordId, text, ctx) => {
    rememberRequest(recordId, text);
    startWatch(recordId, text, ctx);
  };

  // The stored requests against the catalog, at a session start and after a settle: each with a row is reported
  // and dropped; the newest still running takes the watch, the others wait for it to settle.
  const resumeStored = async (rows, ctx) => {
    let running = null;
    for (const entry of storedRequests()) {
      const step = rows.findIndex((row) => requestIdOf(row) === entry.id);
      if (step >= 0) {
        forgetRequest(entry.id);
        deliverReport(step, rows, entry.text, ctx);
        continue;
      }
      let record;
      try {
        record = await requestRecord(entry.id);
      } catch {
        record = undefined; // unreadable now: keep it stored and watch it
      }
      if (record === null) {
        forgetRequest(entry.id);
        ctx.ui.notify(goneText(entry.id.slice(0, 8)), "warning");
      } else {
        running = entry;
      }
    }
    if (running) startWatch(running.id, running.text, ctx);
  };

  // The background clarification: one at a time, its state read by the widget and the look-in key.
  let clarification = null;

  const stopClarification = (ctx) => {
    if (!clarification) return;
    clarification.controller.abort();
    clearInterval(clarification.spinner);
    clarification = null;
    ctx.ui.setWidget(CLARIFY_WIDGET_KEY, undefined);
  };

  pi.on("session_shutdown", async (_event, ctx) => {
    stopClarification(ctx);
    stopWatch(ctx);
  });

  // What the opened clarification shows under its first line: its latest steps, one line each.
  const clarifyLines = () =>
    clarification.transcript.slice(-8).map((item) => `  ${item.kind}: ${clip(item.text.replace(/\s+/g, " "), 160)}`);

  // The clarification's widget: one line while it is closed, its latest steps under it once opened.
  const redrawClarification = (ctx) => {
    if (!clarification) return;
    const frame = SPINNER_FRAMES[clarification.frame % SPINNER_FRAMES.length];
    const since = elapsedText(Date.now() - clarification.startedAt);
    const head =
      `${frame} reef: clarifying '${clarification.ask}' ${since} - ${clarification.phase} - ` +
      `${WATCH_SHORTCUT} or /reefine to ${clarification.expanded ? "close" : "look in"}`;
    ctx.ui.setWidget(CLARIFY_WIDGET_KEY, clarification.expanded ? [head, ...clarifyLines()] : [head]);
  };

  // Looking in on what runs in the background: the clarification while it runs, else the step's spinner. Both
  // open in place, above the input, and close the same way. pi offers no click target for a widget, so the key
  // they name is how a person opens them.
  pi.registerShortcut(WATCH_SHORTCUT, {
    description: "Look in on the running reef harness request",
    handler: async (ctx) => {
      if (clarification) {
        clarification.expanded = !clarification.expanded;
        redrawClarification(ctx);
        return;
      }
      if (!watch) {
        ctx.ui.notify("reef: no harness request is running", "info");
        return;
      }
      watch.expanded = !watch.expanded;
      redraw(ctx);
    },
  });

  // The person backed out of the clarification: the model hears it as a result, not an error, so the turn ends
  // without a filing.
  const cancelled = () => ({ content: [{ type: "text", text: CANCELLED_TEXT }], details: { cancelled: true } });

  const askUser = async (params, signal, ctx) => {
    if (!ctx.hasUI) return { content: [{ type: "text", text: NO_UI_TEXT }], details: {} };
    const answers = [];
    for (const item of params.questions) {
      // The recommended option leads, marked; the others keep their order. The answer filed is the option alone.
      const recommended = item.options.includes(item.recommended) ? item.recommended : null;
      const options = recommended
        ? [recommended + RECOMMENDED, ...item.options.filter((option) => option !== recommended)]
        : [...item.options];
      // Esc on a question is the person dropping the request, not an unanswered question: the dialogs stop
      // here and nothing is filed. A dialog the turn aborted reads the same way.
      const choice = await ctx.ui.select(item.question, [...options, OTHER, CANCEL], { signal });
      if (choice === undefined || choice === CANCEL) return cancelled();
      let answer = recommended && choice === recommended + RECOMMENDED ? recommended : choice;
      if (choice === OTHER) {
        answer = await ctx.ui.input(item.question, "", { signal });
        // Esc on the free text answer steps back out of the request too, for one meaning of Esc throughout.
        if (answer === undefined) return cancelled();
      }
      answers.push({ question: item.question, answer });
    }
    return { content: [{ type: "text", text: JSON.stringify(answers) }], details: {} };
  };

  const fileClarified = async (params, ctx) => {
    const text = filedText(params.request, params.clarifications);
    const recordId = await fileRequest(text, ctx); // a failure throws: the caller reports the message
    filed(recordId, text, ctx);
    return {
      content: [
        {
          type: "text",
          text:
            `filed request ${recordId}; reef is running the step, which usually takes a few minutes, ` +
            `and will report here when it settles. Watch it here: ${requestPageLink(recordId)}`,
        },
      ],
      details: { recordId },
    };
  };

  pi.registerTool({
    ...ASK_USER_TOOL,
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      const result = await askUser(params, signal, ctx);
      // The notice says the request is gone rather than leaving the person guessing; the background
      // clarification says it in its entry instead.
      if (result.details.cancelled) ctx.ui.notify("reef: request cancelled; nothing was filed", "info");
      return result;
    },
  });

  pi.registerTool({
    ...FILE_REQUEST_TOOL,
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      return fileClarified(params, ctx);
    },
  });

  // The chat's record of a clarification: one line, and the whole clarification once the person expands it with
  // pi's own expand key. It is a custom entry, so the session's model never reads it.
  pi.registerEntryRenderer(CLARIFY_ENTRY_TYPE, (entry, { expanded }, theme) => {
    if (!tui) return undefined;
    const data = entry.data;
    const colour = { filed: "success", cancelled: "muted", unfiled: "warning", failed: "error" }[data.outcome];
    const container = new tui.Container();
    const hint = expanded ? "" : theme.fg("dim", " (ctrl+o to expand the clarification)");
    container.addChild(new tui.Text(theme.fg(colour, `reef-harness: ${data.summary}`) + hint, 1, 0));
    if (!expanded) return container;
    container.addChild(new tui.Text(theme.fg("muted", `asked: ${data.request}`), 1, 0));
    for (const item of data.transcript) {
      container.addChild(new tui.Text(`${theme.fg("accent", `[${item.kind}]`)} ${item.text}`, 1, 0));
    }
    return container;
  });

  // The clarification itself: the session's model called directly with the two tools, so its turns stay out of
  // the session. A filing, a cancel or a reply without a tool call ends it; the chat keeps the entry either way.
  const clarify = async (text, ctx) => {
    const state = {
      ask: clip(text, 60),
      startedAt: Date.now(),
      phase: "thinking it through",
      frame: 0,
      expanded: false,
      transcript: [],
      controller: new AbortController(),
      spinner: null,
    };
    clarification = state;
    state.spinner = setInterval(() => {
      state.frame++;
      redrawClarification(ctx);
    }, SPINNER_INTERVAL_MS);
    if (typeof state.spinner.unref === "function") state.spinner.unref();
    redrawClarification(ctx);
    const note = (kind, noteText) => state.transcript.push({ kind, text: noteText });
    const finish = (outcome, summary) => {
      if (clarification !== state) return; // the session shut down meanwhile
      stopClarification(ctx);
      const seconds = Math.round((Date.now() - state.startedAt) / 1000);
      pi.appendEntry(CLARIFY_ENTRY_TYPE, {
        outcome,
        summary: `${summary} (${seconds}s)`,
        request: text,
        transcript: state.transcript,
      });
    };
    const messages = [
      { role: "user", content: clarifyMessage(text, recentConversation(ctx.sessionManager)), timestamp: Date.now() },
    ];
    const tools = [ASK_USER_TOOL, FILE_REQUEST_TOOL].map(({ name, description, parameters }) => ({
      name,
      description,
      parameters,
    }));
    for (let turn = 0; turn < CLARIFY_MAX_TURNS; turn++) {
      state.phase = "thinking it through";
      let reply;
      try {
        reply = await ctx.modelRegistry.complete(
          ctx.model,
          { systemPrompt: CLARIFY_SYSTEM_PROMPT, messages, tools },
          { signal: state.controller.signal },
        );
      } catch (error) {
        reply = { content: [], stopReason: "error", errorMessage: message(error) };
      }
      if (clarification !== state) return;
      if (reply.stopReason === "error" || reply.stopReason === "aborted") {
        const why = reply.errorMessage || reply.stopReason;
        note("error", why);
        finish("failed", `the clarification failed: ${why}`);
        ctx.ui.notify(`reef: the clarification failed (${why}); file it as is with /reefine --direct`, "error");
        return;
      }
      messages.push(reply);
      const calls = reply.content.filter((part) => part.type === "toolCall");
      for (const part of reply.content) {
        if (part.type === "thinking" && part.thinking.trim()) note("thinking", part.thinking.trim());
        if (part.type === "text" && part.text.trim()) note("reply", part.text.trim());
        if (part.type === "toolCall") note(part.name, JSON.stringify(part.arguments));
      }
      if (!calls.length) {
        const said = textOfContent(reply.content).trim();
        finish("unfiled", "the clarification ended without filing");
        ctx.ui.notify(
          `reef: the clarification ended without filing${said ? `: ${clip(said, 300)}` : ""}; ` +
            "ask again, or file it as is with /reefine --direct",
          "warning",
        );
        return;
      }
      for (const call of calls) {
        const problem = argumentsProblem(call.name, call.arguments);
        let result;
        let failure = problem;
        if (!problem) {
          try {
            if (call.name === ASK_USER_TOOL.name) {
              state.phase = "waiting for your answers";
              redrawClarification(ctx);
              result = await askUser(call.arguments, state.controller.signal, ctx);
            } else {
              state.phase = "filing";
              redrawClarification(ctx);
              result = await fileClarified(call.arguments, ctx);
            }
          } catch (error) {
            failure = message(error);
          }
        }
        if (clarification !== state) return;
        const resultText = failure ? `error: ${failure}` : textOfContent(result.content);
        note("result", resultText);
        if (result && result.details.cancelled) {
          finish("cancelled", "request cancelled; nothing was filed");
          return;
        }
        if (result && result.details.recordId) {
          const recordId = result.details.recordId;
          finish("filed", `filed request ${recordId.slice(0, 8)}; watch it at ${requestPageLink(recordId)}`);
          return;
        }
        messages.push({
          role: "toolResult",
          toolCallId: call.id,
          toolName: call.name,
          content: [{ type: "text", text: resultText }],
          isError: Boolean(failure),
          timestamp: Date.now(),
        });
      }
    }
    finish("failed", `the clarification took more than ${CLARIFY_MAX_TURNS} model calls`);
    ctx.ui.notify("reef: the clarification did not settle; file it as is with /reefine --direct", "error");
  };

  pi.registerCommand("reefine", {
    description: "Ask reef to grow this harness: /reefine [--direct] <what it should do>",
    handler: async (args, ctx) => {
      const words = (args || "").trim();
      const direct = words === "--direct" || words.startsWith("--direct ");
      const text = (direct ? words.slice("--direct".length) : words).trim();
      if (!text) {
        // The way in that needs neither a key the terminal may swallow nor a click it may not offer.
        if (clarification) {
          ctx.ui.notify([`reef: clarifying '${clarification.ask}' - ${clarification.phase}`, ...clarifyLines()].join("\n"), "info");
          return;
        }
        if (watch) {
          const { phase, since } = progressLines(watch);
          ctx.ui.notify([`reef: ${phase}${since}`, ...watchLines()].join("\n"), "info");
          return;
        }
        ctx.ui.notify("Usage: /reefine <what the harness should do>", "warning");
        return;
      }
      if (!installedRelease()) {
        ctx.ui.notify(noReleaseText(), "error");
        return;
      }
      if (!direct && ctx.hasUI) {
        if (clarification) {
          ctx.ui.notify(`reef: still clarifying '${clarification.ask}'; ask again once it is filed`, "warning");
          return;
        }
        if (!ctx.model) {
          ctx.ui.notify("reef: no model to clarify with; pick one with /model, or use /reefine --direct", "error");
          return;
        }
        // Not awaited: the clarification runs beside the session, so the person's input stays theirs.
        clarify(text, ctx);
        return;
      }
      let recordId;
      try {
        recordId = await fileRequest(text, ctx);
      } catch (error) {
        ctx.ui.notify(message(error), "error");
        return;
      }
      ctx.ui.notify(
        `Training request ${recordId} accepted; the step usually takes a few minutes. ` +
          `Watch it here: ${requestPageLink(recordId)}`,
        "info",
      );
      filed(recordId, text, ctx);
    },
  });

  // The served head: the newest row that is neither pending nor a rejected or skipped step, since those publish
  // nothing and carry the head's id. The catalog's own current flag sits on the newest row, whatever it is.
  const headStep = (rows) => {
    for (let index = rows.length - 1; index >= 0; index--) {
      if (!["pending", "rejected", "skipped"].includes(resultOf(rows[index]))) return index;
    }
    return -1;
  };

  // Match the local tree separately from the served head; they can point at different releases.
  const installedStep = (rows) => {
    const installed = installedRelease();
    return installed ? rows.findIndex((row) => row.release_id === installed) : -1;
  };

  const versionHistory = (rows) => {
    if (!rows.length) return "no release on record";
    const installed = installedStep(rows);
    const head = headStep(rows);
    const results = rows.map((row) => resultOf(row, rows));
    const versionWidth = Math.max("Version".length, `v${rows.length - 1}`.length);
    const resultWidth = Math.max("Result".length, ...results.map((result) => result.length));
    const header = `${"Version".padEnd(versionWidth)}  Release   ${"Result".padEnd(resultWidth)}  Status`;
    const lines = [`Harness versions (${rows.length} entries, oldest first)`, "", header, "-".repeat(header.length)];
    for (const [step, row] of rows.entries()) {
      const marks = [];
      if (step === installed) marks.push("installed");
      if (step === head) marks.push("current");
      lines.push(
        `${`v${step}`.padEnd(versionWidth)}  ${String(row.release_id || "").slice(0, 8).padEnd(8)}  ` +
          `${results[step].padEnd(resultWidth)}  ${marks.join(", ") || "-"}`,
      );
      // Requests live below the columns so long or multilingual text cannot shift the table.
      const request = metricsOf(row).training_request;
      const text = request && typeof request.text === "string" ? request.text.replace(/\s+/g, " ").trim() : "";
      if (text) lines.push(`  "${clip(text, 60)}"`);
    }
    lines.push(
      "",
      "installed: running in this tree; current: served by Reef",
      "Details: /versions <version>",
      "Install: /versions <version> install",
    );
    return lines.join("\n");
  };

  // The page in the person's browser. pi opens its own links this way and exposes no opener to an extension, so
  // the launcher is named here per platform. The URL is one argument, never shell source, and a launcher that is
  // missing (a headless host has no xdg-open) leaves the printed URL as the way in.
  const openPage = async (url, ctx) => {
    const [command, args] =
      process.platform === "darwin"
        ? ["open", [url]]
        : process.platform === "win32"
          ? ["rundll32", ["url.dll,FileProtocolHandler", url]]
          : ["xdg-open", [url]];
    try {
      const opened = await pi.exec(command, args);
      if (opened.code !== 0) ctx.ui.notify(`reef: open it yourself: ${url}`, "info");
    } catch {
      ctx.ui.notify(`reef: open it yourself: ${url}`, "info");
    }
  };

  // One line naming what the step is, for the dialog that offers its page: the result and where it sits.
  const stepSummary = (step, rows) => {
    const head = headStep(rows);
    const installed = installedStep(rows);
    const marks = [resultOf(rows[step], rows)];
    if (step === installed) marks.push(step === head ? `${INSTALLED_MARK}, current` : INSTALLED_MARK);
    else if (step === head) marks.push("current");
    return `${rows[step].release_id} (${marks.join(", ")})`;
  };

  pi.registerCommand("versions", {
    description: "List this harness's versions, or open one: /versions [version] [install]",
    handler: async (args, ctx) => {
      const words = (args || "").trim().split(/\s+/).filter(Boolean);
      const install = words[1] === "install";
      // v3 or 3, digits only before Number(): "1e1" and "0x3" are numbers to it and no version to the catalog.
      const usable =
        words.length === 0 || (/^v?\d+$/.test(words[0]) && (words.length === 1 || (install && words.length === 2)));
      if (!usable) {
        ctx.ui.notify("Usage: /versions [version] [install]", "warning");
        return;
      }
      const step = words.length ? Number(words[0].replace(/^v/, "")) : null;
      let rows;
      try {
        rows = await releases();
      } catch (error) {
        ctx.ui.notify(message(error), "error");
        return;
      }
      if (step === null) {
        ctx.ui.notify(versionHistory(rows), "info");
        return;
      }
      const row = rows[step];
      if (!row) {
        ctx.ui.notify(`no v${step}: the catalog holds v0 to v${rows.length - 1}`, "warning");
        return;
      }
      const selectionResult = resultOf(row, rows);
      if (install) {
        // A pending step installs: the confirmation promotes it first. Only a step with no tree of its own refuses.
        if (["rejected", "skipped"].includes(selectionResult)) {
          ctx.ui.notify(
            `v${step} is ${selectionResult} and published no tree; /versions lists the ones that did`,
            "warning",
          );
          return;
        }
        const why =
          selectionResult === "pending"
            ? `This release changes an extension, which runs in pi with your privileges. Read ${stepPageLink(step)} first.`
            : `Read the change first: ${stepPageLink(step)}`;
        await offerInstall(String(row.release_id), why, ctx, { pending: selectionResult === "pending" });
        return;
      }
      // The page holds the design, the review and the numbers, so the command offers it rather than reprinting it.
      const url = stepPageLink(step);
      const summary = stepSummary(step, rows);
      if (!ctx.hasUI) {
        ctx.ui.notify(`Harness v${step}: ${summary}\npage: ${url}`, "info");
        return;
      }
      const read = await ctx.ui.confirm(`Open harness v${step}?`, summary);
      if (!read) {
        ctx.ui.notify(`page: ${url}`, "info");
        return;
      }
      await openPage(url, ctx);
    },
  });

  // What is ready to install and not installed yet: one step names itself; several share the placeholder. The
  // update notice offers the newest of them; this line names the rest, which a person installs by step.
  const reviewLine = (steps) => {
    const install = steps.length === 1 ? `/versions v${steps[0]} install` : "/versions <version> install";
    return `${steps.length} release(s) ready to install: /versions ${steps.map((step) => `v${step}`).join(", ")} (install with ${install})`;
  };

  // Said once per session start with a UI: the two commands exist, what waits for a review, and the result of
  // any request filed before a restart or reported while the person was away.
  pi.on("session_start", async (_event, ctx) => {
    if (!ctx.hasUI) return;
    const lines = ["reef: /reefine <what it should do> asks for a harness change; /versions lists the versions."];
    let rows = [];
    try {
      rows = await releases();
      const waiting = rows.map((row, step) => (resultOf(row, rows) === "pending" ? step : -1)).filter((step) => step >= 0);
      if (waiting.length) lines.push(reviewLine(waiting));
    } catch {
      // The catalog is a courtesy here: the first line stands without it, and a stored request gets the watch.
    }
    ctx.ui.notify(lines.join("\n"), "info");
    resumeStored(rows, ctx);
  });
}
