// Harness requests: the /reef-harness and /reef-versions commands and the
// reef_ask_user and reef_file_request tools for reef-pi. The person asks in
// plain words. With a UI the session model first thinks the request through,
// asks what is unclear (reef_ask_user) and files it (reef_file_request); with
// --direct, or headless, the command files it as is. A filed request goes to
// reef with this session's id and the installed release through native manual
// training, and every filing answers with a link to the request's page. The
// service proposer writes the change, and a watch here polls the catalog, shows
// in the footer whether the request is queued or its step is running and for
// how long, and reports the step's result in the session as a custom message
// the chat keeps, with why the proposer produced nothing when it did. The
// filed requests not yet reported are kept beside the release file, so a
// restarted pi reports their results at its next session start. /reef-versions
// lists the release chain with each step's result and request, prints a
// step's page link and, for a pending release, the promote action and a trial
// install, and runs the promote after a confirmation. A result only reports
// the result and the commands to act on it, leaving the user's input free.
// /reef-versions <step> install starts the install and setup flow on demand;
// promote also offers the install of the head it creates. Nothing here writes
// a mutation. Kept free of annotations on purpose: plain JavaScript in a .ts file, so plain
// node can parse it in CI and pi's TS loader accepts it unchanged. Evaluation
// episodes set PI_OFFLINE and this extension then registers nothing, so the
// evaluation never sees the commands or the tools.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// The release file the install script and harness_pull write at the tree root.
const RELEASE_FILE = ".reef-harness-release";
// The wrapper the install runs through: run_agent exports its path, and a tree run directly has it beside the
// release file; without either the person hears the commands instead.
const WRAPPER_NAME = "reef-pi";
const INSTALL_LATER_TEXT = "reef: install it later with reef-pi update, then reef-pi setup";
const NO_WRAPPER_TEXT = "reef: no reef-pi wrapper found; install it with reef-pi update, then reef-pi setup";
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
// Every request to reef gives up after this: a hung connection must not stall a command or the watch's ticks.
const FETCH_TIMEOUT_MS = 10000;
// The custom message type the report is appended to the session as; pi renders plain text content itself.
const REPORT_MESSAGE_TYPE = "reef-harness";
// The service caps a request's text; the filed text stays within it.
const REQUEST_MAX_CHARS = 4000;
// The choice under every question that opens a free text answer.
const OTHER = "Other (type an answer)";
const NO_UI_TEXT = "no UI in this session: proceed with your best assumptions and list them in the request";

// Tool parameters as plain JSON schema: pi compiles them with typebox, which reads JSON schema as is, so the
// extension needs no import beyond node.
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

// What the session model does with a request before it files it: the request rides as data in a fence.
function clarifyMessage(text) {
  return [
    "The user asked for this harness change:",
    "",
    "```",
    text,
    "```",
    "",
    "Before filing it with reef_file_request, decide what would be built: when the behavior triggers, what it " +
      "does, what state it keeps and how it learns that state. Ask with reef_ask_user only about a decision that " +
      "changes what gets built and that the request leaves open. Rules for the questions:",
    "- at most 3, one decision per question, worded so the user can answer without knowing how the harness works;",
    "- 2 to 4 options that are concrete, mutually exclusive and cover the likely answers; no two options that " +
      "mean the same thing; the user can always type their own;",
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

export default function requests(pi) {
  if (process.env.PI_OFFLINE) return; // hermetic episodes never see the commands or the tools
  const agentDir = process.env.PI_CODING_AGENT_DIR;
  const serviceUrl = process.env.REEF_SERVICE_URL;
  const scenario = process.env.REEF_SCENARIO;
  if (!agentDir || !serviceUrl || !scenario) return;
  // The wrapper relocates the agent into a temp copy and exports the true
  // install root; a tree run directly falls back to the release file beside it.
  const destDir = process.env.REEF_HARNESS_DEST || join(agentDir, "..");

  const reefHeaders = () => {
    const token = process.env.REEF_TOKEN;
    return { "x-reef-scenario": scenario, ...(token ? { authorization: `Bearer ${token}` } : {}) };
  };

  // A page a browser opens: the query carries what curl sends as headers, the scenario and the token.
  const pageLink = (path) => {
    const token = process.env.REEF_TOKEN;
    const query = `scenario=${encodeURIComponent(scenario)}${token ? `&token=${encodeURIComponent(token)}` : ""}`;
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

  // POST the request with this session and the installed release; the answer names the record the step's
  // catalog row carries. Throws with the message the notice shows.
  const fileRequest = async (text, ctx) => {
    const releaseId = installedRelease();
    if (!releaseId) throw new Error(noReleaseText());
    const body = { text, session: ctx.sessionManager.getSessionId(), release_id: releaseId };
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

  // The request's own record: its compacted_at is null while the request is queued and a time once a step
  // took it, which is when the wait a person sees starts.
  const requestRecord = async (recordId) => {
    const path = `/reef/scenarios/${encodeURIComponent(scenario)}/records/${encodeURIComponent(recordId)}`;
    const response = await fetchWithTimeout(`${serviceUrl}${path}`, { headers: reefHeaders() });
    if (response.status === 404) return null; // the service no longer knows the request: its scenario was reset
    if (!response.ok) throw new Error(`reef refused the record read (HTTP ${response.status})`);
    return await response.json();
  };

  // A request the service no longer knows is dropped and said once, instead of a watch that never settles.
  const goneText = (id8) =>
    `reef: request ${id8} is no longer on the service (its scenario was reset); ask again with /reef-harness`;

  // A promoted row stays pending in the catalog; the promote is a later row naming it, so with the rows given
  // the pending row reads "promoted at step N".
  const resultOf = (row, rows = []) => {
    if (row.pending) {
      const promoted = rows.findIndex(
        (other) => other.operation === "promote" && other.rollback_target_release_id === row.release_id,
      );
      return promoted >= 0 ? `promoted at step ${promoted}` : "pending";
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
    const details = ` Details: /reef-versions ${step}.`;
    if (selectionResult === "selected") {
      return (
        `reef: '${ask}' is published as release ${release}. Install when ready with /reef-versions ${step} install.` +
        details
      );
    }
    if (selectionResult === "pending") {
      return (
        `reef: '${ask}' is ready as release ${release}. This release changes an extension, so it is not ` +
        `installed until you promote it: /reef-versions ${step} promote. Page: ${stepPageLink(step)}`
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
    return `reef: '${ask}' settled as ${selectionResult} (release ${release}); /reef-versions ${step} shows it.`;
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

  // The install after a confirmation that says why; a decline names the commands for later.
  const offerInstall = async (releaseId, why, ctx) => {
    if (!ctx.hasUI) return;
    const confirmed = await ctx.ui.confirm(`Install release ${releaseId.slice(0, 8)} now?`, why);
    if (!confirmed) {
      ctx.ui.notify(INSTALL_LATER_TEXT, "info");
      return;
    }
    await installRelease(releaseId, ctx);
  };

  const promotedText = (step, headId) =>
    `Promoted step ${step}: the head is now ${headId}; the update notice offers it at the next session start.`;

  // The promote of a pending row: a promote republishes the tree as a commit of its own, so the answer names the
  // new head, which is what to install. A failure is notified here and answers null.
  const promoteRelease = async (step, row, ctx) => {
    let response;
    try {
      response = await fetchWithTimeout(`${serviceUrl}/reef/scenarios/${encodeURIComponent(scenario)}/promote`, {
        method: "POST",
        headers: { ...reefHeaders(), "content-type": "application/json" },
        body: JSON.stringify({ release_id: row.release_id }),
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
    ctx.ui.notify(promotedText(step, answer.release_id), "info");
    return typeof answer.release_id === "string" && answer.release_id ? answer.release_id : null;
  };

  // An explicit promote command also offers the install of the head it made.
  const promoteThenInstall = async (step, row, ctx) => {
    const headId = await promoteRelease(step, row, ctx);
    if (headId) await offerInstall(headId, promotedText(step, headId), ctx);
  };

  // The watch: one at a time, so a second filing replaces the first; session_shutdown clears it.
  let watch = null;

  const stopWatch = (ctx) => {
    if (!watch) return;
    clearInterval(watch.timer);
    watch = null;
    ctx.ui.setStatus("reef", undefined);
  };

  const startWatch = (recordId, text, ctx) => {
    stopWatch(ctx);
    const ask = clip(text.trim(), 60);
    const id8 = recordId.slice(0, 8);
    const deadline = Date.now() + WATCH_CAP_MS;
    // startedAt is the first poll that saw a step holding the request; the footer counts from it.
    const mine = { timer: null, polling: false, startedAt: null, status: null, missing: 0 };
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
        // A background result never opens a dialog: the report names the commands for when the person is ready.
        await resumeStored(rows, ctx); // another filed request still waiting takes the watch over
        return;
      }
      if (Date.now() >= deadline) {
        // The request stays stored: the next session start reports the result once the catalog has it.
        stopWatch(ctx);
        ctx.ui.notify(`reef: no result yet for '${ask}'; /reef-versions shows it when it settles`, "warning");
        return;
      }
      if (mine.startedAt === null) {
        let record;
        try {
          record = await requestRecord(recordId);
        } catch {
          record = undefined; // a failed record read keeps the footer as it was; the next tick reads again
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
        if (record && typeof record.compacted_at === "number") mine.startedAt = Date.now();
      }
      if (mine.startedAt !== null) {
        show(`reef: step for request ${id8} running for ${elapsedText(Date.now() - mine.startedAt)}`);
      }
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
    // A headless session exits when its turn ends; the timer must not hold the process open for the result.
    if (typeof mine.timer.unref === "function") mine.timer.unref();
    watch = mine;
    show(`reef: request ${id8} queued`);
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

  pi.on("session_shutdown", async (_event, ctx) => stopWatch(ctx));

  pi.registerTool({
    name: "reef_ask_user",
    label: "Ask the user",
    description:
      "Ask the user up to 4 questions before filing a harness change with reef_file_request, each about one " +
      "decision that changes what gets built, with 2 to 4 concrete options that do not overlap; the user can " +
      "always type an answer of their own. Never ask for a setup value (a phone number, a credential, an " +
      "account, a permission): reef-pi setup collects those after the install.",
    parameters: ASK_USER_PARAMETERS,
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      if (!ctx.hasUI) return { content: [{ type: "text", text: NO_UI_TEXT }], details: {} };
      const answers = [];
      for (const item of params.questions) {
        const choice = await ctx.ui.select(item.question, [...item.options, OTHER]);
        const answer = choice === undefined || choice === OTHER ? await ctx.ui.input(item.question, "") : choice;
        answers.push({ question: item.question, answer: answer === undefined ? "no answer" : answer });
      }
      return { content: [{ type: "text", text: JSON.stringify(answers) }], details: {} };
    },
  });

  pi.registerTool({
    name: "reef_file_request",
    label: "File a harness request",
    description:
      "File a harness change with reef: the user's original request and the answers reef_ask_user collected. " +
      "Reef's service writes the change and reports here when the step settles.",
    parameters: FILE_REQUEST_PARAMETERS,
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const text = filedText(params.request, params.clarifications);
      const recordId = await fileRequest(text, ctx); // a failure throws: the model reads the message
      filed(recordId, text, ctx);
      return {
        content: [
          {
            type: "text",
            text:
              `filed request ${recordId}; reef is running the step, which usually takes one to three minutes, ` +
              `and will report here when it settles. Watch it here: ${requestPageLink(recordId)}`,
          },
        ],
        details: {},
      };
    },
  });

  pi.registerCommand("reef-harness", {
    description: "Ask reef to grow this harness: /reef-harness [--direct] <what it should do>",
    handler: async (args, ctx) => {
      const words = (args || "").trim();
      const direct = words === "--direct" || words.startsWith("--direct ");
      const text = (direct ? words.slice("--direct".length) : words).trim();
      if (!text) {
        ctx.ui.notify("Usage: /reef-harness <what the harness should do>", "warning");
        return;
      }
      if (!installedRelease()) {
        ctx.ui.notify(noReleaseText(), "error");
        return;
      }
      if (!direct && ctx.hasUI) {
        // The session model asks what is unclear, then files through the tool; a busy agent takes it as a follow up.
        pi.sendUserMessage(clarifyMessage(text), ctx.isIdle() ? undefined : { deliverAs: "followUp" });
        ctx.ui.notify("reef: clarifying, then filing", "info");
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
        `Training request ${recordId} accepted; the step usually takes one to three minutes. ` +
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

  const requestText = (row) => {
    const request = metricsOf(row).training_request;
    const text = request && typeof request.text === "string" ? request.text.trim() : "";
    return text ? `"${clip(text, 60)}"` : "";
  };

  const lineOf = (step, rows) =>
    [
      String(step),
      String(rows[step].release_id || "").slice(0, 8),
      resultOf(rows[step], rows),
      step === headStep(rows) ? "current" : "",
      requestText(rows[step]),
    ]
      .filter(Boolean)
      .join("  ");

  // What the proposer planned and what its review left uncovered, when the step recorded them.
  const notesLines = (row) => {
    const notes = metricsOf(row).proposal_notes;
    const design = notes && typeof notes.design === "string" ? notes.design.trim() : "";
    const lines = design ? [`design: ${clip(design, 200)}`] : [];
    const uncovered = uncoveredOf(row);
    if (uncovered.length) lines.push(`not covered: ${uncovered.join("; ")}`);
    return lines;
  };

  // What the step's model calls cost in tokens, when the endpoint reported them: the proposer's and the evaluation's.
  const tokenText = (row) => {
    const metrics = metricsOf(row);
    const over = (side, key) =>
      Object.values(side || {}).reduce((total, agent) => total + (Number(agent && agent[key]) || 0), 0);
    const proposerIn = Number(metrics.proposer_input_tokens) || 0;
    const proposerOut = Number(metrics.proposer_output_tokens) || 0;
    const evaluationIn = over(metrics.candidate_agents, "input_tokens") + over(metrics.current_agents, "input_tokens");
    const evaluationOut = over(metrics.candidate_agents, "output_tokens") + over(metrics.current_agents, "output_tokens");
    if (!proposerIn && !proposerOut && !evaluationIn && !evaluationOut) return "";
    return `tokens: proposer ${proposerIn} in / ${proposerOut} out, evaluation ${evaluationIn} in / ${evaluationOut} out`;
  };

  const pageUrl = (step) => `${serviceUrl}/reef/harness/releases/${step}/page`;
  // The token stays in the environment: the printed command names it as the variable, never its value.
  const curl = () =>
    `curl -fsS -H 'x-reef-scenario: ${scenario}' ` + (process.env.REEF_TOKEN ? '-H "Authorization: Bearer $REEF_TOKEN" ' : "");

  const installLine = (releaseId) =>
    `${curl()}'${serviceUrl}/reef/harness/install?adapter=pi&release_id=${encodeURIComponent(releaseId)}'` +
    ` | bash -s -- '${destDir}'`;

  const stepLines = (step, rows) => {
    const row = rows[step];
    const head = headStep(rows);
    const selectionResult = resultOf(row, rows);
    const lines = [
      `Harness step ${step}: ${row.release_id} (${selectionResult}${step === head ? ", current" : ""})`,
      ...notesLines(row),
      `page: ${stepPageLink(step)}`,
      `read it: ${curl()}'${pageUrl(step)}' > harness-step-${step}.html`,
    ];
    if (selectionResult === "pending") {
      const body = JSON.stringify({ release_id: row.release_id });
      lines.push(
        `promote: ${curl()}-X POST -H 'content-type: application/json' -d '${body}' ` +
          `'${serviceUrl}/reef/scenarios/${encodeURIComponent(scenario)}/promote'`,
        `or from here: /reef-versions ${step} promote`,
        `trial install (replaces the tree at ${destDir}): ${installLine(row.release_id)}`,
      );
      if (head >= 0) lines.push(`back to the head: ${installLine(rows[head].release_id)}`);
    }
    const tokens = tokenText(row);
    if (tokens) lines.push(tokens);
    return lines;
  };

  pi.registerCommand("reef-versions", {
    description: "List this harness's versions, or show one: /reef-versions [step] [promote|install]",
    handler: async (args, ctx) => {
      const words = (args || "").trim().split(/\s+/).filter(Boolean);
      const promote = words[1] === "promote";
      const install = words[1] === "install";
      // Digits only before Number(): "1e1" and "0x3" are numbers to it and no step to the catalog.
      const usable =
        words.length === 0 ||
        (/^\d+$/.test(words[0]) && (words.length === 1 || ((promote || install) && words.length === 2)));
      if (!usable) {
        ctx.ui.notify("Usage: /reef-versions [step] [promote|install]", "warning");
        return;
      }
      const step = words.length ? Number(words[0]) : null;
      let rows;
      try {
        rows = await releases();
      } catch (error) {
        ctx.ui.notify(message(error), "error");
        return;
      }
      if (step === null) {
        ctx.ui.notify(rows.length ? rows.map((_, index) => lineOf(index, rows)).join("\n") : "no release on record", "info");
        return;
      }
      const row = rows[step];
      if (!row) {
        ctx.ui.notify(`no step ${step}: the catalog holds steps 0 to ${rows.length - 1}`, "warning");
        return;
      }
      const selectionResult = resultOf(row, rows);
      if (install) {
        if (["pending", "rejected", "skipped"].includes(selectionResult) || selectionResult.startsWith("promoted")) {
          ctx.ui.notify(`step ${step} is ${selectionResult}; choose a published step to install with /reef-versions`, "warning");
          return;
        }
        await offerInstall(String(row.release_id), `Read the change first: ${stepPageLink(step)}`, ctx);
        return;
      }
      if (!promote) {
        ctx.ui.notify(stepLines(step, rows).join("\n"), "info");
        return;
      }
      if (selectionResult.startsWith("promoted")) {
        ctx.ui.notify(`step ${step} is already ${selectionResult}; nothing to promote`, "warning");
        return;
      }
      if (selectionResult !== "pending") {
        ctx.ui.notify(`step ${step} is not pending (${selectionResult}); nothing to promote`, "warning");
        return;
      }
      const confirmed = await ctx.ui.confirm(
        `Promote harness step ${step}?`,
        `Release ${row.release_id} then serves every session that installs the head. Read ${stepPageLink(step)} first.`,
      );
      if (!confirmed) {
        ctx.ui.notify(`step ${step} not promoted`, "info");
        return;
      }
      await promoteThenInstall(step, row, ctx);
    },
  });

  // How to see and promote what waits for a review: one step names itself; several share the placeholder.
  const reviewLine = (steps) => {
    const promote = steps.length === 1 ? `/reef-versions ${steps[0]} promote` : "/reef-versions <step> promote";
    return `${steps.length} release(s) await your review: /reef-versions ${steps.join(", ")} (promote with ${promote})`;
  };

  // Said once per session start with a UI: the two commands exist, what waits for a review, and the result of
  // any request filed before a restart or reported while the person was away.
  pi.on("session_start", async (_event, ctx) => {
    if (!ctx.hasUI) return;
    const lines = ["reef: /reef-harness <what it should do> asks for a harness change; /reef-versions lists the versions."];
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
