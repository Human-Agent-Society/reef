// Harness requests: the /reef-harness and /reef-versions commands for reef-pi. The
// person asks in plain words; the request goes to reef with this session's id
// and the installed release through native manual training; the service
// proposer writes the change without requiring inference receipts or a
// feedback report. /reef-versions lists the release chain with each step's
// verdict and request, prints a step's page and, for a pending release, the
// promote action and a trial install, and runs the promote after a
// confirmation. Nothing here writes a mutation. Kept free of annotations on
// purpose: plain JavaScript in a .ts file, so plain node can parse it in CI
// and pi's TS loader accepts it unchanged. Gate episodes set PI_OFFLINE and
// this extension then registers nothing, so the gate never sees the commands.
import { readFileSync } from "node:fs";
import { join } from "node:path";

// The release sidecar the install script and harness_pull write at the tree root.
const SIDECAR = ".reef-harness-release";

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

export default function requests(pi) {
  if (process.env.PI_OFFLINE) return; // hermetic episodes never see the command
  const agentDir = process.env.PI_CODING_AGENT_DIR;
  const serviceUrl = process.env.REEF_SERVICE_URL;
  const scenario = process.env.REEF_SCENARIO;
  if (!agentDir || !serviceUrl || !scenario) return;
  // The wrapper relocates the agent into a temp copy and exports the true
  // install root; a tree run directly falls back to the sidecar beside it.
  const destDir = process.env.REEF_HARNESS_DEST || join(agentDir, "..");

  const reefHeaders = () => {
    const token = process.env.REEF_TOKEN;
    return { "x-reef-scenario": scenario, ...(token ? { authorization: `Bearer ${token}` } : {}) };
  };

  const installedRelease = () => {
    const sidecar = readJson(join(destDir, SIDECAR));
    return sidecar && typeof sidecar.release_id === "string" && sidecar.release_id ? sidecar.release_id : null;
  };

  pi.registerCommand("reef-harness", {
    description: "Ask reef to grow this harness: /reef-harness <what it should do>",
    handler: async (args, ctx) => {
      const text = (args || "").trim();
      if (!text) {
        ctx.ui.notify("Usage: /reef-harness <what the harness should do>", "warning");
        return;
      }
      const releaseId = installedRelease();
      if (!releaseId) {
        ctx.ui.notify(
          `no ${SIDECAR} sidecar at ${destDir}: this tree did not come through reef's install channel, ` +
            "so a request cannot name the release it runs; nothing was sent",
          "error",
        );
        return;
      }
      const body = { text, session: ctx.sessionManager.getSessionId(), release_id: releaseId };
      let response;
      try {
        // Not under the turn's abort signal: an Esc after the body went out would report a filed request as unreachable.
        response = await fetch(`${serviceUrl}/reef/train`, {
          method: "POST",
          headers: { ...reefHeaders(), "content-type": "application/json" },
          body: JSON.stringify(body),
        });
      } catch (error) {
        ctx.ui.notify(`reef unreachable at ${serviceUrl}: ${message(error)}`, "error");
        return;
      }
      if (!response.ok) {
        ctx.ui.notify(`reef refused the request (HTTP ${response.status}): ${await response.text()}`, "error");
        return;
      }
      const answer = await response.json();
      ctx.ui.notify(`Training request ${answer.agent_record_id} accepted.`, "info");
    },
  });

  // The catalog oldest first; a step is a row's position in it, the creation row being 0, which is the commit
  // step the service keys the page by (a rejected step publishes nothing, so only its position names it).
  const releases = async () => {
    let response;
    try {
      response = await fetch(`${serviceUrl}/reef/harness/releases`, { headers: reefHeaders() });
    } catch (error) {
      throw new Error(`reef unreachable at ${serviceUrl}: ${message(error)}`);
    }
    if (!response.ok) throw new Error(`reef refused the catalog read (HTTP ${response.status}): ${await response.text()}`);
    const rows = (await response.json()).releases;
    return Array.isArray(rows) ? rows : [];
  };

  // A promoted row stays pending in the catalog; the promote is a later row naming it, so with the rows given
  // the pending row reads "promoted at step N".
  const verdictOf = (row, rows = []) => {
    if (row.pending) {
      const promoted = rows.findIndex(
        (other) => other.operation === "promote" && other.rollback_target_release_id === row.release_id,
      );
      return promoted >= 0 ? `promoted at step ${promoted}` : "pending";
    }
    const metrics = row.metrics && typeof row.metrics === "object" ? row.metrics : {};
    if (typeof metrics.selected === "boolean") return metrics.selected ? "selected" : "rejected";
    if (metrics.skipped) return "skipped";
    return String(row.operation || "unknown");
  };

  // The served head: the newest row that is neither pending nor a rejected or skipped step, since those publish
  // nothing and carry the head's id. The catalog's own current flag sits on the newest row, whatever it is.
  const headStep = (rows) => {
    for (let index = rows.length - 1; index >= 0; index--) {
      if (!["pending", "rejected", "skipped"].includes(verdictOf(rows[index]))) return index;
    }
    return -1;
  };

  const requestText = (row) => {
    const request = row.metrics && row.metrics.training_request;
    const text = request && typeof request.text === "string" ? request.text.trim() : "";
    if (!text) return "";
    return `"${text.length > 60 ? `${text.slice(0, 57)}...` : text}"`;
  };

  const lineOf = (step, rows) =>
    [
      String(step),
      String(rows[step].release_id || "").slice(0, 8),
      verdictOf(rows[step], rows),
      step === headStep(rows) ? "current" : "",
      requestText(rows[step]),
    ]
      .filter(Boolean)
      .join("  ");

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
    const verdict = verdictOf(row, rows);
    const lines = [
      `Harness step ${step}: ${row.release_id} (${verdict}${step === head ? ", current" : ""})`,
      `page: ${pageUrl(step)}`,
      `read it: ${curl()}'${pageUrl(step)}' > harness-step-${step}.html`,
    ];
    if (verdict === "pending") {
      const body = JSON.stringify({ release_id: row.release_id });
      lines.push(
        `promote: ${curl()}-X POST -H 'content-type: application/json' -d '${body}' ` +
          `'${serviceUrl}/reef/scenarios/${encodeURIComponent(scenario)}/promote'`,
        `or from here: /reef-versions ${step} promote`,
        `trial install (replaces the tree at ${destDir}): ${installLine(row.release_id)}`,
      );
      if (head >= 0) lines.push(`back to the head: ${installLine(rows[head].release_id)}`);
    }
    return lines;
  };

  pi.registerCommand("reef-versions", {
    description: "List this harness's versions, or show one: /reef-versions [step] [promote]",
    handler: async (args, ctx) => {
      const words = (args || "").trim().split(/\s+/).filter(Boolean);
      const promote = words[1] === "promote";
      // Digits only before Number(): "1e1" and "0x3" are numbers to it and no step to the catalog.
      const usable = words.length === 0 || (/^\d+$/.test(words[0]) && (words.length === 1 || (promote && words.length === 2)));
      if (!usable) {
        ctx.ui.notify("Usage: /reef-versions [step] [promote]", "warning");
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
      if (!promote) {
        ctx.ui.notify(stepLines(step, rows).join("\n"), "info");
        return;
      }
      const verdict = verdictOf(row, rows);
      if (verdict.startsWith("promoted")) {
        ctx.ui.notify(`step ${step} is already ${verdict}; nothing to promote`, "warning");
        return;
      }
      if (verdict !== "pending") {
        ctx.ui.notify(`step ${step} is not pending (${verdict}); nothing to promote`, "warning");
        return;
      }
      const confirmed = await ctx.ui.confirm(
        `Promote harness step ${step}?`,
        `Release ${row.release_id} then serves every session that installs the head. Read ${pageUrl(step)} first.`,
      );
      if (!confirmed) {
        ctx.ui.notify(`step ${step} not promoted`, "info");
        return;
      }
      let response;
      try {
        response = await fetch(`${serviceUrl}/reef/scenarios/${encodeURIComponent(scenario)}/promote`, {
          method: "POST",
          headers: { ...reefHeaders(), "content-type": "application/json" },
          body: JSON.stringify({ release_id: row.release_id }),
        });
      } catch (error) {
        ctx.ui.notify(`reef unreachable at ${serviceUrl}: ${message(error)}`, "error");
        return;
      }
      if (!response.ok) {
        ctx.ui.notify(`reef refused the promote (HTTP ${response.status}): ${await response.text()}`, "error");
        return;
      }
      const answer = await response.json();
      ctx.ui.notify(
        `Promoted step ${step}: the head is now ${answer.release_id}; the update notice offers it at the next session start.`,
        "info",
      );
    },
  });
}
