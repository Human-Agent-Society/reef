// Harness requests: the /reef-harness command for reef-pi. The person asks in plain
// words; the request goes to reef with this session's id and the installed
// release through native manual training; the service proposer writes the
// change without requiring inference receipts or a feedback report. Nothing
// here writes a mutation. Kept free of annotations on purpose: plain JavaScript
// in a .ts file, so plain node can parse it in CI and pi's TS loader
// accepts it unchanged. Gate episodes set PI_OFFLINE and this extension then
// registers nothing, so the gate never sees the command.
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
}
