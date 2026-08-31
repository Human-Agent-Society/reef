// Notify when the pulled tree is behind the channel head. Never auto-applies:
// Kept annotation-free on purpose: plain JavaScript in a .ts file, so plain
// node can parse-check it in CI and pi's TS loader accepts it unchanged.
// the notice names the head version and the one command install; the user
// reviews the head's gate metrics first. Hermetic episodes set PI_OFFLINE and
// this extension then makes no network calls at all.
import { readFileSync } from "node:fs";
import { join } from "node:path";

export default async function versionCheck() {
  if (process.env.PI_OFFLINE) return; // hermetic episodes stay silent
  const agentDir = process.env.PI_CODING_AGENT_DIR;
  const serviceUrl = process.env.REEF_SERVICE_URL;
  const scenario = process.env.REEF_SCENARIO;
  if (!agentDir || !serviceUrl || !scenario) return;
  let pinned;
  try {
    // harness_pull and the install script write the sidecar at the tree root.
    pinned = JSON.parse(readFileSync(join(agentDir, "..", ".reef-harness-version"), "utf8")).artifact_version;
  } catch {
    return; // no sidecar: this tree did not come through the channel
  }
  let response;
  try {
    const token = process.env.REEF_TOKEN;
    response = await fetch(`${serviceUrl}/reef/harness/versions`, {
      headers: {
        "x-reef-scenario": scenario,
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
    });
  } catch {
    return; // the notice must never break the harness
  }
  if (!response.ok) return;
  const { versions } = await response.json();
  const head = versions[versions.length - 1];
  if (head && head.artifact_version !== pinned) {
    console.error(
      `reef: harness version ${pinned} is behind head ${head.artifact_version}; ` +
        "when you have reviewed its gate metrics, update with:\n" +
        `  curl -H 'x-reef-scenario: ${scenario}' '${serviceUrl}/reef/harness/install?adapter=pi' | bash`,
    );
  }
}
