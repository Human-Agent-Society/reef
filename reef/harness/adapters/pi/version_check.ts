// Prompt when the pulled tree is behind the channel head. An update runs only
// after the user explicitly selects it:
// Kept annotation-free on purpose: plain JavaScript in a .ts file, so plain
// node can parse-check it in CI and pi's TS loader accepts it unchanged.
// Interactive sessions offer to run the update or skip before accepting input.
// Headless sessions print the instructions instead. Hermetic episodes
// set PI_OFFLINE and this extension then makes no network calls at all.
import { readFileSync } from "node:fs";
import { join } from "node:path";

export default function versionCheck(pi) {
  let checked = false;

  pi.on("session_start", async (_event, ctx) => {
    if (checked || process.env.PI_OFFLINE) return; // hermetic episodes stay silent
    checked = true;
    const agentDir = process.env.PI_CODING_AGENT_DIR;
    const serviceUrl = process.env.REEF_SERVICE_URL;
    const scenario = process.env.REEF_SCENARIO;
    if (!agentDir || !serviceUrl || !scenario) return;
    // The wrapper relocates the agent into a temp copy and exports the true
    // install root; a directly-run tree falls back to the sidecar beside it.
    const destDir = process.env.REEF_HARNESS_DEST || join(agentDir, "..");
    let pinned;
    try {
      // harness_pull and the install script write the sidecar at the tree root.
      pinned = JSON.parse(readFileSync(join(destDir, ".reef-harness-release"), "utf8")).release_id;
    } catch {
      return; // no sidecar: this tree did not come through the channel
    }
    let response;
    const token = process.env.REEF_TOKEN;
    try {
      response = await fetch(`${serviceUrl}/reef/harness/releases`, {
        headers: {
          "x-reef-scenario": scenario,
          ...(token ? { authorization: `Bearer ${token}` } : {}),
        },
      });
    } catch {
      return; // the notice must never break the harness
    }
    if (!response.ok) return;
    const { releases } = await response.json();
    const head = releases[releases.length - 1];
    if (!head || head.release_id === pinned) return;

    const instruction =
      `curl -fsS -H 'x-reef-scenario: ${scenario}' ` +
      (token ? '-H "Authorization: Bearer $REEF_TOKEN" ' : "") +
      // The installer takes the destination as its first argument; without
      // it a reinstall lands at ./reef-harness relative to the agent's cwd.
      `'${serviceUrl}/reef/harness/install?adapter=pi' | bash -s -- '${destDir}'`;
    const updateOption = `Update with ${instruction}`;
    const title =
      "Reef harness update available\n\n" +
      `Current: ${pinned}\n` +
      `Latest:  ${head.release_id}`;

    if (!ctx.hasUI) {
      console.error(`${title}\n\nUpdate with:\n  ${instruction}`);
      return;
    }

    const choice = await ctx.ui.select(title, [updateOption, "Skip"]);
    if (choice !== updateOption) return;

    ctx.ui.notify("Updating Reef harness...", "info");
    let result;
    try {
      // Values travel as positional arguments rather than shell source. The
      // downloaded installer is the only content deliberately executed.
      result = await pi.exec("bash", [
        "-c",
        'set -o pipefail\nargs=(-fsS -H "x-reef-scenario: $1")\n' +
          'if [[ -n "$3" ]]; then args+=(-H "Authorization: Bearer $3"); fi\n' +
          'curl "${args[@]}" "$2/reef/harness/install?adapter=pi" | bash -s -- "$4"',
        "reef-harness-update",
        scenario,
        serviceUrl,
        token || "",
        destDir,
      ]);
    } catch (error) {
      ctx.ui.notify(`Reef harness update failed: ${error instanceof Error ? error.message : String(error)}`, "error");
      return;
    }
    if (result.code !== 0) {
      const detail = result.stderr.trim();
      ctx.ui.notify(`Reef harness update failed${detail ? `:\n${detail}` : "."}`, "error");
      return;
    }
    ctx.ui.notify("Reef harness updated. Restart reef-pi to load it.", "info");
  });
}
