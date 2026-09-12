// Prompt when the pulled tree is behind the channel head. An update runs only
// after the user explicitly selects it:
// Kept annotation-free on purpose: plain JavaScript in a .ts file, so plain
// node can parse-check it in CI and pi's TS loader accepts it unchanged.
// Interactive sessions offer to run the update or skip before accepting input.
// Headless sessions print the instructions instead. Hermetic episodes
// set PI_OFFLINE and this extension then makes no network calls at all.
// While the head requires setup not checked off, the setup list replaces the update: the install would refuse.
import { readFileSync } from "node:fs";
import { join } from "node:path";

// The manifest's rule (reef.core.requirements.required_by): the union over the chain, newest name wins.
function requiredBy(releases, releaseId) {
  const published = new Map();
  for (const row of releases) {
    if (row && typeof row.release_id === "string" && !published.has(row.release_id)) published.set(row.release_id, row);
  }
  const chain = [];
  const seen = new Set();
  let current = releaseId;
  while (typeof current === "string" && published.has(current) && !seen.has(current)) {
    seen.add(current);
    const row = published.get(current);
    chain.push(row);
    current = row.rollback_target_release_id || row.parent_release_id;
  }
  const merged = new Map();
  for (const row of chain.reverse()) {
    const requires = ((row.metrics || {}).training_request || {}).requires;
    if (!Array.isArray(requires)) continue;
    for (const item of requires) {
      if (item && typeof item.name === "string") merged.set(item.name, item);
    }
  }
  return [...merged.values()];
}

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
    // install root; a directly-run tree falls back to the release file beside it.
    const destDir = process.env.REEF_HARNESS_DEST || join(agentDir, "..");
    let releaseInfo;
    try {
      // harness_pull and the install script write the release file at the tree root.
      releaseInfo = JSON.parse(readFileSync(join(destDir, ".reef-harness-release"), "utf8"));
    } catch {
      return; // no release file: this tree did not come through the channel
    }
    if (!releaseInfo || typeof releaseInfo !== "object") return; // a release file that is not a record pins nothing
    const pinned = releaseInfo.release_id;
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
    if (!Array.isArray(releases)) return;
    // A release held for review is served to no session, so it is never the head this offers.
    const head = [...releases].reverse().find((row) => row && !row.pending);
    if (!head || head.release_id === pinned) return;
    const pinnedRow = releases.find((row) => row && row.release_id === pinned);
    // A trial install of a pending release is the person's choice: no offer until a promote republishes it.
    if (pinnedRow && pinnedRow.pending && !releases.some((row) => row && row.rollback_target_release_id === pinned)) return;

    const checkedOff = new Map();
    for (const item of Array.isArray(releaseInfo.setup) ? releaseInfo.setup : []) {
      if (item && typeof item.name === "string") checkedOff.set(item.name, item);
    }
    // A check off records the check it stood for; one without it (an older release file) counts by name.
    const met = (item) => {
      const record = checkedOff.get(item.name);
      return record !== undefined && (!("check" in record) || (record.check ?? null) === (item.check ?? null));
    };
    const unmet = requiredBy(releases, head.release_id).filter((item) => !met(item));
    if (unmet.length > 0) {
      const list = unmet.map((item) => `  ${item.name} (${item.kind})${item.check ? `: ${item.check}` : ""}`).join("\n");
      const message =
        `Reef harness update available (${head.release_id}), but it requires setup first:\n${list}\n` +
        "Run reef-pi setup, then start reef-pi again.";
      if (ctx.hasUI) ctx.ui.notify(message, "warning");
      else console.error(message);
      return;
    }

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
