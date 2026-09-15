// Prompt when the pulled tree is behind the channel head. An update runs only
// after the user explicitly selects it:
// Kept annotation-free on purpose: plain JavaScript in a .ts file, so plain
// node can parse-check it in CI and pi's TS loader accepts it unchanged.
// Interactive sessions offer to run the update or skip before accepting input.
// Headless sessions print the instructions instead. Hermetic episodes
// set PI_OFFLINE and this extension then makes no network calls at all.
// While the head requires setup not checked off, an interactive session offers to set it up here through the
// reef-pi wrapper (each item asked once, the answer stored by the wrapper), then offers the update; without a
// wrapper on disk, or headless, the setup list replaces the update: the install would refuse. The update itself
// runs through the wrapper when one is on disk and through the install pipeline otherwise.
// Before that, an env variable the installed release requires and this shell lacks gets one warning line.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

// The wrapper the setup and the update run through: run_agent exports its path, and a tree run directly has it
// beside the release file.
function wrapperPath(destDir) {
  const exported = process.env.REEF_HARNESS_WRAPPER;
  if (exported && existsSync(exported)) return exported;
  const beside = join(destDir, "reef-pi");
  return existsSync(beside) ? beside : null;
}

// One wrapper call; a wrapper that could not be started reads as a failed one.
async function runWrapper(pi, wrapper, args) {
  try {
    return await pi.exec(wrapper, args);
  } catch (error) {
    return { stdout: "", stderr: error instanceof Error ? error.message : String(error), code: 1, killed: false };
  }
}

function textOf(value, fallback) {
  return typeof value === "string" && value.trim() ? value : fallback;
}

// The setup loop: what the release still needs from the person, asked here once and stored by the wrapper (an env
// value in its env file, a check off for a check that passed). A declined item stays unmet and is named at the
// end. Returns the names left unmet, or null when the wrapper could not list the items.
async function runSetup(pi, wrapper, releaseId, ctx) {
  const listed = await runWrapper(pi, wrapper, ["setup", "--json", "--release", releaseId]);
  if (listed.code !== 0) {
    ctx.ui.notify(listed.stderr.trim() || `reef: reef-pi setup --json exited ${listed.code}`, "error");
    return null;
  }
  let items;
  try {
    items = JSON.parse(listed.stdout).items;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    ctx.ui.notify(`reef: reef-pi setup --json printed no JSON: ${detail}`, "error");
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
      if (value) result = await runWrapper(pi, wrapper, ["setup", "--set", `${name}=${value}`, "--release", releaseId]);
    } else {
      const confirmed = await ctx.ui.confirm(textOf(item.prompt, "Run this check?"), textOf(item.check, ""));
      if (confirmed) result = await runWrapper(pi, wrapper, ["setup", "--run", name, "--release", releaseId]);
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
}

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
    // What the installed release needs from this shell, one line per unset variable: a check off records that
    // the variable was set once, and says nothing about the shell that started this session.
    // A release held for review gets its notice from the requests extension's session-start line, which names
    // the command that promotes it.
    for (const item of requiredBy(releases, pinned)) {
      if (item.kind !== "env") continue;
      const variable = typeof item.check === "string" && item.check ? item.check : item.name;
      if (process.env[variable]) continue;
      const warning = `reef: ${variable} is not set; the installed harness needs it (reef-pi setup lists it)`;
      if (ctx.hasUI) ctx.ui.notify(warning, "warning");
      else console.error(warning);
    }
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
    const wrapper = wrapperPath(destDir);
    const id8 = String(head.release_id || "").slice(0, 8);
    if (unmet.length > 0) {
      const list = unmet.map((item) => `  ${item.name} (${item.kind})${item.check ? `: ${item.check}` : ""}`).join("\n");
      const message =
        `Reef harness update available (${head.release_id}), but it requires setup first:\n${list}\n` +
        "Run reef-pi setup, then start reef-pi again.";
      if (!ctx.hasUI) {
        console.error(message);
        return;
      }
      // The setup runs here only through the wrapper, and only once the person said so; else the list, as before.
      const setUp = wrapper ? await ctx.ui.confirm(`Set up release ${id8} now?`, list) : false;
      if (!setUp) {
        ctx.ui.notify(message, "warning");
        return;
      }
      const left = await runSetup(pi, wrapper, head.release_id, ctx);
      if (left === null || left.length > 0) return; // the loop named what is left; the install would refuse
    }

    const instruction =
      `curl -fsS -H 'x-reef-scenario: ${scenario}' ` +
      (token ? '-H "Authorization: Bearer $REEF_TOKEN" ' : "") +
      // The installer takes the destination as its first argument; without
      // it a reinstall lands at ./reef-harness relative to the agent's cwd.
      `'${serviceUrl}/reef/harness/install?adapter=pi' | bash -s -- '${destDir}'`;
    // The option names what runs: the wrapper's update when one is on disk, else the pipeline itself.
    const updateOption = `Update with ${wrapper ? "reef-pi update" : instruction}`;
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
      result = wrapper
        ? await pi.exec(wrapper, ["update"])
        : await pi.exec("bash", [
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
    // pi's /reload re-runs session_start on the installed tree; only the person can type it.
    ctx.ui.notify(`Installed release ${id8}. Type /reload to load it now.`, "info");
  });
}
