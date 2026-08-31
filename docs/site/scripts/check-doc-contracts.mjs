import { readFileSync, readdirSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const docsRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const repoRoot = resolve(docsRoot, "..");

function read(path) {
  return readFileSync(resolve(repoRoot, path), "utf8");
}

const configSource = read("recipes/basic/local-sglang.yaml");
const routesDirectory = resolve(repoRoot, "reef/service/routes");
const homeSource = read("docs/site/app/page.tsx");
const wireGuide = read("docs/reference/http-api.rst");
const rootReadme = read("README.md");

function readRouteFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      return readRouteFiles(path);
    }
    if (entry.isFile() && entry.name.endsWith(".py")) {
      return [path];
    }
    return [];
  });
}

const documentationExtensions = new Set([".md", ".mdx", ".rst", ".tsx", ".svg"]);
const sourceExtensions = new Set([".py"]);
const terminologyExcludedDirectories = new Set([
  ".next",
  ".vercel",
  "node_modules",
  "out",
  "public",
  "logo-concepts",
  "tttd",
]);

function readTerminologyFiles(directory, extensions) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      return terminologyExcludedDirectories.has(entry.name)
        ? []
        : readTerminologyFiles(path, extensions);
    }
    if (!entry.isFile()) return [];
    const extension = entry.name.slice(entry.name.lastIndexOf("."));
    return extensions.has(extension) ? [path] : [];
  });
}

const routeFiles = readRouteFiles(routesDirectory);
const routeSource = routeFiles.map(read).join("\n");

const reefConfig = configSource.match(/^reef:\s*\n((?:[ \t]+.*\n?)*)/m)?.[1];
const port = reefConfig?.match(/^[ \t]+port:\s*["']?(\d+)/m)?.[1];
if (!port) {
  throw new Error("Could not derive the Reef port from recipes/basic/local-sglang.yaml");
}

const routes = [...routeSource.matchAll(/app\.router\.add_(get|post)\("([^"]+)"/g)].map(
  ([, method, path]) => ({ method: method.toUpperCase(), path }),
);
if (!routes.length) throw new Error("Could not derive aiohttp routes from reef/service/routes");

const health = routes.find(({ method, path }) => method === "GET" && path.includes("health"))?.path;
if (!health) throw new Error("Could not derive the Reef health route");

const failures = [];
if (!homeSource.includes("reef serve -c recipes/basic/local-sglang.yaml")) {
  failures.push("docs/site/app/page.tsx must start Reef with the bundled config");
}
if (!homeSource.includes(`localhost:${port}${health}`)) {
  failures.push(`docs/site/app/page.tsx must use localhost:${port}${health}`);
}
if (!rootReadme.includes(`127.0.0.1:${port}${health}`)) {
  failures.push(`README.md must use 127.0.0.1:${port}${health}`);
}
for (const { method, path } of routes) {
  if (!wireGuide.includes(`\`${method} ${path}\``)) {
    failures.push(`docs/reference/http-api.rst is missing the ${method} ${path} route`);
  }
}

const terminologyFiles = [
  resolve(repoRoot, "README.md"),
  ...readTerminologyFiles(resolve(repoRoot, "docs"), documentationExtensions),
  ...readTerminologyFiles(resolve(repoRoot, "recipes"), documentationExtensions),
  ...readTerminologyFiles(resolve(repoRoot, "reef"), documentationExtensions),
];
const droppedConcepts = [
  ["evidence", /\bevidence\b/i],
  ["lineage", /\blineage\b/i],
  ["attribution", /\battribution\b/i],
  ["execution runtime", /\bexecution runtime\b/i],
  ["Track A / Track B", /\bTrack [AB]\b/i],
  ["scored records", /\bscored records\b/i],
  // Absolute serving claims — the copy says "continues throughout" /
  // "keeps serving", never a "never" (diagrams included).
  ["never stops", /\bnever stops\b/i],
  ["never pauses", /\bnever pauses\b/i],
];
for (const path of terminologyFiles) {
  const source = readFileSync(path, "utf8");
  for (const [name, pattern] of droppedConcepts) {
    if (pattern.test(source)) {
      failures.push(`${relative(repoRoot, path)} uses dropped concept ${name}`);
    }
  }
}

const droppedIdentifiers = [
  ["execution_runtime", /\bexecution_runtime\b/],
  ["ScenarioLineage", /\bScenarioLineage\b/],
  ["scenario.lineage module", /reef\.scenario\.lineage/],
  ["evidence progress key", /["']evidence["']\s*:/],
  ["scored records", /\bscored records\b/i],
  // Records vocabulary unification (AgentData* -> AgentRecord/RecordStore/...).
  ["AgentData", /\bAgentData\b/],
  ["AgentDataStore", /\bAgentDataStore\b/],
  ["AgentDataCodec", /\bAgentDataCodec\b/],
  ["AgentDataConflict", /\bAgentDataConflict\b/],
  // The codec was inlined into RecordStore as private methods; the class
  // exists under neither the old nor the renamed name.
  ["RecordCodec", /\bRecordCodec\b/],
  // Earlier surface-runtime spellings stay removed.
  ["SurfaceRuntime", /\bSurfaceRuntime\b/],
  ["TrainingSurfaceRuntime", /\bTrainingSurfaceRuntime\b/],
  // The old pairing interlock base's candidate type; the reported-feedback
  // processor replaced it (recipe processor class names survive as adapters).
  ["PairedCandidate", /\bPairedCandidate\b/],
  // The online_grpo example arm is gone. This scan covers reef/**.py only, so the
  // identifier stays mentionable in docs and benchmark history notes.
  ["online_grpo", /\bonline_grpo\b/],
];
for (const path of readTerminologyFiles(resolve(repoRoot, "reef"), sourceExtensions)) {
  const source = readFileSync(path, "utf8");
  for (const [name, pattern] of droppedIdentifiers) {
    if (pattern.test(source)) {
      failures.push(`${relative(repoRoot, path)} uses dropped identifier ${name}`);
    }
  }
}

const glossarySource = read("docs/reference/glossary.rst");
const glossaryTerms = [
  "Scenario",
  "Receipt",
  "Report",
  "Feedback",
  "Recipe",
  "Recipe reference",
  "Loss family",
  "Preparer",
  "Release chain",
  "Artifact",
  "Surface",
  "Harness",
  "Runtime",
  "Skill",
  "OpenClaw-RL",
  "SAO",
  "TTT-Discover",
];
for (const term of glossaryTerms) {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  if (!new RegExp(`(?:^|\\n)${escaped}\\n-+\\n`).test(glossarySource)) {
    failures.push(`docs/reference/glossary.rst must define ${term} as a section heading`);
  }
}

if (failures.length) {
  console.error(`Documentation contract failures:\n${failures.join("\n")}`);
  process.exitCode = 1;
} else {
  console.log(`Docs match the bundled Reef port ${port} and ${routes.length} Reef routes.`);
}
