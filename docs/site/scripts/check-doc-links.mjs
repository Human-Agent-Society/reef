import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import GithubSlugger from "github-slugger";
import { RstToHtmlCompiler } from "rst-compiler";

const docsRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const repoRoot = resolve(docsRoot, "..");
const excludedDirectories = new Set([
  ".agents",
  ".claude",
  ".git",
  ".mypy_cache",
  ".next",
  ".pytest_cache",
  ".reef",
  ".ruff_cache",
  ".venv",
  "historical",
  "node_modules",
  "out",
  "public",
  "roadmap",
  // Upstream submodule docs keep their own conventions and dead links;
  // reef links INTO third_party are still resolved as targets.
  "third_party",
]);

function filesWithExtension(directory, extension) {
  return readdirSync(directory).flatMap((name) => {
    if (excludedDirectories.has(name)) return [];
    const target = join(directory, name);
    if (statSync(target).isDirectory()) return filesWithExtension(target, extension);
    return extname(name) === extension ? [target] : [];
  });
}

function plainHeading(value) {
  return value
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/[`*_~]/g, "")
    .replace(/\s+#+\s*$/, "")
    .trim();
}

// Headings are all this check needs, but a page may use the site's custom
// directives (docs/site/lib/docs.ts). Register them as no-ops so compiling for
// anchors never fails on a directive the bare compiler does not know.
const RAW_TEXT_DIRECTIVES = ["flow", "diagram", "config", "page"];
const PASSTHROUGH_DIRECTIVES = ["code", "code-block", ...RAW_TEXT_DIRECTIVES];

function compileForHeadings(source) {
  const compiler = new RstToHtmlCompiler();
  compiler.usePlugin({
    onBeforeParse(parserOptions) {
      parserOptions.directivesWithRawText.push(...RAW_TEXT_DIRECTIVES);
    },
  });
  compiler.useDirectiveGenerator({
    directives: PASSTHROUGH_DIRECTIVES,
    generate(generatorState) {
      generatorState.writeLine("");
    },
  });
  compiler.useDirectiveGenerator({
    directives: ["steps"],
    generate(generatorState, node) {
      generatorState.visitNodes(node.children);
    },
  });
  return compiler.compile(source).body;
}

const headingCache = new Map();
function documentHeadings(file) {
  const cached = headingCache.get(file);
  if (cached) return cached;

  const headings = new Set();
  if (extname(file) === ".rst") {
    const html = compileForHeadings(readFileSync(file, "utf8"));
    for (const match of html.matchAll(/<h[1-6] id="([^"]+)">/g)) headings.add(match[1]);
  } else {
    const slugger = new GithubSlugger();
    let inCodeBlock = false;
    for (const line of readFileSync(file, "utf8").split("\n")) {
      if (line.trimStart().startsWith("```")) {
        inCodeBlock = !inCodeBlock;
        continue;
      }
      if (inCodeBlock) continue;
      const match = line.match(/^#{1,6}\s+(.+)$/);
      if (match) headings.add(slugger.slug(plainHeading(match[1])));
    }
  }
  headingCache.set(file, headings);
  return headings;
}

const failures = [];
function validateLink(file, href) {
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("/")) return;

  const hashIndex = href.indexOf("#");
  const rawPath = hashIndex < 0 ? href : href.slice(0, hashIndex);
  const rawFragment = hashIndex < 0 ? "" : href.slice(hashIndex + 1);
  const pathname = decodeURIComponent(rawPath);
  const target = pathname ? resolve(dirname(file), pathname) : file;
  const display = `${relative(repoRoot, file)} -> ${href}`;

  if (!existsSync(target)) {
    failures.push(`missing target: ${display}`);
    return;
  }
  if (rawFragment && statSync(target).isFile() && [".md", ".rst"].includes(extname(target))) {
    const fragment = decodeURIComponent(rawFragment);
    if (!documentHeadings(target).has(fragment)) failures.push(`missing anchor: ${display}`);
  }
}

const markdownFiles = filesWithExtension(repoRoot, ".md");
for (const file of markdownFiles) {
  const content = readFileSync(file, "utf8");
  const linkPattern = /\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\s*\)/g;
  for (const match of content.matchAll(linkPattern)) validateLink(file, match[1] ?? match[2]);
}

const rstFiles = filesWithExtension(docsRoot, ".rst");
for (const file of rstFiles) {
  const content = readFileSync(file, "utf8");
  for (const match of content.matchAll(/<([^<>\s]+)>`__?/g)) validateLink(file, match[1]);
  for (const match of content.matchAll(/^\.\. (?:image|figure)::\s+(\S+)/gm)) validateLink(file, match[1]);
}

function docSlug(file) {
  return relative(docsRoot, file).replaceAll("\\", "/").slice(0, -".rst".length);
}

const siteSlugs = new Set(rstFiles.map(docSlug));
// /docs is served by app/docs/page.tsx, which redirects into the reading order.
if (existsSync(join(docsRoot, "site/app/docs/page.tsx"))) siteSlugs.add("");
const siteSources = [
  ...filesWithExtension(join(docsRoot, "site/app"), ".tsx"),
  ...filesWithExtension(join(docsRoot, "site/components"), ".tsx"),
];
for (const file of siteSources) {
  const content = readFileSync(file, "utf8");
  for (const match of content.matchAll(/href=["'](\/docs(?:\/[^"'#?]*)?)["']/g)) {
    const route = match[1].replace(/\/$/, "");
    const slug = route === "/docs" ? "" : route.slice("/docs/".length);
    if (!siteSlugs.has(slug)) {
      failures.push(`missing site route: ${relative(repoRoot, file)} -> ${match[1]}`);
    }
  }
}

if (failures.length) {
  console.error(`Documentation link failures:\n${failures.join("\n")}`);
  process.exitCode = 1;
} else {
  console.log(`Validated ${markdownFiles.length} Markdown files, ${rstFiles.length} RST files, and internal docs routes.`);
}
