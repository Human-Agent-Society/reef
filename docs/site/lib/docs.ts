import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import langJson from "@shikijs/langs/json";
import langPython from "@shikijs/langs/python";
import langShellscript from "@shikijs/langs/shellscript";
import langShellsession from "@shikijs/langs/shellsession";
import langYaml from "@shikijs/langs/yaml";
import { RstToHtmlCompiler } from "rst-compiler";
import { createHighlighterCoreSync } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

export type TocItem = {
  id: string;
  title: string;
  level: 2 | 3;
};

export type Doc = {
  slug: string;
  title: string;
  description: string;
  content: string;
  sourcePath: string;
  toc: TocItem[];
};

export type NavItem = {
  title: string;
  description: string;
  href: string;
  slug: string;
};

export type NavGroup = {
  title: string;
  items: NavItem[];
};

const docsRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");

const navigationSources: ReadonlyArray<{ title: string; files: ReadonlyArray<string> }> = [
  {
    // What Reef is, what to install, the loop every later page assumes, and
    // what is underneath it. Architecture sits last because it spends the
    // vocabulary quickstart.rst teaches. The first entry is where /docs lands.
    title: "Getting Started",
    files: [
      "getting-started/intro.rst",
      "getting-started/installation.rst",
      "getting-started/quickstart.rst",
      "getting-started/architecture.rst",
    ],
  },
  {
    // Operating Reef: pick a method, run the lane it belongs to, keep the
    // deployment running, and find out why it did not.
    title: "User Guide",
    files: [
      "user-guide/recipes.rst",
      "user-guide/evolve-your-harness.rst",
      "user-guide/evolve-your-model.rst",
      "user-guide/recipes/sao.rst",
      "user-guide/recipes/tttd.rst",
      "user-guide/recipes/openclawrl.rst",
      "user-guide/recipes/skillclaw.rst",
      "user-guide/recipes/gepa.rst",
      "user-guide/operate.rst",
      "user-guide/troubleshooting.rst",
    ],
  },
  {
    // Writing your own method against Reef's contracts.
    title: "Developer Guide",
    files: [
      "developer-guide/write-a-recipe.rst",
      "developer-guide/write-a-harness-method.rst",
      "developer-guide/harness-adapters.rst",
      "developer-guide/loss-families.rst",
      "developer-guide/surface.rst",
      "developer-guide/processors.rst",
    ],
  },
  {
    title: "CLI Reference",
    files: [
      "reference/cli.rst",
      "reference/configuration.rst",
    ],
  },
  {
    title: "API Reference",
    files: [
      "reference/http-api.rst",
      "reference/python-api.rst",
      "reference/glossary.rst",
    ],
  },
  {
    title: "Contributing",
    files: [
      "contributing/codebase-structure.rst",
      "contributing/adding-components.rst",
      "contributing/development.rst",
      "contributing/testing.rst",
    ],
  },
];

// The site loads only the files listed above, so a page missing from the list
// does not merely go unlinked — it 404s. Adding a page under a section
// directory now fails the build until it is placed in the reading order. Not
// scanned: rfcs/ (design records, read in the repo).
const navigated = new Set<string>(navigationSources.flatMap((group) => group.files));
// Pages live in one directory per navigation section, so a page's path names
// its section and the URL follows the path.
const sectionDirectories = ["getting-started", "user-guide", "user-guide/recipes", "developer-guide", "reference", "contributing"];
const navigableSources = [
  ...readdirSync(docsRoot),
  ...sectionDirectories.flatMap((directory) =>
    readdirSync(join(docsRoot, directory)).map((name) => `${directory}/${name}`),
  ),
].filter((name) => name.endsWith(".rst"));
const unnavigated = navigableSources.filter((name) => !navigated.has(name));
if (unnavigated.length > 0) {
  throw new Error(
    `Documentation pages missing from navigationSources, so they would 404: ${unnavigated.join(", ")}`,
  );
}

// Every page carries its path as its slug, so each one has a route of its own
// and none of them claims /docs. That route redirects to docsIndexHref below.
function slugFromSourcePath(sourcePath: string) {
  return sourcePath.slice(0, -".rst".length);
}

function decodeHtml(value: string) {
  return value
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&apos;", "'")
    .replace(/&#(\d+);/g, (_, value: string) => String.fromCodePoint(Number(value)))
    .replace(/&#x([0-9a-f]+);/gi, (_, value: string) => String.fromCodePoint(Number.parseInt(value, 16)));
}

function plainHtml(value: string) {
  return decodeHtml(value.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
}

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

// Every token color is a var(--syntax-*) reference, so the highlighted HTML is
// theme-neutral: the light and dark values live in globals.css and the site's
// data-theme toggle switches them with no second render. The unscoped entry
// sets the block's background and text to the same variables the plain pre
// styling uses, keeping shiki's inline styles in step with the chrome.
const syntaxTheme = {
  name: "reef",
  settings: [
    { settings: { foreground: "var(--code-text)", background: "var(--code)" } },
    { scope: "comment", settings: { foreground: "var(--syntax-comment)", fontStyle: "italic" } },
    { scope: "string", settings: { foreground: "var(--syntax-string)" } },
    { scope: "string.regexp", settings: { foreground: "var(--syntax-regexp)" } },
    // Keywords and declarations: Python def/class are storage.type, not keyword.
    { scope: ["keyword", "storage", "constant.language"], settings: { foreground: "var(--syntax-keyword)" } },
    // Plain operators (=, +, |) stay in the base ink; word operators read as keywords.
    { scope: "keyword.operator", settings: { foreground: "var(--code-text)" } },
    {
      scope: ["keyword.operator.word", "keyword.operator.logical", "keyword.operator.expression"],
      settings: { foreground: "var(--syntax-keyword)" },
    },
    {
      scope: [
        "entity.name.function",
        "entity.name.class",
        "entity.name.type",
        "entity.name.namespace",
        "meta.function-call.generic",
      ],
      settings: { foreground: "var(--syntax-title)" },
    },
    // Mapping keys: entity.name.tag covers YAML, support.type.property-name JSON.
    // Values keep the string color, so keys and values stay distinct.
    { scope: ["entity.name.tag", "support.type.property-name"], settings: { foreground: "var(--syntax-title)" } },
    { scope: "constant.numeric", settings: { foreground: "var(--syntax-number)" } },
    { scope: ["support.function", "support.class", "support.type"], settings: { foreground: "var(--syntax-built-in)" } },
    { scope: ["variable", "variable.parameter"], settings: { foreground: "var(--syntax-variable)" } },
  ],
};

// The directive generator below runs synchronously, so this uses the sync core
// with the JavaScript regex engine: no WASM to await, and the whole corpus
// highlights in well under a second at build time. The grammars loaded here
// are exactly the languages the docs use; "text" and "mermaid" are absent on
// purpose so those blocks fall through to the escaped-pre path, which keeps
// ASCII diagrams verbatim and preserves the language-mermaid class contract
// that markdown.tsx uses to mount diagrams.
const highlighter = createHighlighterCoreSync({
  themes: [syntaxTheme],
  langs: [langPython, langShellscript, langYaml, langShellsession, langJson],
  engine: createJavaScriptRegexEngine(),
});

function highlightedHtml(code: string, language: string) {
  if (!highlighter.getLoadedLanguages().includes(language)) return undefined;
  const html = highlighter.codeToHtml(code, {
    lang: language,
    theme: "reef",
    transformers: [
      {
        pre(node) {
          // Keep the classes the unhighlighted path emits so styling and any
          // downstream selector on language-X apply to both kinds of block.
          node.properties.class = `${String(node.properties.class ?? "")} code language-${language}`.trim();
        },
      },
    ],
  });
  // One line of HTML: the RST generator re-indents every literal newline it
  // writes, and inside <pre> that indent becomes phantom leading spaces on
  // nested code blocks. As the entity, the newline survives untouched and
  // parses back into real text, so rendering, copying, and consumers that
  // read textContent (the mermaid path) all see the original line breaks.
  return html.replace(/\n/g, "&#10;");
}

// Inline markup inside figure labels and reference rows: the directive bodies
// below are raw text, so the only markup honored is the double-backquote
// literal, converted after HTML-escaping the rest of the line.
function inlineHtml(value: string) {
  return escapeHtml(value).replace(/``([^`]+)``/g, '<code class="literal">$1</code>');
}

// Connector glyphs are fixed-size inline SVG, never authored characters: they
// do not scale with any viewBox, so the ASCII-only source rule and the
// body-size label rule both hold. CSS rotates them when a figure stacks.
function arrowSvg(kind: "next" | "loop") {
  const path =
    kind === "next"
      ? '<path d="M5 12h13"/><path d="m13 7 5 5-5 5"/>'
      : '<polyline points="9 10 4 15 9 20"/><path d="M20 4v7a4 4 0 0 1-4 4H4"/>';
  return (
    `<svg class="fig-arrow fig-arrow-${kind}" width="18" height="18" viewBox="0 0 24 24"` +
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"' +
    ` stroke-linejoin="round" aria-hidden="true">${path}</svg>`
  );
}

// `.. flow::` is the lifecycle figure: one stage per body line as
// `Title :: caption` (caption optional, `Title*` for the emphasis tint), and
// `:loop:` draws the labeled return arrow. Stages are DOM text in flex cards,
// so labels hold their size and the strip reflows to a vertical stack on
// narrow screens instead of scaling down the way generated SVG text does.
function flowHtml(node: { rawBodyText: string; config: { getField(name: string): string | null } | null }) {
  const stages = node.rawBodyText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [head, ...rest] = line.split("::");
      const caption = rest.join("::").trim();
      const emphasized = head.trim().endsWith("*");
      const title = emphasized ? head.trim().slice(0, -1).trim() : head.trim();
      const captionHtml = caption ? `<span class="fig-flow-caption">${inlineHtml(caption)}</span>` : "";
      return (
        `<div class="fig-flow-stage${emphasized ? " fig-emphasis" : ""}">` +
        `<span class="fig-flow-title">${inlineHtml(title)}</span>${captionHtml}</div>`
      );
    });
  const loop = node.config?.getField("loop");
  const loopHtml = loop
    ? `<div class="fig-flow-loop">${arrowSvg("loop")}<span>${inlineHtml(loop)}</span></div>`
    : "";
  return (
    `<figure class="fig fig-flow"><div class="fig-flow-track">${stages.join(arrowSvg("next"))}</div>` +
    `${loopHtml}</figure>`
  );
}

// `.. config::` is reference rows: `key | description` or
// `key | default | description` per body line. Each key gets an id, so rows
// are deep-linkable without spending headings on them.
function configHtml(rawBodyText: string) {
  const rows = rawBodyText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split("|").map((part) => part.trim());
      const [key, fallback, description] =
        parts.length >= 3 ? [parts[0], parts[1], parts.slice(2).join(" | ")] : [parts[0], "", parts[1] ?? ""];
      const id = `cfg-${key.toLowerCase().replace(/[^a-z0-9_.-]+/g, "-")}`;
      const fallbackHtml = fallback ? `<span class="config-default">${inlineHtml(fallback)}</span>` : "";
      return (
        `<div class="config-row" id="${escapeHtml(id)}"><code class="config-key">${inlineHtml(key)}</code>` +
        `<span class="config-desc">${fallbackHtml}${inlineHtml(description)}</span></div>`
      );
    });
  return `<div class="config-card">${rows.join("")}</div>`;
}

// The page brief additionally honors RST's `text <target>`__ link form, so a
// prerequisite can point at the page that satisfies it. The href is emitted
// verbatim; markdown.tsx resolves it against the source path like any link.
function briefHtml(value: string) {
  return inlineHtml(value).replace(/`([^`<]+?) &lt;([^&]+?)&gt;`__/g, '<a href="$2">$1</a>');
}

// `.. page::` is the brief under a page's lede: who the page is for, what to
// have ready, and what the reader has at the end. Three fields, no body; a
// page that carries none of them renders nothing.
function pageBriefHtml(node: { config: { getField(name: string): string | null } | null }) {
  const rows = (
    [
      ["For", "for"],
      ["Before you start", "needs"],
      ["You will have", "outcome"],
    ] as const
  )
    .map(([label, field]) => [label, node.config?.getField(field)?.replace(/\s+/g, " ").trim() ?? ""] as const)
    .filter(([, value]) => value);
  if (!rows.length) return "";
  const cells = rows.map(([label, value]) => `<dt>${label}</dt><dd>${briefHtml(value)}</dd>`).join("");
  return `<dl class="page-brief">${cells}</dl>`;
}

function compileRst(content: string, sourcePath: string) {
  const compiler = new RstToHtmlCompiler();
  // The bodies of these directives are DSL or hand-built markup, not RST, so
  // the parser must hand them over verbatim the way it does for code blocks.
  compiler.usePlugin({
    onBeforeParse(parserOptions) {
      parserOptions.directivesWithRawText.push("flow", "diagram", "config", "page");
    },
  });
  compiler.useDirectiveGenerator({
    directives: ["page"],
    generate(generatorState, node) {
      const html = pageBriefHtml(node);
      if (html) generatorState.writeLine(html);
    },
  });
  compiler.useDirectiveGenerator({
    directives: ["code", "code-block"],
    generate(generatorState, node) {
      const language = /^[a-z0-9_+#.-]+$/i.test(node.initContentText)
        ? node.initContentText.toLowerCase()
        : "text";
      const fallback = escapeHtml(node.rawBodyText).replace(/\n/g, "&#10;");
      generatorState.writeLine(
        highlightedHtml(node.rawBodyText, language) ??
          `<pre class="code language-${escapeHtml(language)}">${fallback}</pre>`,
      );
    },
  });
  compiler.useDirectiveGenerator({
    directives: ["flow"],
    generate(generatorState, node) {
      generatorState.writeLine(flowHtml(node));
    },
  });
  // `.. diagram::` is the escape hatch for genuinely spatial figures: the body
  // is hand-built HTML over the fig-node/fig-group/fig-edge primitives in
  // globals.css, passed through inside the shared figure chrome.
  compiler.useDirectiveGenerator({
    directives: ["diagram"],
    generate(generatorState, node) {
      const caption = node.config?.getField("caption");
      generatorState.writeLine('<figure class="fig fig-diagram">');
      generatorState.writeLine(node.rawBodyText);
      if (caption) generatorState.writeLine(`<figcaption>${inlineHtml(caption)}</figcaption>`);
      generatorState.writeLine("</figure>");
    },
  });
  compiler.useDirectiveGenerator({
    directives: ["config"],
    generate(generatorState, node) {
      generatorState.writeLine(configHtml(node.rawBodyText));
    },
  });
  // `.. steps::` is a walkthrough wrapper around an ordinary RST enumerated
  // list: the body compiles as usual and CSS counters draw the numbered
  // circles and connector line, so code blocks inside keep the shiki path.
  compiler.useDirectiveGenerator({
    directives: ["steps"],
    generate(generatorState, node) {
      generatorState.writeLine('<div class="steps">');
      generatorState.visitNodes(node.children);
      generatorState.writeLine("</div>");
    },
  });

  const output = compiler.compile(content);
  if (compiler.outputErrors.length > 0) {
    throw new Error(`Could not compile ${sourcePath}: ${compiler.outputErrors.join("; ")}`);
  }
  return output.body;
}

function heading(html: string, level: 1 | 2 | 3) {
  const match = html.match(new RegExp(`<h${level} id="([^"]+)">([\\s\\S]*?)</h${level}>`));
  if (!match) return undefined;
  return { id: match[1], title: plainHtml(match[2]) };
}

function tableOfContents(html: string) {
  const items: TocItem[] = [];
  for (const match of html.matchAll(/<h([23]) id="([^"]+)">([\s\S]*?)<\/h\1>/g)) {
    items.push({
      id: match[2],
      title: plainHtml(match[3]),
      level: Number(match[1]) as 2 | 3,
    });
  }
  return items;
}

// Page summaries are independent of the opening paragraph: a walkthrough
// often starts with an introduction that does not describe the whole page.
// Pages without an authored summary continue to use their first paragraph.
const docDescriptions: Partial<Record<string, string>> = {
  "getting-started/intro":
    "Learn how Reef connects inference, feedback, and continual learning to evolve model weights or agent harnesses.",
  "getting-started/installation":
    "Install Reef for harness evolution on a laptop, GPU-based model training, or client-only access to an existing deployment.",
  "getting-started/quickstart":
    "Start Reef locally without a GPU, send an inference request, report feedback, and inspect the release history.",
  "user-guide/recipes":
    "Compare Reef recipes for model training and harness evolution. Choose a method based on your workload, feedback, and compute requirements.",
  "user-guide/evolve-your-harness":
    "Evolve agent rules, prompts, skills, and configuration with Reef. Evaluate candidate changes and publish accepted harness versions without a GPU.",
  "user-guide/evolve-your-model":
    "Configure Reef to train model weights from feedback and serve accepted updates through the same inference endpoint.",
  "reference/http-api":
    "Use the Reef HTTP API to send inference requests, report feedback, inspect scenarios, and manage model and harness releases.",
};

function loadDoc(sourcePath: string): Doc {
  const source = readFileSync(join(docsRoot, sourcePath), "utf8");
  const content = compileRst(source, sourcePath);
  const title = heading(content, 1)?.title;
  if (!title) throw new Error(`${sourcePath} is missing a level-one heading`);
  const slug = slugFromSourcePath(sourcePath);
  const description = docDescriptions[slug] ?? plainHtml(content.match(/<p>([\s\S]*?)<\/p>/)?.[1] ?? "");
  return {
    slug,
    title,
    description,
    content,
    sourcePath,
    toc: tableOfContents(content),
  };
}

const docs = navigationSources.flatMap((group) => group.files.map(loadDoc));
const docsBySlug = new Map(docs.map((doc) => [doc.slug, doc]));

export const navigation: NavGroup[] = navigationSources.map((group) => ({
  title: group.title,
  items: group.files.map((sourcePath) => {
    const doc = docsBySlug.get(slugFromSourcePath(sourcePath));
    if (!doc) throw new Error(`Documentation page not found: ${sourcePath}`);
    return {
      title: doc.title,
      description: doc.description,
      href: `/docs/${doc.slug}`,
      slug: doc.slug,
    };
  }),
}));

// Where /docs sends the reader: the first page of the first group, which is the
// start of the reading order.
export const docsIndexHref = navigation[0].items[0].href;

export function getAllDocs() {
  return docs;
}

export function getDoc(slug: string) {
  return docsBySlug.get(slug);
}

export function getAdjacentDocs(slug: string) {
  const items = navigation.flatMap((group) => group.items);
  const index = items.findIndex((item) => item.slug === slug);
  if (index < 0) return {};
  return { previous: items[index - 1], next: items[index + 1] };
}

export function getSearchDocuments() {
  return docs.map(({ slug, title, description, content }) => ({
    slug,
    title,
    description,
    text: plainHtml(content),
  }));
}
