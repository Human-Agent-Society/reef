import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import langJson from "@shikijs/langs/json";
import langPython from "@shikijs/langs/python";
import langShellscript from "@shikijs/langs/shellscript";
import langShellsession from "@shikijs/langs/shellsession";
import langYaml from "@shikijs/langs/yaml";
import {
  RstDefinitionList,
  RstDefinitionListItem,
  type RstDirective,
  type RstDirectiveGenerator,
  RstFieldList,
  RstFieldListItem,
  type RstGeneratorState,
  type RstNode,
  type RstNodeGenerator,
  RstParagraph,
  RstStrongEmphasis,
  RstToHtmlCompiler,
} from "rst-compiler";
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
      "getting-started/harness-tutorial.rst",
      "getting-started/core-loop.rst",
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
      "user-guide/tinker.rst",
      "user-guide/recipes/sft.rst",
      "user-guide/recipes/sao.rst",
      "user-guide/recipes/sdft.rst",
      "user-guide/recipes/sdpo.rst",
      "user-guide/recipes/opd.rst",
      "user-guide/recipes/tttd.rst",
      "user-guide/recipes/guidance-ttt.rst",
      "user-guide/recipes/openclawrl.rst",
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
      "developer-guide/sandboxes.rst",
      "developer-guide/executors.rst",
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
    title: "Advanced Topics",
    files: [
      "advanced_topics/state-model.rst",
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
// directory now fails the build until it is placed in the reading order.
const navigated = new Set<string>(navigationSources.flatMap((group) => group.files));
// Pages live in one directory per navigation section, so a page's path names
// its section and the URL follows the path.
const sectionDirectories = ["getting-started", "user-guide", "user-guide/recipes", "developer-guide", "reference", "advanced_topics", "contributing"];
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

// The items of the definition lists in a container, and its fields; an Error
// naming the page for any other child, which the site would not show.
function containerParts(node: RstDirective, sourcePath: string) {
  const items: RstDefinitionListItem[] = [];
  const fields: RstFieldListItem[] = [];
  for (const child of node.children) {
    if (child instanceof RstDefinitionList) {
      items.push(...child.children.filter((item) => item instanceof RstDefinitionListItem));
    } else if (child instanceof RstFieldList) {
      fields.push(...child.children.filter((item) => item instanceof RstFieldListItem));
    } else {
      throw new Error(`${sourcePath}: a ${node.initContentText} container holds a definition list, not ${child.nodeType}`);
    }
  }
  return { items, fields };
}

// `.. container:: flow` is the lifecycle figure: a definition list with one
// stage per item (a bold term for the emphasis tint), and an optional `:Loop:`
// field for the labeled return arrow. GitHub shows the list as it is. Stages
// are DOM text in flex cards, so labels hold their size and the strip reflows
// to a vertical stack on narrow screens.
function writeFlow(generatorState: RstGeneratorState, node: RstDirective, sourcePath: string) {
  const { items, fields } = containerParts(node, sourcePath);
  const loop = fields.find((field) => field.nameText === "Loop");
  if (fields.length !== (loop ? 1 : 0)) throw new Error(`${sourcePath}: a flow takes no field but :Loop:`);
  generatorState.writeLine('<figure class="fig fig-flow"><div class="fig-flow-track">');
  items.forEach((item, index) => {
    if (index > 0) generatorState.writeLine(arrowSvg("next"));
    const [first] = item.term;
    const emphasized = item.term.length === 1 && first instanceof RstStrongEmphasis;
    generatorState.writeLine(`<div class="fig-flow-stage${emphasized ? " fig-emphasis" : ""}">`);
    generatorState.writeLine('<span class="fig-flow-title">');
    // A bold term keeps its text in the node itself, not in children.
    if (emphasized) generatorState.writeLine(escapeHtml(first.textContent));
    else generatorState.writeLineVisitor(() => generatorState.visitNodes(item.term));
    generatorState.writeLine('</span><span class="fig-flow-caption">');
    visitInline(generatorState, item.definition);
    generatorState.writeLine("</span></div>");
  });
  generatorState.writeLine("</div>");
  if (loop) {
    generatorState.writeLine(`<div class="fig-flow-loop">${arrowSvg("loop")}<span>`);
    visitInline(generatorState, loop.body);
    generatorState.writeLine("</span></div>");
  }
  generatorState.writeLine("</figure>");
}

// `.. container:: config` is reference rows: a definition list whose term is
// the key as a literal, with the default as an optional classifier
// (``key`` : default). Each key gets an id, so rows are deep-linkable without
// spending headings on them.
function writeConfig(generatorState: RstGeneratorState, node: RstDirective, sourcePath: string) {
  const { items, fields } = containerParts(node, sourcePath);
  if (fields.length) throw new Error(`${sourcePath}: a config container takes no field`);
  generatorState.writeLine('<div class="config-card">');
  for (const item of items) {
    const key = item.term.map((text) => text.textContent).join("");
    const id = `cfg-${key.toLowerCase().replace(/[^a-z0-9_.-]+/g, "-")}`;
    generatorState.writeLine(
      `<div class="config-row" id="${escapeHtml(id)}"><code class="config-key">${escapeHtml(key)}</code>`,
    );
    generatorState.writeLine('<span class="config-desc">');
    for (const classifier of item.classifiers) {
      generatorState.writeLine('<span class="config-default">');
      generatorState.writeLineVisitor(() => generatorState.visitNodes(classifier));
      generatorState.writeLine("</span>");
    }
    visitInline(generatorState, item.definition);
    generatorState.writeLine("</span></div>");
  }
  generatorState.writeLine("</div>");
}

// The page brief is a plain RST field list whose fields are these labels, so
// GitHub renders it as a table: who the page is for, what to have ready, and
// what the reader has at the end. Any other field list renders as usual.
const PAGE_BRIEF_FIELDS = new Set(["For", "Before you start", "You will have"]);

// A body of one paragraph writes its inline content without the <p>: a page
// takes its first <p> as its description, and the brief can come before it.
function visitInline(generatorState: RstGeneratorState, nodes: ReadonlyArray<RstNode>) {
  const [first] = nodes;
  if (nodes.length === 1 && first instanceof RstParagraph) {
    generatorState.writeLineVisitor(() => generatorState.visitNodes(first.children));
  } else {
    generatorState.visitNodes(nodes);
  }
}

function compileRst(content: string, sourcePath: string) {
  const compiler = new RstToHtmlCompiler();
  const fieldList = compiler.nodeGenerators.get("FieldList") as RstNodeGenerator<"FieldList">;
  compiler.useNodeGenerator({
    nodeType: "FieldList",
    generate(generatorState, node) {
      const items = node.children.filter((child) => child instanceof RstFieldListItem);
      if (items.length !== node.children.length || !items.every((item) => PAGE_BRIEF_FIELDS.has(item.nameText))) {
        fieldList.generate(generatorState, node);
        return;
      }
      generatorState.writeLine('<dl class="page-brief">');
      for (const item of items) {
        generatorState.writeLine(`<dt>${escapeHtml(item.nameText)}</dt>`);
        generatorState.writeLine("<dd>");
        visitInline(generatorState, item.body);
        generatorState.writeLine("</dd>");
      }
      generatorState.writeLine("</dl>");
    },
  });
  // A figure of a `-light.svg` with a `-dark.svg` beside it shows the one that
  // matches the theme. The RST names only the light file, which is what GitHub
  // renders; every other figure renders as usual.
  const figure = compiler.directiveGenerators.get("figure") as RstDirectiveGenerator;
  compiler.useDirectiveGenerator({
    directives: ["figure"],
    generate(generatorState, node) {
      const light = node.initContentText;
      const dark = light.replace(/-light\.svg$/, "-dark.svg");
      if (dark === light || !existsSync(join(docsRoot, dirname(sourcePath), dark))) {
        figure.generate(generatorState, node);
        return;
      }
      const alt = escapeHtml(node.config?.getField("alt")?.replace(/\s+/g, " ").trim() ?? "");
      generatorState.writeLine('<figure class="fig fig-diagram">');
      generatorState.writeLine(`<img class="fig-light" src="${escapeHtml(light)}" alt="${alt}">`);
      generatorState.writeLine(`<img class="fig-dark" src="${escapeHtml(dark)}" alt="${alt}">`);
      if (node.children.length) {
        generatorState.writeLine("<figcaption>");
        visitInline(generatorState, node.children);
        generatorState.writeLine("</figcaption>");
      }
      generatorState.writeLine("</figure>");
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
  // A walkthrough is `.. container:: steps` around an ordinary enumerated
  // list: the compiler writes `<div class="container steps">`, and CSS
  // counters draw the numbered circles and the connector line.
  const container = compiler.directiveGenerators.get("container") as RstDirectiveGenerator;
  compiler.useDirectiveGenerator({
    directives: ["container"],
    generate(generatorState, node) {
      const kind = node.initContentText.trim();
      if (kind === "flow") {
        writeFlow(generatorState, node, sourcePath);
      } else if (kind === "config") {
        writeConfig(generatorState, node, sourcePath);
      } else {
        container.generate(generatorState, node);
      }
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
  "getting-started/harness-tutorial":
    "Start Reef, connect your coding agent, give feedback on a session, and see Reef evolve the agent's harness from it.",
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
