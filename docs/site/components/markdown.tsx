import path from "node:path";
import Link from "next/link";
import parse, {
  attributesToProps,
  domToReact,
  Element,
  type DOMNode,
  type HTMLReactParserOptions,
} from "html-react-parser";
import { CircleAlert, ExternalLink, Info, Lightbulb, TriangleAlert } from "lucide-react";
import { MermaidDiagram } from "@/components/mermaid-diagram";
import { getDoc } from "@/lib/docs";
import { siteConfig } from "@/lib/site";

// One callout shape, color-keyed: the icon and title word are presentation,
// so they are injected here rather than authored, and the severity kinds the
// compiler's built-in admonition plugin emits all map onto three tints.
const admonitionIcons = {
  note: Info,
  hint: Lightbulb,
  tip: Lightbulb,
  caution: TriangleAlert,
  warning: TriangleAlert,
  important: CircleAlert,
} as const;

function textFromNodes(nodes: DOMNode[]): string {
  return nodes.map((node) => {
    if (node.type === "text") return node.data;
    if (node instanceof Element) return textFromNodes(node.children as DOMNode[]);
    return "";
  }).join("");
}

function resolveHref(href: string, sourcePath: string) {
  if (!href || href.startsWith("#") || /^(https?:|mailto:)/.test(href)) return href;

  const [pathname, hash] = href.split("#");
  const repoSource = path.posix.join("docs", sourcePath);
  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(repoSource), pathname));
  const suffix = hash ? `#${hash}` : "";

  // Only pages the site actually builds become internal links. The rest of the
  // repository — rfcs/, historical/, code, examples — is linked on GitHub,
  // where those files are read.
  if (resolved.startsWith("docs/") && resolved.endsWith(".rst")) {
    const slug = resolved.slice("docs/".length, -".rst".length);
    if (getDoc(slug)) return `/docs/${slug}${suffix}`;
  }

  const hasExtension = /\.[a-z0-9]+$/i.test(resolved);
  return `${siteConfig.repository}/${hasExtension ? "blob" : "tree"}/main/${resolved}${suffix}`;
}

function resolveImageSource(src: string, sourcePath: string) {
  if (!src || /^(https?:|data:|\/)/.test(src)) return src;
  const repoSource = path.posix.join("docs", sourcePath);
  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(repoSource), src));
  return `${siteConfig.repository}/raw/main/${resolved}`;
}

export function ReStructuredText({ content, sourcePath }: { content: string; sourcePath: string }) {
  const options: HTMLReactParserOptions = {
    replace(node) {
      if (!(node instanceof Element)) return;
      const children = node.children as DOMNode[];

      if (node.name === "pre" && node.attribs.class?.split(" ").includes("language-mermaid")) {
        return <MermaidDiagram chart={textFromNodes(children).replace(/\n$/, "")} />;
      }

      if (node.name === "div" && node.attribs.class?.split(" ").includes("admonition")) {
        const classes = node.attribs.class.split(" ");
        const kind = classes.find((name): name is keyof typeof admonitionIcons => name in admonitionIcons) ?? "note";
        const Icon = admonitionIcons[kind];
        return (
          <aside className={node.attribs.class}>
            <p className="admonition-title">
              <Icon size={15} aria-hidden />
              {kind.charAt(0).toUpperCase() + kind.slice(1)}
            </p>
            {domToReact(children, options)}
          </aside>
        );
      }

      if (/^h[1-3]$/.test(node.name)) {
        const level = Number(node.name.slice(1)) as 1 | 2 | 3;
        const Tag = `h${level}` as const;
        const id = node.attribs.id;
        const title = textFromNodes(children);
        return (
          <Tag id={id}>
            {domToReact(children, options)}
            {level > 1 && id && (
              <a className="heading-anchor" href={`#${id}`} aria-label={`Link to ${title}`}>#</a>
            )}
          </Tag>
        );
      }

      if (node.name === "table") {
        return (
          <div className="table-scroll">
            <table {...attributesToProps(node.attribs)}>{domToReact(children, options)}</table>
          </div>
        );
      }

      if (node.name === "a") {
        const resolved = resolveHref(node.attribs.href ?? "", sourcePath);
        const renderedChildren = domToReact(children, options);
        if (/^https?:/.test(resolved)) {
          return (
            <a href={resolved} target="_blank" rel="noreferrer">
              {renderedChildren}<ExternalLink className="external-link-icon" size={12} />
            </a>
          );
        }
        return <Link href={resolved}>{renderedChildren}</Link>;
      }

      if (node.name === "img") {
        return (
          // RST image directives always supply a source; an empty alt remains
          // meaningful for decorative images and avoids inventing copy.
          // eslint-disable-next-line @next/next/no-img-element
          <img {...node.attribs} src={resolveImageSource(node.attribs.src ?? "", sourcePath)} alt={node.attribs.alt ?? ""} />
        );
      }
    },
  };

  return <>{parse(content, options)}</>;
}
