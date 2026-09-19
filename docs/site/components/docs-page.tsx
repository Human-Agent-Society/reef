import Link from "next/link";
import { ArrowLeft, ArrowRight, PencilLine } from "lucide-react";
import type { Doc, NavGroup, NavItem } from "@/lib/docs";
import { siteConfig } from "@/lib/site";
import { ReStructuredText } from "./markdown";
import { Sidebar } from "./sidebar";
import { TableOfContents } from "./table-of-contents";

function PagerLink({ item, direction }: { item: NavItem; direction: "previous" | "next" }) {
  return (
    <Link className={`pager-link ${direction}`} href={item.href}>
      <small>{direction === "previous" ? "Previous" : "Next"}</small>
      <span>{direction === "previous" && <ArrowLeft size={16} />}{item.title}{direction === "next" && <ArrowRight size={16} />}</span>
      <p>{item.description}</p>
    </Link>
  );
}

export function DocsPage({ doc, navigation, previous, next }: { doc: Doc; navigation: NavGroup[]; previous?: NavItem; next?: NavItem }) {
  const group = navigation.find((candidate) => candidate.items.some((item) => item.slug === doc.slug));

  return (
    <div className="docs-layout">
      <Sidebar navigation={navigation} />
      <main id="main-content" className="docs-main" tabIndex={-1}>
        {group && (
          <p className="doc-eyebrow" aria-label="Section">
            <Link href={group.items[0].href}>{group.title}</Link>
          </p>
        )}
        <article className="markdown-body">
          <ReStructuredText content={doc.content} sourcePath={doc.sourcePath} />
        </article>
        <div className="edit-row">
          <a href={`${siteConfig.repository}/edit/main/docs/${doc.sourcePath}`} target="_blank" rel="noreferrer"><PencilLine size={14} /> Edit this page on GitHub</a>
        </div>
        <nav className="pager" aria-label="Previous and next pages">
          {previous ? <PagerLink item={previous} direction="previous" /> : <span />}
          {next ? <PagerLink item={next} direction="next" /> : <span />}
        </nav>
      </main>
      <TableOfContents items={doc.toc} />
    </div>
  );
}
