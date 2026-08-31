import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { DocsPage } from "@/components/docs-page";
import { getAdjacentDocs, getAllDocs, getDoc, navigation } from "@/lib/docs";
import { siteConfig } from "@/lib/site";

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllDocs().filter((doc) => doc.slug).map((doc) => ({ slug: doc.slug.split("/") }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string[] }> }): Promise<Metadata> {
  const { slug } = await params;
  const doc = getDoc(slug.join("/"));
  if (!doc) return {};
  const path = `/docs/${doc.slug}/`;
  return {
    title: doc.title,
    description: doc.description,
    alternates: { canonical: path },
    openGraph: { title: doc.title, description: doc.description, type: "article", url: path },
  };
}

// TechArticle + breadcrumb structured data, so search results can show the
// section a page lives in and treat the site as one documentation set.
function structuredData(slug: string, title: string, description: string) {
  const group = navigation.find((candidate) => candidate.items.some((item) => item.slug === slug));
  const page = `${siteConfig.url}/docs/${slug}/`;
  const crumbs = [
    { name: "Reef", item: `${siteConfig.url}/` },
    ...(group ? [{ name: group.title, item: `${siteConfig.url}${group.items[0].href}/` }] : []),
    { name: title, item: page },
  ];
  return [
    {
      "@context": "https://schema.org",
      "@type": "TechArticle",
      headline: title,
      description,
      url: page,
      isPartOf: { "@type": "WebSite", name: siteConfig.title, url: `${siteConfig.url}/` },
      publisher: { "@type": "Organization", name: "Human-Agent Society", url: siteConfig.repository },
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: crumbs.map((crumb, index) => ({ "@type": "ListItem", position: index + 1, ...crumb })),
    },
  ];
}

export default async function DocumentationPage({ params }: { params: Promise<{ slug: string[] }> }) {
  const { slug } = await params;
  const value = slug.join("/");
  const doc = getDoc(value);
  if (!doc) notFound();
  const adjacent = getAdjacentDocs(value);
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData(doc.slug, doc.title, doc.description)) }}
      />
      <DocsPage doc={doc} navigation={navigation} {...adjacent} />
    </>
  );
}
