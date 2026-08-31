import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { docsIndexHref } from "@/lib/docs";

// There is no docs landing page: the sidebar is the index, and its first entry
// is the start of the reading order. /docs redirects there so that the header
// link, the 404 page, and external links to /docs all reach a real page.
export const metadata: Metadata = { robots: { index: false, follow: true } };

export default function DocumentationHome() {
  redirect(docsIndexHref);
}
