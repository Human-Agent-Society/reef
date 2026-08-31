import type { MetadataRoute } from "next";
import { getAllDocs } from "@/lib/docs";
import { siteConfig } from "@/lib/site";

export const dynamic = "force-static";

// One entry per page that carries a canonical: the home page and every doc.
// /docs itself redirects and is noindex, so it is not listed.
export default function sitemap(): MetadataRoute.Sitemap {
  return [
    { url: `${siteConfig.url}/`, priority: 1 },
    ...getAllDocs()
      .filter((doc) => doc.slug)
      .map((doc) => ({ url: `${siteConfig.url}/docs/${doc.slug}/`, priority: 0.8 })),
  ];
}
