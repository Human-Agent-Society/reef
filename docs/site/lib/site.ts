export const siteConfig = {
  title: "Reef Documentation",
  description: "Documentation for Reef, a continual learning controller: it serves an inference endpoint, records what it served, and publishes the next version of the weights or the harness.",
  repository: "https://github.com/Human-Agent-Society/reef",
  // The canonical origin for sitemap, robots, canonical links, and Open Graph.
  // Vercel previews override it with their own host so shared previews resolve.
  url:
    process.env.NEXT_PUBLIC_SITE_URL
    ?? (process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : "https://reefinfra.ai"),
} as const;
