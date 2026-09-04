export const siteConfig = {
  title: "Reef — Continual learning infrastructure",
  description: "Reef is continual learning infrastructure for AI agents. Serve inference, collect feedback, and evolve model weights or agent harnesses.",
  repository: "https://github.com/Human-Agent-Society/reef",
  // The canonical origin for sitemap, robots, canonical links, and Open Graph.
  // Vercel previews override it with their own host so shared previews resolve.
  url:
    process.env.NEXT_PUBLIC_SITE_URL
    ?? (process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : "https://reefinfra.ai"),
} as const;
