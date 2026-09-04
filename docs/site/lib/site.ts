export const siteConfig = {
  title: "Reef — Continual Self-Improvement for AI agents",
  description: "Open-source continual learning infrastructure for Self-Improving Agents. Connect agent inference, feedback, model training, and versioned deployment.",
  repository: "https://github.com/Human-Agent-Society/reef",
  repoName: "Human-Agent-Society/reef",
  // The canonical origin for sitemap, robots, canonical links, and Open Graph.
  // Vercel previews override it with their own host so shared previews resolve.
  url:
    process.env.NEXT_PUBLIC_SITE_URL
    ?? (process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : "https://reefinfra.ai"),
} as const;
