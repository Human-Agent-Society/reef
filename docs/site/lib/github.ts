import { siteConfig } from "./site";

export type RepoStats = { stars: number | null; version: string | null };

// Read once at build (the site is a static export): the star count and the
// newest tag, for the header. Either value is null when GitHub is unreachable
// or rate limited, and the header simply leaves it out.
export async function getRepoStats(): Promise<RepoStats> {
  const api = `https://api.github.com/repos/${siteConfig.repoName}`;
  const init = { headers: { accept: "application/vnd.github+json", "user-agent": "reef-docs" }, cache: "force-cache" as const };
  const stats: RepoStats = { stars: null, version: null };
  try {
    const repo = await fetch(api, init);
    if (repo.ok) {
      const data: { stargazers_count?: number } = await repo.json();
      if (typeof data.stargazers_count === "number") stats.stars = data.stargazers_count;
    }
  } catch {
    /* offline build: no star count */
  }
  try {
    const tags = await fetch(`${api}/tags?per_page=1`, init);
    if (tags.ok) {
      const data: { name?: string }[] = await tags.json();
      if (data[0]?.name) stats.version = data[0].name;
    }
  } catch {
    /* offline build: no version */
  }
  return stats;
}

export function formatStars(stars: number) {
  return stars >= 1000 ? `${(stars / 1000).toFixed(stars >= 10000 ? 0 : 1)}k` : String(stars);
}
