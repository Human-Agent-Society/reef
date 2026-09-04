import { readFileSync } from "node:fs";
import { join } from "node:path";

export type RepoStats = { stars: number | null; version: string | null };

// scripts/fetch-repo-stats.mjs writes this file before dev and build; a missing or bad file means no count and no tag.
export function getRepoStats(): RepoStats {
  try {
    const data = JSON.parse(readFileSync(join(process.cwd(), "lib", "repo-stats.generated.json"), "utf8"));
    return {
      stars: typeof data.stars === "number" ? data.stars : null,
      version: typeof data.version === "string" && data.version ? data.version : null,
    };
  } catch {
    return { stars: null, version: null };
  }
}

const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

export function formatStars(stars: number) {
  return compact.format(stars).toLowerCase();
}
