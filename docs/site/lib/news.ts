export type NewsItem = { kind: "news" | "release"; text: string; href: string; date: string };

// Project news for the home page ticker. Add a line here when something ships.
export const news: NewsItem[] = [
  { kind: "news", text: "Reefine ships: refine a coding harness from plain-language instructions, no GPU", href: "/docs/user-guide/recipes/reefine", date: "2026-09-12" },
  { kind: "news", text: "reef serve starts inference without a YAML file", href: "/docs/reference/cli", date: "2026-09-12" },
  { kind: "news", text: "Coral test-time training lands as a beta recipe", href: "https://github.com/Human-Agent-Society/reef/tree/main/recipes", date: "2026-09-09" },
  { kind: "news", text: "Reef is open source under Apache-2.0", href: "https://github.com/Human-Agent-Society/reef", date: "2026-08-31" },
];

// The ticker carries the milestones and the latest release, newest first. Merged pull requests stay in the activity section.
export function tickerItems(version: string | null, versionDate: string | null, repository: string): NewsItem[] {
  const items: NewsItem[] = [...news];
  if (version && versionDate) {
    items.push({ kind: "release", text: `${version} released`, href: `${repository}/releases/tag/${version}`, date: versionDate.slice(0, 10) });
  }
  return items.sort((a, b) => b.date.localeCompare(a.date));
}
