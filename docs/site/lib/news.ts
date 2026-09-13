import { humanizePullTitle, type MergedPull } from "./github";

export type NewsItem = { kind: "news" | "release" | "pull"; text: string; href: string; date: string };

// Milestones worth a line even after the pull requests behind them have scrolled off.
export const news: NewsItem[] = [
  { kind: "news", text: "Reefine ships: refine a coding harness from plain-language instructions, no GPU", href: "/docs/user-guide/recipes/reefine", date: "2026-09-12" },
  { kind: "news", text: "reef serve starts inference without a YAML file", href: "/docs/reference/cli", date: "2026-09-12" },
  { kind: "news", text: "Coral test-time training lands as a beta recipe", href: "https://github.com/Human-Agent-Society/reef/tree/main/recipes", date: "2026-09-09" },
  { kind: "news", text: "Reef is open source under Apache-2.0", href: "https://github.com/Human-Agent-Society/reef", date: "2026-08-31" },
];

// The ticker mixes the milestones with the newest merged pull requests and the latest release, newest first.
export function tickerItems(pulls: MergedPull[], version: string | null, versionDate: string | null, repository: string): NewsItem[] {
  const items: NewsItem[] = [...news];
  if (version && versionDate) {
    items.push({ kind: "release", text: `${version} released`, href: `${repository}/releases/tag/${version}`, date: versionDate.slice(0, 10) });
  }
  for (const pull of pulls.slice(0, 6)) {
    items.push({ kind: "pull", text: `${humanizePullTitle(pull.title)} · #${pull.number}`, href: pull.url, date: pull.mergedAt.slice(0, 10) });
  }
  return items.sort((a, b) => b.date.localeCompare(a.date));
}
