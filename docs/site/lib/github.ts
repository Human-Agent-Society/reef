import { readFileSync } from "node:fs";
import { join } from "node:path";

export type Contributor = { login: string; avatar: string; url: string; contributions: number };
export type MergedPull = { number: number; title: string; url: string; author: string; avatar: string; mergedAt: string };
export type DayCount = { date: string; count: number };

export type RepoStats = {
  fetchedAt: string | null;
  stars: number | null;
  forks: number | null;
  version: string | null;
  versionDate: string | null;
  contributors: Contributor[];
  mergedPulls: { total: number | null; recent: number | null; windowDays: number };
  commitsByDay: DayCount[];
  recentPulls: MergedPull[];
};

const empty: RepoStats = {
  fetchedAt: null,
  stars: null,
  forks: null,
  version: null,
  versionDate: null,
  contributors: [],
  mergedPulls: { total: null, recent: null, windowDays: 30 },
  commitsByDay: [],
  recentPulls: [],
};

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

function list<T>(value: unknown, keep: (item: Record<string, unknown>) => T | null): T[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => (item && typeof item === "object" ? (keep(item as Record<string, unknown>) ?? []) : []));
}

// scripts/fetch-repo-stats.mjs writes this file before dev and build; a missing or bad file means every field is empty.
export function getRepoStats(): RepoStats {
  try {
    const data = JSON.parse(readFileSync(join(process.cwd(), "lib", "repo-stats.generated.json"), "utf8"));
    const merged = data.mergedPulls && typeof data.mergedPulls === "object" ? data.mergedPulls : {};
    return {
      fetchedAt: text(data.fetchedAt),
      stars: number(data.stars),
      forks: number(data.forks),
      version: text(data.version),
      versionDate: text(data.versionDate),
      contributors: list(data.contributors, (c) => {
        const login = text(c.login);
        const avatar = text(c.avatar);
        const url = text(c.url);
        return login && avatar && url ? { login, avatar, url, contributions: number(c.contributions) ?? 0 } : null;
      }),
      mergedPulls: { total: number(merged.total), recent: number(merged.recent), windowDays: number(merged.windowDays) ?? 30 },
      commitsByDay: list(data.commitsByDay, (d) => {
        const date = text(d.date);
        return date ? { date, count: number(d.count) ?? 0 } : null;
      }),
      recentPulls: list(data.recentPulls, (p) => {
        const numberValue = number(p.number);
        const title = text(p.title);
        const url = text(p.url);
        const author = text(p.author);
        const avatar = text(p.avatar);
        const mergedAt = text(p.mergedAt);
        return numberValue && title && url && author && avatar && mergedAt ? { number: numberValue, title, url, author, avatar, mergedAt } : null;
      }),
    };
  } catch {
    return empty;
  }
}

const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

export function formatStars(stars: number) {
  return compact.format(stars).toLowerCase();
}

// "feat(recipe): ship Reefine" reads as "Ship Reefine" on the home page; the type and scope stay in the PR.
export function humanizePullTitle(title: string) {
  const stripped = title.replace(/^[a-z]+(\([^)]*\))?!?:\s*/i, "").replace(/^\[[^\]]*\]\s*/, "").trim();
  return stripped.charAt(0).toUpperCase() + stripped.slice(1);
}

export function relativeDay(iso: string, now = new Date()) {
  const days = Math.max(0, Math.floor((now.getTime() - new Date(iso).getTime()) / 86_400_000));
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString("en", { month: "short", day: "numeric" });
}
