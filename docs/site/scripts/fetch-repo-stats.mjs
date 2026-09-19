// Before dev and build: write the repository numbers the header and the home page show.
// The site is a static export, so the values are as of the build. Every request fails
// independently and leaves its field empty; nothing here can fail the build.
import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repo = "Human-Agent-Society/reef";
const api = "https://api.github.com";
const target = resolve(dirname(fileURLToPath(import.meta.url)), "../lib/repo-stats.generated.json");
// A token lifts the anonymous limits (the search endpoints allow ten calls a minute). CI passes one;
// on a developer machine the GitHub CLI's token is used when it is logged in.
function localToken() {
  const result = spawnSync("gh", ["auth", "token"], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
  return result.status === 0 ? result.stdout.trim() : "";
}
const token = process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN ?? localToken();
const headers = {
  accept: "application/vnd.github+json",
  "user-agent": "reef-docs",
  ...(token ? { authorization: `Bearer ${token}` } : {}),
};

const DAY = 24 * 60 * 60 * 1000;
const WINDOW_DAYS = 30;
const now = new Date();
// The window is the last 30 days, or the repository's whole life when it is younger than that.
let windowStart = new Date(now.getTime() - WINDOW_DAYS * DAY);

async function get(path) {
  const response = await fetch(`${api}${path}`, { headers, signal: AbortSignal.timeout(8000) });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

// Each step names what it fills so the log says exactly which number is missing.
async function step(name, fill) {
  try {
    await fill();
  } catch (error) {
    console.warn(`repo stats: no ${name} (${error.message})`);
  }
}

const stats = {
  fetchedAt: now.toISOString(),
  stars: null,
  forks: null,
  version: null,
  versionDate: null,
  contributors: [],
  mergedPulls: { total: null, recent: null, windowDays: WINDOW_DAYS },
  commitsByDay: [],
  recentPulls: [],
};

await step("star count", async () => {
  const data = await get(`/repos/${repo}`);
  if (typeof data.stargazers_count === "number") stats.stars = data.stargazers_count;
  if (typeof data.forks_count === "number") stats.forks = data.forks_count;
  const created = new Date(data.created_at);
  if (!Number.isNaN(created.getTime()) && created > windowStart) windowStart = new Date(created.toISOString().slice(0, 10));
});
const windowDays = Math.max(1, Math.ceil((now.getTime() - windowStart.getTime()) / DAY));
const sinceDate = windowStart.toISOString().slice(0, 10);
stats.mergedPulls.windowDays = windowDays;

await step("release tag", async () => {
  const data = await get(`/repos/${repo}/releases/latest`);
  if (typeof data.tag_name === "string" && data.tag_name) stats.version = data.tag_name;
  if (typeof data.published_at === "string") stats.versionDate = data.published_at;
});

await step("contributor list", async () => {
  const data = await get(`/repos/${repo}/contributors?per_page=100`);
  stats.contributors = data
    .filter((user) => user.type === "User" && !user.login.endsWith("[bot]"))
    .map((user) => ({ login: user.login, avatar: user.avatar_url, url: user.html_url, contributions: user.contributions }));
});

function pullSummary(item, mergedAt) {
  return { number: item.number, title: item.title, url: item.html_url, author: item.user.login, avatar: item.user.avatar_url, mergedAt };
}

await step("merged pull requests", async () => {
  const query = encodeURIComponent(`repo:${repo} is:pr is:merged`);
  const data = await get(`/search/issues?q=${query}&sort=updated&order=desc&per_page=12`);
  if (typeof data.total_count === "number") stats.mergedPulls.total = data.total_count;
  stats.recentPulls = data.items
    .filter((item) => item.pull_request?.merged_at)
    .sort((a, b) => b.pull_request.merged_at.localeCompare(a.pull_request.merged_at))
    .slice(0, 8)
    .map((item) => pullSummary(item, item.pull_request.merged_at));
});

// The search endpoint is the one most likely to be rate limited; the plain list still gives the feed.
if (stats.recentPulls.length === 0) {
  await step("merged pull request list", async () => {
    const data = await get(`/repos/${repo}/pulls?state=closed&sort=updated&direction=desc&per_page=30`);
    stats.recentPulls = data
      .filter((item) => item.merged_at)
      .sort((a, b) => b.merged_at.localeCompare(a.merged_at))
      .slice(0, 8)
      .map((item) => pullSummary(item, item.merged_at));
  });
}

await step("recent merged count", async () => {
  const query = encodeURIComponent(`repo:${repo} is:pr is:merged merged:>=${sinceDate}`);
  const data = await get(`/search/issues?q=${query}&per_page=1`);
  if (typeof data.total_count === "number") stats.mergedPulls.recent = data.total_count;
});

await step("commit activity", async () => {
  const counts = new Map();
  for (let day = 0; day < windowDays; day += 1) {
    counts.set(new Date(windowStart.getTime() + day * DAY).toISOString().slice(0, 10), 0);
  }
  counts.set(now.toISOString().slice(0, 10), counts.get(now.toISOString().slice(0, 10)) ?? 0);
  for (let page = 1; page <= 4; page += 1) {
    const commits = await get(`/repos/${repo}/commits?since=${windowStart.toISOString()}&per_page=100&page=${page}`);
    for (const commit of commits) {
      const date = commit.commit?.committer?.date?.slice(0, 10);
      if (counts.has(date)) counts.set(date, counts.get(date) + 1);
    }
    if (commits.length < 100) break;
  }
  const days = [...counts].map(([date, count]) => ({ date, count }));
  // A day that has only just begun in UTC has no commits yet; leave it off rather than end the chart on a gap.
  if (days.length > 1 && days[days.length - 1].count === 0) days.pop();
  stats.commitsByDay = days;
  stats.mergedPulls.windowDays = days.length;
});

mkdirSync(dirname(target), { recursive: true });
writeFileSync(target, JSON.stringify(stats) + "\n");
console.log(
  `repo stats: stars=${stats.stars} version=${stats.version} contributors=${stats.contributors.length} ` +
    `merged=${stats.mergedPulls.total} recent=${stats.mergedPulls.recent} commits=${stats.commitsByDay.reduce((sum, d) => sum + d.count, 0)}`,
);
