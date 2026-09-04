// Before dev and build: write the star count and the newest release tag for the header.
// The site is a static export, so the values are as of the build. A failure leaves nulls and never fails the build.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repo = "Human-Agent-Society/reef";
const target = resolve(dirname(fileURLToPath(import.meta.url)), "../lib/repo-stats.generated.json");
const headers = { accept: "application/vnd.github+json", "user-agent": "reef-docs" };

async function get(url) {
  const response = await fetch(url, { headers, signal: AbortSignal.timeout(5000) });
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

const stats = { stars: null, version: null };
try {
  const data = await get(`https://api.github.com/repos/${repo}`);
  if (typeof data.stargazers_count === "number") stats.stars = data.stargazers_count;
} catch (error) {
  console.warn(`repo stats: no star count (${error.message})`);
}
try {
  const data = await get(`https://api.github.com/repos/${repo}/releases/latest`);
  if (typeof data.tag_name === "string" && data.tag_name) stats.version = data.tag_name;
} catch (error) {
  console.warn(`repo stats: no release tag (${error.message})`);
}
mkdirSync(dirname(target), { recursive: true });
writeFileSync(target, JSON.stringify(stats) + "\n");
console.log(`repo stats: stars=${stats.stars} version=${stats.version}`);
