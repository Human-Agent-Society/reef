import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

function fixture(context) {
  const directory = mkdtempSync(join(tmpdir(), "reef-stats-"));
  context.after(() => rmSync(directory, { recursive: true, force: true }));
  mkdirSync(join(directory, "scripts"));
  const script = join(directory, "scripts/fetch-repo-stats.mjs");
  copyFileSync(new URL("./fetch-repo-stats.mjs", import.meta.url), script);
  const hook = join(directory, "mock-fetch.mjs");
  writeFileSync(hook, `
    const user = { login: "contributor", type: "User", avatar_url: "https://example.com/avatar", html_url: "https://example.com/user", contributions: 1 };
    const mergedAt = new Date().toISOString();
    const pull = { number: 1, title: "Fix", html_url: "https://example.com/pull/1", user, pull_request: { merged_at: mergedAt } };
    globalThis.fetch = async (url) => {
      const path = new URL(url).pathname;
      if (process.env.STATS_FAILURE === "all" || (process.env.STATS_FAILURE === "core" && !path.startsWith("/search/"))) {
        return new Response("{}", { status: 403 });
      }
      let data;
      if (path.endsWith("/releases/latest")) {
        return new Response("{}", { status: 404 });
      } else if (path.endsWith("/contributors")) {
        data = [user];
      } else if (path.endsWith("/commits")) {
        data = [{ commit: { committer: { date: mergedAt } } }];
      } else if (path.startsWith("/search/")) {
        data = { total_count: process.env.STATS_FAILURE ? 352 : 351, items: [pull] };
      } else {
        data = { stargazers_count: 4274, forks_count: 347, created_at: mergedAt };
      }
      return Response.json(data);
    };
  `);
  return {
    target: join(directory, "lib/repo-stats.generated.json"),
    cache: join(directory, ".next/cache/repo-stats.json"),
    run(failure = "", lifecycle = "prebuild") {
      return spawnSync(process.execPath, ["--import", hook, script], {
        encoding: "utf8",
        env: { ...process.env, GITHUB_TOKEN: "test-token", STATS_FAILURE: failure, npm_lifecycle_event: lifecycle },
      });
    },
  };
}

test("successful fetch caches complete statistics; a missing release is optional", (context) => {
  const site = fixture(context);
  const result = site.run();
  assert.equal(result.status, 0, result.stderr);
  const stats = JSON.parse(readFileSync(site.target, "utf8"));
  assert.equal(stats.stars, 4274);
  assert.equal(stats.version, null);
  assert.equal(stats.contributors.length, 1);
  assert.equal(stats.mergedPulls.total, 351);
  assert.equal(stats.commitsByDay.reduce((sum, day) => sum + day.count, 0), 1);
  assert.equal(readFileSync(site.cache, "utf8"), readFileSync(site.target, "utf8"));
});

for (const source of ["target", "cache"]) {
  test(`partial GitHub outage reuses the complete ${source} snapshot without mixing dates or counts`, (context) => {
    const site = fixture(context);
    assert.equal(site.run().status, 0);
    const previous = readFileSync(site.target, "utf8");
    rmSync(source === "target" ? site.cache : site.target);
    const result = site.run("core");
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stderr, /using complete snapshot/);
    assert.equal(readFileSync(site.target, "utf8"), previous);
  });
}

test("production refuses an incomplete snapshot when GitHub core requests fail", (context) => {
  const site = fixture(context);
  const result = site.run("core");
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Set GITHUB_TOKEN or GH_TOKEN/);
  assert.equal(existsSync(site.target), false);
});

test("development can start offline, but its partial snapshot cannot pass a production build", (context) => {
  const site = fixture(context);
  assert.equal(site.run("all", "predev").status, 0);
  const previous = readFileSync(site.target, "utf8");
  assert.equal(JSON.parse(previous).stars, null);
  assert.equal(existsSync(site.cache), false);
  assert.notEqual(site.run("all").status, 0);
  assert.equal(readFileSync(site.target, "utf8"), previous);
});

test("a corrupt generated file falls back to the complete build cache", (context) => {
  const site = fixture(context);
  assert.equal(site.run().status, 0);
  const previous = readFileSync(site.cache, "utf8");
  writeFileSync(site.target, "{");
  const result = site.run("all");
  assert.equal(result.status, 0, result.stderr);
  assert.equal(readFileSync(site.target, "utf8"), previous);
});
