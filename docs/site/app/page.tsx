/* eslint-disable @next/next/no-img-element -- GitHub avatars are remote, unoptimized images */
import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, GitMerge, Star } from "lucide-react";
import { CommitChart } from "@/components/home/commit-chart";
import { CountUp } from "@/components/home/count-up";
import { LogoWall } from "@/components/home/logo-wall";
import { LoopDiagram } from "@/components/home/loop-diagram";
import { NewsTicker } from "@/components/home/news-ticker";
import { GitHubIcon } from "@/components/github-icon";
import { formatStars, getRepoStats, humanizePullTitle, relativeDay } from "@/lib/github";
import { tickerItems } from "@/lib/news";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = {
  title: { absolute: siteConfig.title },
  description: siteConfig.description,
  alternates: { canonical: "/" },
  openGraph: { title: siteConfig.title, description: siteConfig.description, url: "/" },
};

const recipeGroups = [
  {
    title: "Model weights",
    note: "GPU · Slime and SGLang",
    recipes: [
      { name: "SAO", href: "/docs/user-guide/recipes/sao", signal: "learns from a single rollout" },
      { name: "OpenClaw-RL", href: "/docs/user-guide/recipes/openclawrl", signal: "learns from personalized chat" },
      { name: "TTT-Discover", href: "/docs/user-guide/recipes/tttd", signal: "trains at test time" },
      { name: "Guidance-TTT", href: `${siteConfig.repository}/tree/main/recipes/tttd/examples/guidance_ttt`, signal: "guides a program search" },
    ],
  },
  {
    title: "Agent harness",
    note: "no GPU · any model endpoint",
    recipes: [
      { name: "Reefine", href: "/docs/user-guide/recipes/reefine", signal: "plain-language harness edits" },
      { name: "SkillClaw", href: "/docs/user-guide/recipes/skillclaw", signal: "grows skills from failures" },
      { name: "GEPA", href: "/docs/user-guide/recipes/gepa", signal: "reflective prompt evolution" },
      { name: "Meta-Harness", href: `${siteConfig.repository}/tree/main/recipes/meta_harness`, signal: "searches full harness designs" },
    ],
  },
];

export default function Home() {
  const stats = getRepoStats();
  const now = stats.fetchedAt ? new Date(stats.fetchedAt) : new Date();
  const commits = stats.commitsByDay.reduce((sum, day) => sum + day.count, 0);
  const ticker = tickerItems(stats.version, stats.versionDate, siteConfig.repository);
  const heroStats = [
    stats.stars !== null && { label: "GitHub stars", value: stats.stars, style: "compact" as const },
    stats.contributors.length > 0 && { label: "Contributors", value: stats.contributors.length, style: "plain" as const },
    stats.mergedPulls.total !== null && { label: "PRs merged", value: stats.mergedPulls.total, style: "plain" as const },
    commits > 0 && { label: `Commits · ${stats.commitsByDay.length}d`, value: commits, style: "plain" as const },
  ].filter((item): item is { label: string; value: number; style: "compact" | "plain" } => Boolean(item));

  return (
    <main className="home">
      <NewsTicker items={ticker} />

      <section className="home-hero">
        <div>
          <p className="pill">Open source · Apache-2.0{stats.version ? ` · ${stats.version}` : ""}</p>
          <h1>Infrastructure for continually <em>self‑improving</em> agents</h1>
          <p className="lead">The first <strong>inference-native</strong> infrastructure for agents to learn from <strong>live traffic</strong> and continually update their <strong>weights and harness</strong>.</p>
          <div className="home-actions">
            <Link className="primary-action" href="/docs/getting-started/quickstart">Quickstart <ArrowRight size={16} /></Link>
            <a className="secondary-action" href={siteConfig.repository} target="_blank" rel="noreferrer">
              <GitHubIcon size={16} /> GitHub
              {stats.stars !== null && <span className="action-stars"><Star size={13} aria-hidden="true" />{formatStars(stats.stars)}</span>}
            </a>
          </div>
        </div>
        {heroStats.length > 0 && (
          <dl className="hero-stats">
            {heroStats.map((item) => (
              <div key={item.label}>
                <dd><CountUp value={item.value} style={item.style} /></dd>
                <dt>{item.label}</dt>
              </div>
            ))}
          </dl>
        )}
      </section>
      <section className="hero-stage" aria-label="How Reef improves an agent">
        <LoopDiagram />
      </section>

      <section className="band logo-band">
        <p className="section-label centered">Contributors from</p>
        <LogoWall />
      </section>

      {(stats.commitsByDay.length > 0 || stats.recentPulls.length > 0) && (
        <section className="home-section pulse">
          <div className="pulse-chart">
            <p className="section-label">Activity</p>
            <h2>Recent activity</h2>
            <p className="pulse-caption">
              {commits > 0 && <><strong>{commits}</strong> commits</>}
              {commits > 0 && stats.mergedPulls.recent !== null && " and "}
              {stats.mergedPulls.recent !== null && <><strong>{stats.mergedPulls.recent}</strong> merged pull requests</>}
              {` in ${stats.mergedPulls.windowDays} days.`}
            </p>
            <CommitChart days={stats.commitsByDay} />
            {stats.contributors.length > 0 && (
              <a className="avatar-stack" href={`${siteConfig.repository}/graphs/contributors`} target="_blank" rel="noreferrer">
                {stats.contributors.slice(0, 9).map((person) => (
                  <img key={person.login} src={`${person.avatar}&s=64`} alt="" title={person.login} width={32} height={32} loading="lazy" />
                ))}
                <span>{stats.contributors.length} contributors</span>
              </a>
            )}
          </div>
          {stats.recentPulls.length > 0 && (
            <div className="pulse-feed">
              <p className="section-label"><GitMerge size={13} aria-hidden="true" /> Latest merged</p>
              <ol className="pull-list">
                {stats.recentPulls.slice(0, 5).map((pull) => (
                  <li key={pull.number}>
                    <a href={pull.url} target="_blank" rel="noreferrer">
                      <img src={`${pull.avatar}&s=64`} alt="" width={28} height={28} loading="lazy" />
                      <span className="pull-title">{humanizePullTitle(pull.title)}</span>
                      <span className="pull-meta">#{pull.number} · {pull.author} · {relativeDay(pull.mergedAt, now)}</span>
                    </a>
                  </li>
                ))}
              </ol>
              <a className="text-link" href={`${siteConfig.repository}/pulls?q=is%3Apr+is%3Amerged`} target="_blank" rel="noreferrer">All pull requests <ArrowRight size={15} /></a>
            </div>
          )}
        </section>
      )}

      <section className="band">
        <div className="home-section quick-start">
          <div>
            <p className="section-label">Quickstart</p>
            <h2>Start a local stack</h2>
            <p>Start a stack, send a provider-native request, grade the receipt Reef returns.</p>
            <div className="home-actions">
              <Link className="primary-action" href="/docs/getting-started/quickstart">Open the quickstart <ArrowRight size={16} /></Link>
            </div>
          </div>
          <pre className="terminal"><code>{`export REEF_TOKEN="$(openssl rand -hex 16)"
reef serve -c recipes/basic/local-sglang.yaml

curl http://localhost:8900/healthz`}</code></pre>
          <div className="recipe-strip" aria-label="Bundled recipes">
            {recipeGroups.map((group) => (
              <div className="recipe-row" key={group.title}>
                <div className="recipe-row-label">
                  <strong>{group.title}</strong>
                  <span>{group.note}</span>
                </div>
                <ul>
                  {group.recipes.map((recipe) => (
                    <li key={recipe.name}>
                      {recipe.href.startsWith("http") ? (
                        <a href={recipe.href} title={recipe.signal} target="_blank" rel="noreferrer">
                          <strong>{recipe.name}</strong>
                          <span>{recipe.signal}</span>
                        </a>
                      ) : (
                        <Link href={recipe.href} title={recipe.signal}>
                          <strong>{recipe.name}</strong>
                          <span>{recipe.signal}</span>
                        </Link>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
            <Link className="text-link recipe-strip-link" href="/docs/user-guide/recipes">Compare them all <ArrowRight size={15} /></Link>
          </div>
        </div>
      </section>

      <footer className="site-footer">
        <div>Human-Agent-Society/reef · Apache-2.0</div>
        <nav aria-label="Project links">
          <a href={siteConfig.repository} target="_blank" rel="noreferrer">GitHub</a>
          <a href="/slides/" target="_blank" rel="noreferrer">Slides</a>
          <Link href="/docs/getting-started/intro">Docs</Link>
        </nav>
      </footer>
    </main>
  );
}
