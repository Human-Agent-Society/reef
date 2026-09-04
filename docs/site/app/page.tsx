import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, Boxes, PlugZap } from "lucide-react";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = {
  title: { absolute: siteConfig.title },
  description: siteConfig.description,
  alternates: { canonical: "/" },
  openGraph: { title: siteConfig.title, description: siteConfig.description, url: "/" },
};

const documentationLinks = [
  { title: "Quickstart", href: "/docs/getting-started/quickstart" },
  { title: "Installation", href: "/docs/getting-started/installation" },
  { title: "Recipes", href: "/docs/user-guide/recipes" },
  { title: "Harness Evolution", href: "/docs/user-guide/evolve-your-harness" },
  { title: "API Reference", href: "/docs/reference/http-api" },
];

const surfaces = [
  {
    eyebrow: "No GPU",
    title: "Evolve the harness",
    text: "Reef proposes an edit to your agent's rules, skills, config, tools, or loop, runs the agent both ways on your tasks, and publishes the winner. Eight coding agents, including Reef's own.",
    href: "/docs/user-guide/evolve-your-harness",
    cta: "Harness evolution",
    icon: PlugZap,
  },
  {
    eyebrow: "GPUs",
    title: "Evolve the weights",
    text: "Train from live feedback and hot-swap each accepted version into the serving engine. Ray and Slime run the optimizer.",
    href: "/docs/user-guide/evolve-your-model",
    cta: "Weight training",
    icon: Boxes,
  },
];

const methods = [
  { name: "sao", href: "/docs/user-guide/recipes/sao", signal: "Feedback on each attempt, over a stream of tasks", evolves: "model weights", gpu: "yes" },
  { name: "tttd", href: "/docs/user-guide/recipes/tttd", signal: "A fixed grid of sibling attempts at one problem", evolves: "model weights", gpu: "yes" },
  { name: "openclawrl", href: "/docs/user-guide/recipes/openclawrl", signal: "Nothing on the wire — just agent conversations", evolves: "model weights", gpu: "yes" },
  { name: "skillclaw", href: "/docs/user-guide/recipes/skillclaw", signal: "Scores on requests, and failures worth learning from", evolves: "harness tree", gpu: "no" },
  { name: "gepa", href: "/docs/user-guide/recipes/gepa", signal: "Scores and transcripts to reflect on", evolves: "harness tree", gpu: "no" },
];

const steps = [
  "Serve the request, forwarded to the model unchanged",
  "Record the exchange and the release behind it",
  "Learn: turn eligible feedback into a candidate update",
  "Publish the accepted candidate and serve it next",
];

const shelves = [
  {
    title: "Getting Started",
    links: [
      ["Introduction", "/docs/getting-started/intro"],
      ["Installation", "/docs/getting-started/installation"],
      ["Quickstart", "/docs/getting-started/quickstart"],
      ["Architecture", "/docs/getting-started/architecture"],
    ],
  },
  {
    title: "User Guide",
    links: [
      ["Choosing a recipe", "/docs/user-guide/recipes"],
      ["Evolve your harness", "/docs/user-guide/evolve-your-harness"],
      ["Evolve your model", "/docs/user-guide/evolve-your-model"],
    ],
  },
  {
    title: "Developer Guide",
    links: [
      ["Write a recipe", "/docs/developer-guide/write-a-recipe"],
      ["Write a harness method", "/docs/developer-guide/write-a-harness-method"],
      ["Harness adapters", "/docs/developer-guide/harness-adapters"],
      ["Loss families", "/docs/developer-guide/loss-families"],
    ],
  },
  {
    title: "Reference",
    links: [
      ["CLI", "/docs/reference/cli"],
      ["Configuration", "/docs/reference/configuration"],
      ["HTTP API", "/docs/reference/http-api"],
      ["Python API", "/docs/reference/python-api"],
      ["Glossary", "/docs/reference/glossary"],
    ],
  },
  {
    title: "Contributing",
    links: [
      ["Codebase structure", "/docs/contributing/codebase-structure"],
      ["Adding components", "/docs/contributing/adding-components"],
      ["Development", "/docs/contributing/development"],
      ["Testing", "/docs/contributing/testing"],
    ],
  },
];

export default function Home() {
  return (
    <main className="home">
      <section className="home-hero">
        <div>
          <p className="pill">Reef by Human-Agent Society</p>
          <h1>Continuous post-training for AI agents</h1>
          <p className="lead">{siteConfig.description}</p>
          <div className="home-actions">
            <Link className="primary-action" href="/docs/getting-started/quickstart">Quickstart</Link>
            <a className="secondary-action" href={siteConfig.repository}>View on GitHub</a>
          </div>
          <nav className="home-doc-links" aria-label="Documentation shortcuts">
            {documentationLinks.map((link) => (
              <Link key={link.href} href={link.href}>{link.title}</Link>
            ))}
          </nav>
        </div>
        <aside className="loop-card">
          <p>The Reef loop</p>
          <pre><code>{`serve inference
  ↓
record the receipt
  ↓
report feedback
  ↓
publish the next version
  ↺`}</code></pre>
        </aside>
      </section>

      <section className="band">
        <div className="home-section quick-start">
          <div>
            <p className="section-label">Quickstart</p>
            <h2>Send a request, grade it</h2>
            <p>Start a stack, send a provider-native request, and report a score against the receipt Reef returns.</p>
            <Link className="primary-action" href="/docs/getting-started/quickstart">Run the loop</Link>
          </div>
          <pre><code>{`export REEF_TOKEN="$(openssl rand -hex 16)"
reef serve -c recipes/basic/local-sglang.yaml

curl http://localhost:8900/healthz`}</code></pre>
        </div>
      </section>

      <section className="home-section">
        <div className="section-heading-row">
          <h2>Train weights or improve the harness</h2>
          <p>Use a supported GPU stack to train model weights, or optimize prompts, skills, and rules with a hosted model.</p>
        </div>
        <div className="path-grid path-grid-2">
          {surfaces.map((item) => {
            const Icon = item.icon;
            return (
              <Link className="path-card" href={item.href} key={item.title}>
                <span className="path-card-icon"><Icon size={20} aria-hidden="true" /></span>
                <small>{item.eyebrow}</small>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
                <span className="path-card-link">{item.cta} <ArrowRight size={15} /></span>
              </Link>
            );
          })}
        </div>
      </section>

      <section className="band">
        <div className="home-section">
          <div className="section-heading-row">
            <h2>Bundled methods</h2>
            <p>Each method is one package with its recipe, processor, preparer, and runnable examples.</p>
          </div>
          <div className="table-scroll">
            <table className="home-table">
              <thead>
                <tr><th>Recipe</th><th>The signal you have</th><th>Evolves</th><th>GPUs</th></tr>
              </thead>
              <tbody>
                {methods.map((m) => (
                  <tr key={m.name}>
                    <td><Link href={m.href}><code>{m.name}</code></Link></td>
                    <td>{m.signal}</td>
                    <td>{m.evolves}</td>
                    <td>{m.gpu}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Link className="text-link" href="/docs/user-guide/recipes">Compare them all <ArrowRight size={15} /></Link>
        </div>
      </section>

      <section className="home-section process-section" id="learning-loop">
        <div>
          <h2>How Reef learns</h2>
          <p>Your harness keeps its prompts, tools, environments, and graders. Reef records which release served each response.</p>
          <Link className="text-link" href="/docs/getting-started/architecture">Read the architecture <ArrowRight size={15} /></Link>
        </div>
        <ol className="step-list">
          {steps.map((step, index) => <li key={step}><span>{index + 1}</span>{step}</li>)}
        </ol>
      </section>

      <section className="home-section resource-section">
        <h2>All documentation</h2>
        <div className="shelf-grid">
          {shelves.map((shelf) => (
            <div className="shelf" key={shelf.title}>
              <h3>{shelf.title}</h3>
              <ul>
                {shelf.links.map(([label, href]) => (
                  <li key={href}><Link href={href}>{label}</Link></li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
