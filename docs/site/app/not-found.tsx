import Link from "next/link";

export default function NotFound() {
  return (
    <main className="not-found"><span>404</span><h1>This page drifted beyond the reef.</h1><p>The documentation you requested could not be found.</p><Link className="button primary" href="/docs">Back to documentation</Link></main>
  );
}
