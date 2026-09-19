import type { NewsItem } from "@/lib/news";

function Item({ item }: { item: NewsItem }) {
  const external = item.href.startsWith("http");
  return (
    <a className={`ticker-item ${item.kind}`} href={item.href} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>
      {item.kind === "release" && <span className="ticker-tag">Release</span>}
      <span>{item.text}</span>
      <time dateTime={item.date}>{new Date(item.date).toLocaleDateString("en", { month: "short", day: "numeric", timeZone: "UTC" })}</time>
    </a>
  );
}

// One continuous strip; the track is rendered twice so the loop has no seam. Hover pauses it.
export function NewsTicker({ items }: { items: NewsItem[] }) {
  if (items.length === 0) return null;
  return (
    <div className="news-ticker" aria-label="Project news">
      <span className="ticker-label"><span className="live-dot" aria-hidden="true" />News</span>
      <div className="ticker-viewport">
        <div className="ticker-track">
          {items.map((item) => <Item key={`a-${item.kind}-${item.href}`} item={item} />)}
          {items.map((item) => <Item key={`b-${item.kind}-${item.href}`} item={item} />)}
        </div>
      </div>
    </div>
  );
}
