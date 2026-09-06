"use client";

import { AlignLeft } from "lucide-react";
import { useEffect, useState } from "react";
import type { TocItem } from "@/lib/docs";

export function TableOfContents({ items }: { items: TocItem[] }) {
  const [active, setActive] = useState(items[0]?.id ?? "");

  useEffect(() => {
    const headings = items.map((item) => document.getElementById(item.id)).filter(Boolean) as HTMLElement[];
    if (!headings.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]?.target.id) setActive(visible[0].target.id);
      },
      { rootMargin: "-120px 0px -72%", threshold: [0, 1] },
    );
    headings.forEach((heading) => observer.observe(heading));
    return () => observer.disconnect();
  }, [items]);

  if (!items.length) return null;

  return (
    <aside className="toc">
      <p className="toc-label"><AlignLeft size={15} aria-hidden="true" />On this page</p>
      <nav aria-label="Table of contents">
        {items.map((item) => (
          <a key={item.id} href={`#${item.id}`} className={`${item.level === 3 ? "nested " : ""}${active === item.id ? "active" : ""}`}>
            {item.title}
          </a>
        ))}
      </nav>
    </aside>
  );
}
