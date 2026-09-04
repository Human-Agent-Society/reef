"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";
import type { NavGroup } from "@/lib/docs";

function isCurrent(pathname: string, href: string) {
  return pathname === href || pathname === `${href}/`;
}

// The whole tree, every section with its pages; the tabs jump between sections and the sidebar scrolls on its own.
export function Sidebar({ navigation }: { navigation: NavGroup[] }) {
  const pathname = usePathname();
  const aside = useRef<HTMLElement>(null);

  // The tree is longer than the viewport, so the current page is brought into view on every page change.
  useEffect(() => {
    const active = aside.current?.querySelector<HTMLElement>('a[aria-current="page"]');
    active?.scrollIntoView({ block: "nearest" });
  }, [pathname]);

  return (
    <aside ref={aside} className="docs-sidebar">
      <nav aria-label="Documentation navigation">
        {navigation.map((group) => (
          <div key={group.title} className="sidebar-group">
            <p className="sidebar-group-title">{group.title}</p>
            <div className="sidebar-group-items">
              {group.items.map((item) => {
                const active = isCurrent(pathname, item.href);
                return (
                  <Link key={item.href} href={item.href} className={active ? "active" : undefined} aria-current={active ? "page" : undefined}>
                    {item.title}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>
    </aside>
  );
}
