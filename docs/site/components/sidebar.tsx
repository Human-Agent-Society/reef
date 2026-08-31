"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { NavGroup } from "@/lib/docs";

function isCurrent(pathname: string, href: string) {
  return pathname === href || pathname === `${href}/`;
}

// The sidebar is scoped to the header tab the reader is on: it lists the pages
// of the group that owns the current page, and nothing else. Moving between
// groups is the header's job.
export function Sidebar({ navigation }: { navigation: NavGroup[] }) {
  const pathname = usePathname();
  const group = navigation.find((candidate) => candidate.items.some((item) => isCurrent(pathname, item.href))) ?? navigation[0];

  return (
    <aside className="docs-sidebar">
      <nav aria-label="Documentation navigation">
        <div className="sidebar-group">
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
      </nav>
    </aside>
  );
}
