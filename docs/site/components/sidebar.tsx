"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { NavGroup } from "@/lib/docs";

function isCurrent(pathname: string, href: string) {
  return pathname === href || pathname === `${href}/`;
}

// The whole tree, every group with its pages, so a reader sees where a page
// sits among the rest and reaches any page in one click. The header tabs
// jump between groups; the sidebar scrolls on its own.
export function Sidebar({ navigation }: { navigation: NavGroup[] }) {
  const pathname = usePathname();

  return (
    <aside className="docs-sidebar">
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
