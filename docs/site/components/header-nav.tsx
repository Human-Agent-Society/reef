"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { NavGroup } from "@/lib/docs";

function isCurrent(pathname: string, href: string) {
  return pathname === href || pathname === `${href}/`;
}

// One tab per sidebar group, in reading order. A tab opens the group's first
// page and stays underlined while the reader is on any page in that group,
// so the tabs and the sidebar always agree on where the reader is.
export function HeaderNav({ navigation }: { navigation: NavGroup[] }) {
  const pathname = usePathname();

  return (
    <nav className="header-tabs" aria-label="Primary navigation">
      {navigation.map((group) => {
        const active = group.items.some((item) => isCurrent(pathname, item.href));
        return (
          <Link key={group.title} href={group.items[0].href} className={active ? "active" : undefined} aria-current={active ? "true" : undefined}>
            {group.title}
          </Link>
        );
      })}
    </nav>
  );
}
