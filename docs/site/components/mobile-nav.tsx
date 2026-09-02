"use client";

import Link from "next/link";
import { Menu, X } from "lucide-react";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { NavGroup } from "@/lib/docs";

function isCurrent(pathname: string, href: string) {
  return pathname === href || pathname === `${href}/`;
}

// The phone-width stand-in for the header tabs and the docs sidebar, which
// are both hidden below 800px: one button, one slide-over with the full tree.
export function MobileNav({ navigation }: { navigation: NavGroup[] }) {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const panelRef = useRef<HTMLDivElement>(null);
  const restoreFocus = useRef<HTMLElement | null>(null);

  // Adjust-during-render: a route change (back gesture included) closes the
  // drawer without an effect pass.
  const [lastPathname, setLastPathname] = useState(pathname);
  if (pathname !== lastPathname) {
    setLastPathname(pathname);
    setOpen(false);
  }

  useEffect(() => {
    if (!open) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    function onTrap(event: KeyboardEvent) {
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = panelRef.current.querySelectorAll<HTMLElement>('a[href], button:not([disabled])');
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keydown", onTrap);
    // Focus the drawer (its close button) on open; return focus on close.
    restoreFocus.current = document.activeElement as HTMLElement | null;
    requestAnimationFrame(() => panelRef.current?.querySelector<HTMLElement>("button, a[href]")?.focus());
    // Scroll lock: without it, scrolling past the end of the nav list chains
    // into the page, and closing the drawer strands the reader mid-document.
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keydown", onTrap);
      document.body.style.overflow = previousOverflow;
      restoreFocus.current?.focus();
    };
  }, [open]);

  return (
    <>
      <button className="icon-button menu-toggle" type="button" aria-label="Open navigation" aria-expanded={open} onClick={() => setOpen(true)}>
        <Menu size={19} />
      </button>

      {/* Portal: the fixed overlay must escape the header, whose
          backdrop-filter makes it the containing block for fixed children. */}
      {open && createPortal(
        <div className="mobile-nav-overlay" role="presentation" onMouseDown={() => setOpen(false)}>
          <div ref={panelRef} className="mobile-nav-panel" role="dialog" aria-modal="true" aria-label="Site navigation" onMouseDown={(event) => event.stopPropagation()}>
            <div className="mobile-nav-head">
              <span>Documentation</span>
              <button className="icon-button" type="button" aria-label="Close navigation" onClick={() => setOpen(false)}><X size={18} /></button>
            </div>
            <nav aria-label="Primary navigation">
              {navigation.map((group) => (
                <section key={group.title}>
                  <h3>{group.title}</h3>
                  {group.items.map((item) => (
                    <Link key={item.href} href={item.href} aria-current={isCurrent(pathname, item.href) ? "page" : undefined} onClick={() => setOpen(false)}>
                      {item.title}
                    </Link>
                  ))}
                </section>
              ))}
            </nav>
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}
