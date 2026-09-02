"use client";

import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";

// A keyboard user's focus is on a link in the old page when a client-side
// navigation swaps the content; without this it falls to <body> and the next
// Tab restarts from the top of the page. Move focus to the main region on
// every route change after the first, so reading continues where the new
// page begins. The skip link and heading anchors stay reachable from there.
export function RouteFocus() {
  const pathname = usePathname();
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const main = document.querySelector<HTMLElement>("main.docs-main, main");
    main?.focus({ preventScroll: true });
  }, [pathname]);
  return null;
}
