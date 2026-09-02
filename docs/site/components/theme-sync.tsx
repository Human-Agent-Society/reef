"use client";

import { usePathname } from "next/navigation";
import { useEffect } from "react";
import { apply, storedMode } from "@/lib/theme";

// The inline boot script stamps the theme on a full page load, but a
// client-side navigation (the /docs redirect, a sidebar link) never re-runs
// it, so a page reached that way would keep the server-rendered default.
// Re-apply the stored choice on mount and on every path change.
export function ThemeSync() {
  const pathname = usePathname();
  useEffect(() => apply(storedMode()), [pathname]);
  return null;
}
