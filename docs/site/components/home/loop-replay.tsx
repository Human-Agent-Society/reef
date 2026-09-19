"use client";

import { useEffect, useState } from "react";

const CYCLE_MS = 15000;

// The drawing animates once when mounted; remounting it every cycle plays it again.
export function LoopReplay({ markup }: { markup: string }) {
  const [cycle, setCycle] = useState(0);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const timer = window.setInterval(() => setCycle((n) => n + 1), CYCLE_MS);
    return () => window.clearInterval(timer);
  }, []);

  return <div key={cycle} className="hero-visual" dangerouslySetInnerHTML={{ __html: markup }} />;
}
