"use client";

import { Moon, Sun, SunMoon } from "lucide-react";
import { useEffect, useLayoutEffect, useRef } from "react";
import { MODES, Mode, THEME_KEY, apply, storedMode } from "@/lib/theme";

// Click order matches the founder's personal-site toggle: an explicit dark,
// then light, then back to following the OS. On a light-OS machine the first
// click must visibly change the page; auto -> light would not.
const LABELS: Record<Mode, string> = {
  auto: "Color theme: system. Switch to dark.",
  dark: "Color theme: dark. Switch to light.",
  light: "Color theme: light. Switch to system.",
};

export function ThemeToggle() {
  const button = useRef<HTMLButtonElement>(null);

  function label(mode: Mode) {
    const el = button.current;
    if (!el) return;
    el.setAttribute("aria-label", LABELS[mode]);
    el.title = LABELS[mode];
  }

  // The server renders the system-mode label; sync to storage before paint.
  useLayoutEffect(() => label(storedMode()), []);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const followSystem = () => {
      if (storedMode() === "auto") apply("auto");
    };
    media.addEventListener("change", followSystem);
    return () => media.removeEventListener("change", followSystem);
  }, []);

  function cycleTheme() {
    const next = MODES[(MODES.indexOf(storedMode()) + 1) % MODES.length];
    try {
      if (next === "auto") {
        localStorage.removeItem(THEME_KEY);
      } else {
        localStorage.setItem(THEME_KEY, next);
      }
    } catch {
      /* private browsing: the theme still applies for this page view */
    }
    apply(next);
    label(next);
  }

  return (
    <button
      ref={button}
      className="icon-button theme-toggle"
      type="button"
      onClick={cycleTheme}
      aria-label={LABELS.auto}
      title={LABELS.auto}
    >
      <SunMoon className="system-icon" size={18} />
      <Moon className="moon-icon" size={18} />
      <Sun className="sun-icon" size={18} />
    </button>
  );
}
