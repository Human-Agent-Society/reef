"use client";

import { Moon, Sun, SunMoon } from "lucide-react";
import { useEffect, useLayoutEffect, useRef } from "react";

const MODES = ["auto", "light", "dark"] as const;
type Mode = (typeof MODES)[number];

const LABELS: Record<Mode, string> = {
  auto: "Color theme: system. Switch to light.",
  light: "Color theme: light. Switch to dark.",
  dark: "Color theme: dark. Switch to system.",
};

function storedMode(): Mode {
  try {
    const value = localStorage.getItem("reef-theme");
    return value === "light" || value === "dark" ? value : "auto";
  } catch {
    return "auto";
  }
}

function resolve(mode: Mode): "light" | "dark" {
  if (mode !== "auto") return mode;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function apply(mode: Mode) {
  document.documentElement.dataset.theme = resolve(mode);
  document.documentElement.dataset.themeMode = mode;
}

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
      localStorage.setItem("reef-theme", next);
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
