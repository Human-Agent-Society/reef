"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useEffect, useSyncExternalStore } from "react";
import { MODES, Mode, THEME_KEY, apply, storedMode } from "@/lib/theme";

// Three modes side by side, the current one filled, in the settled order: system, dark, light.
const OPTIONS: { mode: Mode; label: string; Icon: typeof Monitor }[] = [
  { mode: "auto", label: "System color theme", Icon: Monitor },
  { mode: "dark", label: "Dark color theme", Icon: Moon },
  { mode: "light", label: "Light color theme", Icon: Sun },
];

const CHANGE = "reef-theme-change";

function subscribe(onChange: () => void) {
  window.addEventListener(CHANGE, onChange);
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(CHANGE, onChange);
    window.removeEventListener("storage", onChange);
  };
}

export function ThemeToggle() {
  // The server renders system mode; the stored choice replaces it on hydration.
  const mode = useSyncExternalStore(subscribe, storedMode, () => "auto" as Mode);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const followSystem = () => {
      if (storedMode() === "auto") apply("auto");
    };
    media.addEventListener("change", followSystem);
    return () => media.removeEventListener("change", followSystem);
  }, []);

  function choose(next: Mode) {
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
    window.dispatchEvent(new Event(CHANGE));
  }

  // The radio group keys: arrows move, Home and End jump.
  function onKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const index = MODES.indexOf(mode);
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    let target: number;
    if (step !== undefined) target = (index + step + MODES.length) % MODES.length;
    else if (event.key === "Home") target = 0;
    else if (event.key === "End") target = MODES.length - 1;
    else return;
    event.preventDefault();
    choose(MODES[target]);
    (event.currentTarget.querySelector(`[data-mode="${MODES[target]}"]`) as HTMLElement | null)?.focus();
  }

  return (
    <div className="theme-switch" role="radiogroup" aria-label="Color theme" onKeyDown={onKeyDown}>
      {OPTIONS.map(({ mode: option, label, Icon }) => (
        <button
          key={option}
          type="button"
          role="radio"
          data-mode={option}
          aria-checked={mode === option}
          aria-label={label}
          title={label}
          tabIndex={mode === option ? 0 : -1}
          onClick={() => choose(option)}
        >
          <Icon size={15} aria-hidden="true" />
        </button>
      ))}
    </div>
  );
}
