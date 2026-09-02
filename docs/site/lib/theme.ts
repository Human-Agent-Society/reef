// Theme helpers shared by the boot script's client-side counterpart. The
// inline script in the layout stamps data-theme on a full page load; these
// re-apply the same choice after a client-side navigation, where the inline
// script never re-runs (the /docs redirect and every sidebar link).
export const THEME_KEY = "reef-theme";
export const MODES = ["auto", "dark", "light"] as const;
export type Mode = (typeof MODES)[number];

export function storedMode(): Mode {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : "auto";
  } catch {
    return "auto";
  }
}

export function resolve(mode: Mode): "light" | "dark" {
  if (mode !== "auto") return mode;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function apply(mode: Mode) {
  document.documentElement.dataset.theme = resolve(mode);
  document.documentElement.dataset.themeMode = mode;
}
