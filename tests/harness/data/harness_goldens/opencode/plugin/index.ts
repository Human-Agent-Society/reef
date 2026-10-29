import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const json = (name: string) => (existsSync(join(here, name)) ? JSON.parse(readFileSync(join(here, name), "utf8")) : {});

function merge(base: any, overlay: any): any {
  const objects = [base, overlay].every((value) => value && typeof value === "object" && !Array.isArray(value));
  if (!objects) return overlay;
  const merged = { ...base };
  for (const [key, value] of Object.entries(overlay)) merged[key] = key in base ? merge(base[key], value) : value;
  return merged;
}

export const ReefHarness = async (input: any) => {
  const directory = join(here, "extensions");
  const files = existsSync(directory) ? readdirSync(directory).filter((name) => name.endsWith(".ts")).sort() : [];
  const hooks: any[] = [];
  for (const name of files) {
    const module = await import(pathToFileURL(join(directory, name)).href);
    for (const plugin of Object.values(module)) if (typeof plugin === "function") hooks.push(await plugin(input));
  }
  const merged: any = { tool: Object.assign({}, ...hooks.map((hook) => hook.tool ?? {})) };
  for (const key of new Set(hooks.flatMap((hook) => Object.keys(hook)))) {
    if (key === "tool" || key === "config") continue;
    const values = hooks.map((hook) => hook[key]).filter((value) => value !== undefined);
    merged[key] = values.every((value) => typeof value === "function")
      ? async (...args: any[]) => { for (const value of values) await value(...args); }
      : values[0];
  }
  merged.config = async (config: any) => {
    Object.assign(config, merge(config, json("config.json")));
    if (existsSync(join(here, "AGENTS.md"))) config.instructions = [...(config.instructions ?? []), join(here, "AGENTS.md")];
    config.command = { ...(config.command ?? {}), ...json("commands.json") };
    config.skills = { ...(config.skills ?? {}), paths: [...(config.skills?.paths ?? []), join(here, "skill")] };
    for (const hook of hooks) if (typeof hook.config === "function") await hook.config(config);
  };
  return merged;
};
