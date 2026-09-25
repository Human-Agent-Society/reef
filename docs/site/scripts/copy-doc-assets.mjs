import { cpSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const target = resolve(siteRoot, "public/doc-assets");
mkdirSync(target, { recursive: true });
cpSync(resolve(siteRoot, "../assets"), target, { recursive: true });
