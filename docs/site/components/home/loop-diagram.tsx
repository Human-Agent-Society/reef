import { readFileSync } from "node:fs";
import { join } from "node:path";
import { LoopReplay } from "./loop-replay";

// The hero drawing is the first slide of the deck with its colors mapped to the site's tokens,
// so it follows the theme. It is inlined so the CSS variables and the site fonts reach it.
export function LoopDiagram() {
  const markup = readFileSync(join(process.cwd(), "components", "home", "loop-diagram.svg"), "utf8");
  return <LoopReplay markup={markup} />;
}
