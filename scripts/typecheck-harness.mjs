// `npm run typecheck:harness` — the harness's own code must typecheck with zero errors.
//
// `tsconfig.json` covers `src/` and `test/` and is gated by a ratchet (typecheck-ratchet.mjs)
// because of a pre-existing backlog. `scripts/` and `experiments/cards/` are new code with no
// backlog, so they get a hard gate instead: `tsconfig.harness.json` adds them to the program, and
// this script fails on any diagnostic located in a file under those two directories.
//
// Diagnostics in files those scripts merely import (the `src/` backlog) are ignored here; the
// ratchet owns them. The frozen `experiments/<date>-*/run.ts` scripts stay outside every
// typecheck on purpose (`experiments/README.md`).
import { spawnSync } from "node:child_process";

const OWNED_PREFIXES = ["scripts/", "experiments/cards/"];

const result = spawnSync("npx", ["tsc", "--noEmit", "-p", "tsconfig.harness.json"], { encoding: "utf8" });
if (result.error || result.status === null) {
  console.error(`Failed to run tsc: ${result.error ?? "process did not complete"}`);
  process.exit(1);
}
const lines = `${result.stdout ?? ""}${result.stderr ?? ""}`.split("\n").filter((line) => line.includes("error TS"));

// Same fail-closed rule as the ratchet: a config-level diagnostic (no `file(line,col):` prefix,
// or one located in a tsconfig) means the program may not have been checked at all.
const configLevel = lines.find((line) => {
  const trimmed = line.trim();
  return /^[^(]*tsconfig[^(]*\.json\(\d+,\d+\):/.test(trimmed) || /^error TS\d+:/.test(trimmed);
});
if (configLevel) {
  console.error(`tsc reported a config-level diagnostic, so typechecking may not have run: ${configLevel.trim()}`);
  process.exit(1);
}
if (result.status !== 0 && lines.length === 0) {
  console.error("tsc exited non-zero but no diagnostics were parsed from its output. Failing closed.");
  process.exit(1);
}

const owned = lines.filter((line) => OWNED_PREFIXES.some((prefix) => line.trim().startsWith(prefix)));
for (const line of owned) {
  console.error(line);
}
console.log(`Harness typecheck errors (${OWNED_PREFIXES.join(", ")}): ${owned.length} (must be 0)`);
process.exit(owned.length === 0 ? 0 : 1);
