// Ratchet for `npm run typecheck`'s non-gating CI step (PR #30 review, 2026-07-29).
// `tsc --noEmit` currently reports 43 pre-existing in-repo diagnostics (noUncheckedIndexedAccess
// strictness, mostly) that this repo hasn't fixed yet, so the raw exit code can't gate CI without
// blocking every PR. This script fails only when the in-repo count grows past the baseline, so new
// errors can't land silently while the pre-existing ones stay non-blocking.
//
// Scope note: `tsconfig.json`'s `include` is `["src", "test"]`, so `experiments/` is not counted
// here at all (PR #30 review) — it needs its own spot-check, e.g. `npx tsc --noEmit experiments/**/*.ts`.
import { spawnSync } from "node:child_process";

const BASELINE_IN_REPO_ERRORS = 43;

const result = spawnSync("npx", ["tsc", "--noEmit"], { encoding: "utf8" });
const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
process.stdout.write(output);

const inRepoCount = output
  .split("\n")
  .filter((line) => line.includes("error TS") && !line.includes("node_modules")).length;

console.log(`\nIn-repo typecheck errors: ${inRepoCount} (baseline: ${BASELINE_IN_REPO_ERRORS})`);

if (inRepoCount > BASELINE_IN_REPO_ERRORS) {
  console.error(
    `Regression: in-repo typecheck errors rose from ${BASELINE_IN_REPO_ERRORS} to ${inRepoCount}. ` +
      "Fix the new error(s), or if the increase is intentional, lower/raise BASELINE_IN_REPO_ERRORS " +
      "in this script in the same PR that changes the count.",
  );
  process.exit(1);
}

if (inRepoCount < BASELINE_IN_REPO_ERRORS) {
  console.log(`Baseline could be lowered to ${inRepoCount} once that's confirmed stable.`);
}
