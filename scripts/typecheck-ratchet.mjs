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

// PR #32 review (2026-07-30): don't gate on tsc's exit code — it's 2 both at the 43-error
// baseline and when tsc never ran at all (e.g. a broken tsconfig), so a swallowed spawn error
// would otherwise print "0 errors" and exit 0. Gate on the contradiction instead: spawnSync
// failed outright, or tsc reported non-zero status but the line-count parse found nothing.
if (result.error || result.status === null) {
  console.error(`Failed to run tsc: ${result.error ?? "process did not complete"}`);
  process.exit(1);
}

const inRepoLines = output
  .split("\n")
  .filter((line) => line.includes("error TS") && !line.includes("node_modules"));
const inRepoCount = inRepoLines.length;

console.log(`\nIn-repo typecheck errors: ${inRepoCount} (baseline: ${BASELINE_IN_REPO_ERRORS})`);

if (result.status !== 0 && inRepoCount === 0) {
  console.error(
    "tsc exited non-zero but no in-repo error lines were parsed from its output — the parse " +
      "is probably broken (output format changed) rather than the repo being clean. Failing " +
      "closed instead of reporting a false 0.",
  );
  process.exit(1);
}

// PR #33 review (2026-07-31): a broken tsconfig (e.g. an `include` that matches no files) makes
// tsc exit 2 and emit a single config-level diagnostic (TS18003, or a bad `target`/malformed-JSON
// error reported against tsconfig.json itself) instead of per-file `file(line,col): error TSxxxx`
// lines. That diagnostic still contains "error TS", so it's counted into inRepoCount as 1 (or a
// few) — comfortably under the 43 baseline, so the ratchet reports success even though zero files
// were actually typechecked. Fail closed on any such line, regardless of count: config/CLI-level
// diagnostics either name tsconfig.json as the offending file or (like TS18003) omit the
// `file(line,col):` location prefix entirely that every per-file error carries.
const configLevelLine = inRepoLines.find(
  (line) => line.includes("tsconfig.json") || /^error TS\d+:/.test(line.trim()),
);
if (configLevelLine) {
  console.error(
    `tsc reported a config-level diagnostic, meaning typechecking may not have actually run: ${configLevelLine.trim()}`,
  );
  process.exit(1);
}

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
