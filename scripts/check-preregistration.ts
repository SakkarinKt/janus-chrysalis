/**
 * Pre-registration check. Runs in CI's `lint` job; run it by hand with
 *
 *   node scripts/check-preregistration.ts [--base <ref>]
 *
 * It fails if, relative to the base ref, the results ledger lost or altered a line, a recorded
 * manifest was edited, a merged card changed, or a new result was produced by a card or code
 * that was not already on the base. The rules and the reasons for them are in
 * `src/experiment/preregistration.ts`; how to work within them is in `experiments/cards/README.md`.
 *
 * Base ref: `--base` if given; else `origin/$GITHUB_BASE_REF` on a pull request; else
 * `origin/main`. Needs full history (CI: `fetch-depth: 0`).
 *
 * Exit codes: 0 no problems · 1 problems found · 2 cannot check (bad usage, shallow clone,
 * unresolvable base).
 */
import { execFileSync } from "node:child_process";
import { checkPreregistration } from "../src/experiment/preregistration.ts";

function cannotCheck(message: string): never {
  console.error(`check-preregistration: ${message}`);
  process.exit(2);
}

const argv = process.argv.slice(2);
let base: string | undefined;
for (let index = 0; index < argv.length; index++) {
  if (argv[index] === "--base") {
    base = argv[++index];
    if (base === undefined || base === "") {
      cannotCheck("--base needs a ref");
    }
  } else {
    cannotCheck(`unknown argument ${argv[index]}`);
  }
}
const prBase = process.env.GITHUB_BASE_REF;
const baseRef = base ?? (prBase !== undefined && prBase !== "" ? `origin/${prBase}` : "origin/main");

const root = process.cwd();
const shallow = execFileSync("git", ["rev-parse", "--is-shallow-repository"], { cwd: root, encoding: "utf8" }).trim();
if (shallow === "true") {
  cannotCheck("this is a shallow clone, so ancestry cannot be checked. Fetch full history (CI: fetch-depth: 0).");
}

let report;
try {
  report = checkPreregistration(root, baseRef);
} catch (error) {
  cannotCheck((error as Error).message);
}

for (const problem of report.problems) {
  console.error(`${problem.rule}  ${problem.where}: ${problem.detail}`);
}
const summary = `${report.ledgerLines} ledger line(s), ${report.newLines} new since ${baseRef}; ${report.cardsOnBase} card(s) on ${baseRef}`;
if (report.problems.length > 0) {
  console.error(`\ncheck-preregistration: ${report.problems.length} problem(s). ${summary}. See experiments/cards/README.md.`);
  process.exit(1);
}
console.log(`check-preregistration: ok. ${summary}.`);
