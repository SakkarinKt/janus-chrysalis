/**
 * `npm run repro` — re-run something that already has a recorded result and say whether it matches.
 *
 *   npm run repro -- <ledger id>                                  re-run a harness result at HEAD
 *   npm run repro -- artifacts/<card>/<run>/manifest.json         same, by manifest path
 *   npm run repro -- artifacts/<run>/seed-1/manifest.json --at-commit
 *                                                                 re-run a pre-harness script at
 *                                                                 the commit it was recorded at
 *
 *   --at-commit            required for pre-harness (v1) manifests; uses a temporary detached worktree
 *   --tol <n>              largest allowed difference per number (default 0: byte-identical or it fails)
 *   --scratch              compare and report without recording; works on a dirty tree
 *   --allow-dirty          same as --scratch
 *   --budget-minutes <n>   pre-harness runs only: time limit for the script (default 30)
 *   --script <path>        pre-harness runs only: when the script is not experiments/<run>/run.ts
 *   --require-match        exit 1 on a mismatch (for CI)
 *
 * What is compared: every `*.summary.csv` the reference run committed, plus (for harness results)
 * the primary value and the verdict. The outcome is recorded as a manifest under artifacts/repro/
 * and one `repro` line in experiments/ledger.jsonl.
 *
 * Exit codes: 0 the comparison completed (match or not) · 1 refused, the re-run produced no
 * result, or --require-match and it did not match · 2 bad usage.
 */
import { relative } from "node:path";
import { listCards } from "../src/experiment/card.ts";
import { HarnessError, harnessPaths } from "../src/experiment/harness.ts";
import { readLedger } from "../src/experiment/ledger.ts";
import { reproduce } from "../src/experiment/repro.ts";

function usage(message: string): never {
  console.error(`repro: ${message}\nRead the header of scripts/repro.ts for usage.`);
  process.exit(2);
}

function numberOption(name: string, value: string | undefined, minimum: number): number {
  const parsed = Number(value);
  if (value === undefined || value.trim() === "" || !Number.isFinite(parsed) || parsed < minimum) {
    usage(`${name} needs a number >= ${minimum}`);
  }
  return parsed;
}

let target: string | null = null;
let atCommit = false;
let scratch = false;
let requireMatch = false;
let tolerance = 0;
let budgetMinutes = 30;
let script: string | undefined;

const argv = process.argv.slice(2);
for (let index = 0; index < argv.length; index++) {
  const arg = argv[index] ?? "";
  switch (arg) {
    case "--at-commit":
      atCommit = true;
      break;
    case "--scratch":
    case "--allow-dirty":
      scratch = true;
      break;
    case "--require-match":
      requireMatch = true;
      break;
    case "--tol":
      tolerance = numberOption("--tol", argv[++index], 0);
      break;
    case "--budget-minutes":
      budgetMinutes = numberOption("--budget-minutes", argv[++index], 0.001);
      break;
    case "--script":
      script = argv[++index];
      if (script === undefined) {
        usage("--script needs a path");
      }
      break;
    default:
      if (arg.startsWith("-")) {
        usage(`unknown option ${arg}`);
      }
      if (target !== null) {
        usage("give one target at a time");
      }
      target = arg;
  }
}
if (target === null) {
  usage("give a ledger id or the path of a manifest.json");
}

const paths = harnessPaths();
try {
  const outcome = reproduce(target, listCards(paths.cardsDir), readLedger(paths.ledgerPath), paths, {
    tolerance,
    atCommit,
    scratch,
    budgetMinutes,
    command: `npm run repro -- ${argv.join(" ")}`,
    ...(script === undefined ? {} : { script }),
  });
  const { comparison } = outcome;
  console.log(`\n${outcome.of} → ${comparison.match ? "MATCH" : "MISMATCH"}${comparison.identical ? " (byte-identical)" : ""}`);
  console.log(`  summary files compared: ${comparison.filesCompared}`);
  console.log(`  largest difference: ${comparison.maxDifference} (tolerance ${comparison.tolerance})`);
  for (const mismatch of comparison.mismatches) {
    const where = mismatch.line === 0 ? mismatch.file : `${mismatch.file}:${mismatch.line}:${mismatch.column}`;
    console.log(`  ${where}: ${mismatch.detail}`);
  }
  if (comparison.mismatchCount > comparison.mismatches.length) {
    console.log(`  … and ${comparison.mismatchCount - comparison.mismatches.length} more`);
  }
  console.log(`  manifest: ${relative(paths.root, outcome.outDir)}/manifest.json`);
  console.log(outcome.ledgerEntry === null ? "  not recorded (scratch)" : `  recorded: ${outcome.ledgerEntry.ledgerId}`);
  process.exit(requireMatch && !comparison.match ? 1 : 0);
} catch (error) {
  if (error instanceof HarnessError || (error instanceof Error && error.name === "CardError")) {
    console.error(`repro: ${error.message}`);
    process.exit(1);
  }
  throw error;
}
