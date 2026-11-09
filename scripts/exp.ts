/**
 * `npm run exp` — run an experiment card and record its verdict.
 *
 *   npm run exp -- --list            every card and where it stands
 *   npm run exp -- 0001              run card 0001 and record the result
 *   npm run exp -- --next            run the lowest-numbered card that is ready
 *
 *   --scratch         run without recording: any tree state, output under artifacts/_scratch/
 *   --allow-dirty     same as --scratch
 *   --seeds 1,2,3     override the card's seeds (implies --scratch)
 *   --dry-run         say what would run and stop
 *   --require-pass    exit 1 when the verdict is "fail" (for CI canaries)
 *
 * A recorded run needs a clean tree, writes artifacts/<card>/<runId>/manifest.json and appends
 * one line to experiments/ledger.jsonl. A confirmatory or calibration card is recorded once.
 *
 * Exit codes: 0 the run completed (whatever the verdict), or there was nothing to run ·
 * 1 the run was refused or produced no result, or --require-pass and the verdict was "fail" ·
 * 2 bad usage.
 */
import { relative } from "node:path";
import { findCard, listCards } from "../src/experiment/card.ts";
import type { LoadedCard } from "../src/experiment/card.ts";
import { HarnessError, executeCard, harnessPaths } from "../src/experiment/harness.ts";
import { cardStatus, describeStatus, nextCard, readLedger } from "../src/experiment/ledger.ts";

interface Args {
  target: string | null;
  next: boolean;
  list: boolean;
  scratch: boolean;
  dryRun: boolean;
  requirePass: boolean;
  seeds: number[] | undefined;
}

function usage(message: string): never {
  console.error(`exp: ${message}\nRun \`npm run exp -- --list\` to see the cards, or read the header of scripts/exp.ts.`);
  process.exit(2);
}

function parseArgs(argv: string[]): Args {
  const args: Args = { target: null, next: false, list: false, scratch: false, dryRun: false, requirePass: false, seeds: undefined };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index] ?? "";
    switch (arg) {
      case "--next":
        args.next = true;
        break;
      case "--list":
        args.list = true;
        break;
      case "--scratch":
      case "--allow-dirty":
        args.scratch = true;
        break;
      case "--dry-run":
        args.dryRun = true;
        break;
      case "--require-pass":
        args.requirePass = true;
        break;
      case "--seeds": {
        const value = argv[++index];
        const seeds = (value ?? "").split(",").map((part) => Number(part));
        if (value === undefined || seeds.some((seed) => !Number.isSafeInteger(seed) || seed < 0)) {
          usage("--seeds needs a comma-separated list of non-negative integers");
        }
        args.seeds = seeds;
        args.scratch = true;
        break;
      }
      default:
        if (arg.startsWith("-")) {
          usage(`unknown option ${arg}`);
        }
        if (args.target !== null) {
          usage("give one card at a time");
        }
        args.target = arg;
    }
  }
  if (args.list) {
    return args;
  }
  if (args.next === (args.target !== null)) {
    usage("give either a card id or --next");
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const paths = harnessPaths();

try {
  const cards = listCards(paths.cardsDir);
  const ledger = readLedger(paths.ledgerPath);

  if (args.list) {
    if (cards.length === 0) {
      console.log(`No cards under ${relative(paths.root, paths.cardsDir)}.`);
    }
    for (const loaded of cards) {
      console.log(`${loaded.dirName}  [${loaded.card.kind}]  ${describeStatus(cardStatus(loaded, cards, ledger))}`);
    }
    process.exit(0);
  }

  let loaded: LoadedCard;
  if (args.next) {
    const ready = nextCard(cards, ledger);
    if (ready === null) {
      console.log("No card is ready to run. `npm run exp -- --list` shows why.");
      process.exit(0);
    }
    loaded = ready;
  } else {
    loaded = findCard(cards, args.target ?? "");
  }

  const mode = args.scratch ? "scratch" : "result";
  const modeLabel = args.scratch ? "scratch" : "recorded";
  const seeds = args.seeds ?? loaded.card.seeds;
  if (args.dryRun) {
    console.log(
      `Would run ${loaded.dirName} [${loaded.card.kind}] as a ${modeLabel} run: ${seeds.length} seed(s), ` +
        `budget ${loaded.card.budget.maxWallMinutes} min. Status: ${describeStatus(cardStatus(loaded, cards, ledger))}.`,
    );
    process.exit(0);
  }

  console.log(`Running ${loaded.dirName} (${modeLabel}): ${loaded.card.title}`);
  const outcome = executeCard(loaded, cards, ledger, paths, {
    mode,
    ...(args.seeds === undefined ? {} : { seeds: args.seeds }),
    command: `npm run exp -- ${process.argv.slice(2).join(" ")}`,
  });
  const { manifest } = outcome;
  console.log(`\n${loaded.dirName} → ${manifest.verdict.toUpperCase()}`);
  console.log(`  ${manifest.primary.name} = ${manifest.primary.value}`);
  console.log(`  manifest: ${relative(paths.root, outcome.outDir)}/manifest.json`);
  console.log(
    outcome.ledgerEntry === null
      ? "  not recorded (scratch run)"
      : `  recorded: ${relative(paths.root, paths.ledgerPath)} (${outcome.ledgerEntry.ledgerId})`,
  );
  process.exit(args.requirePass && manifest.verdict !== "pass" ? 1 : 0);
} catch (error) {
  if (error instanceof HarnessError || (error instanceof Error && error.name === "CardError")) {
    console.error(`exp: ${error.message}`);
    process.exit(1);
  }
  throw error;
}
