import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { hashDir } from "./provenance.ts";

/**
 * Experiment cards: one directory per experiment under `experiments/cards/NNNN-slug/`, holding
 * `card.json` (what will be run and how it will be judged) and `run.ts` (the code that runs it).
 * A card is written and merged *before* its run, so the pass/fail rule cannot be chosen after
 * seeing the result. `scripts/exp.ts` executes cards; `src/experiment/ledger.ts` records verdicts.
 */

export const CARD_KINDS = ["confirmatory", "calibration", "repro"] as const;
/**
 * - `confirmatory`: tests a hypothesis against a pre-registered rule. Runs once.
 * - `calibration`: measures a quantity later cards depend on, with its own pass rule. Runs once.
 * - `repro`: re-runs something and compares it to a stored reference. Runs on demand, any
 *   number of times, and is never part of the queue.
 */
export type CardKind = (typeof CARD_KINDS)[number];

export type Verdict = "pass" | "fail";

export interface CardDependency {
  /** Four-digit id of the card this one needs a verdict from. */
  card: string;
  /** The verdict that card must have reached. Any other verdict means this card never runs. */
  verdict: Verdict;
  /** Which parts of that card's `resultSummary` this card reads. Documentation for the reviewer. */
  reads?: string[];
}

export interface Card {
  /** Four digits, equal to the directory name's prefix. */
  id: string;
  title: string;
  /** What is expected, stated so that the decision rule can contradict it. */
  hypothesis: string;
  kind: CardKind;
  /** Everything the run is parameterised by. Hashed into the manifest as `configSha256`. */
  config: Record<string, unknown>;
  seeds: number[];
  /** Name of the one number the decision rule is about. */
  primaryMetric: string;
  /** Reported alongside, never part of the rule. */
  secondaryMetrics: string[];
  decisionRule: {
    /** The rule in words, complete enough to apply without reading `run.ts`. */
    text: string;
    /** The numbers the rule uses. `run.ts` reads them from here, never from its own constants. */
    thresholds: Record<string, number>;
  };
  /** What happens to the research plan if the verdict is `fail`. */
  onFail: string;
  dependsOn: CardDependency[];
  budget: { maxWallMinutes: number };
  /** Id of a card this one replaces before that card was run, or `null`. */
  supersedes: string | null;
}

export interface LoadedCard {
  card: Card;
  /** Directory name, e.g. `"0000-g2-canary"`. */
  dirName: string;
  /** Path to the card directory as given to `loadCard`. */
  dir: string;
  /** `hashDir` of the card directory: changes if `card.json`, `run.ts` or any helper changes. */
  sha256: string;
}

/** A dependency's recorded result, handed to the dependent card's `run`. */
export interface DependencyResult {
  ledgerId: string;
  manifestPath: string;
  verdict: Verdict;
  resultSummary: unknown;
}

export interface RunContext {
  card: Card;
  /** The seeds to use. Equal to `card.seeds` except in a scratch run with `--seeds`. */
  seeds: number[];
  /** An existing, empty directory. Everything written here is hashed into the manifest. */
  outDir: string;
  /** Keyed by dependency card id. */
  dependencies: Record<string, DependencyResult>;
  log: (line: string) => void;
}

export interface RunResult {
  primary: { name: string; value: number | null };
  /**
   * The decision rule applied to this run. A run that crashes for a reason the rule covers (for
   * example a NaN halt that the rule counts as failure) must catch it and return `"fail"`.
   * Throwing means "no result": nothing is recorded and the card can be run again.
   */
  verdict: Verdict;
  /** Whatever a reader, or a dependent card, needs. Must be JSON-serialisable. */
  resultSummary: unknown;
}

/** What a card's `run.ts` must export as `run`. */
export type CardRun = (context: RunContext) => RunResult | Promise<RunResult>;

/** `NNNN-slug`: four digits, then lower-case words separated by single hyphens. */
export const CARD_DIR_PATTERN = /^(\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*$/;

export class CardError extends Error {
  readonly problems: string[];

  constructor(dirName: string, problems: string[]) {
    super(`invalid card ${dirName}:\n${problems.map((problem) => `  - ${problem}`).join("\n")}`);
    this.name = "CardError";
    this.problems = problems;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim() !== "";
}

const CARD_ID = /^\d{4}$/;
const KNOWN_KEYS = new Set([
  "id",
  "title",
  "hypothesis",
  "kind",
  "config",
  "seeds",
  "primaryMetric",
  "secondaryMetrics",
  "decisionRule",
  "onFail",
  "dependsOn",
  "budget",
  "supersedes",
]);

/**
 * Validates parsed `card.json` content and returns it typed. Reports every problem at once
 * (a `CardError`), not just the first: a card is reviewed as a whole.
 */
export function parseCard(raw: unknown, dirName: string): Card {
  const problems: string[] = [];
  if (!isRecord(raw)) {
    throw new CardError(dirName, ["card.json must be a JSON object"]);
  }
  for (const key of Object.keys(raw)) {
    if (!KNOWN_KEYS.has(key)) {
      problems.push(`unknown field "${key}"`);
    }
  }

  const dirMatch = CARD_DIR_PATTERN.exec(dirName);
  if (dirMatch === null) {
    problems.push(`directory name must look like 0001-short-slug, got "${dirName}"`);
  }
  if (typeof raw.id !== "string" || !CARD_ID.test(raw.id)) {
    problems.push(`id must be four digits, got ${JSON.stringify(raw.id)}`);
  } else if (dirMatch !== null && dirMatch[1] !== raw.id) {
    problems.push(`id "${raw.id}" does not match the directory prefix "${dirMatch[1]}"`);
  }

  for (const field of ["title", "hypothesis", "primaryMetric", "onFail"] as const) {
    if (!isNonEmptyString(raw[field])) {
      problems.push(`${field} must be a non-empty string`);
    }
  }
  if (typeof raw.kind !== "string" || !(CARD_KINDS as readonly string[]).includes(raw.kind)) {
    problems.push(`kind must be one of ${CARD_KINDS.join(", ")}, got ${JSON.stringify(raw.kind)}`);
  }
  if (!isRecord(raw.config)) {
    problems.push("config must be an object");
  }

  if (!Array.isArray(raw.seeds) || raw.seeds.length === 0) {
    problems.push("seeds must be a non-empty array");
  } else if (!raw.seeds.every((seed) => Number.isSafeInteger(seed) && (seed as number) >= 0)) {
    problems.push("seeds must be non-negative integers");
  } else if (new Set(raw.seeds).size !== raw.seeds.length) {
    problems.push("seeds must not repeat");
  }

  if (!Array.isArray(raw.secondaryMetrics) || !raw.secondaryMetrics.every(isNonEmptyString)) {
    problems.push("secondaryMetrics must be an array of non-empty strings (may be empty)");
  }

  if (!isRecord(raw.decisionRule)) {
    problems.push("decisionRule must be an object with text and thresholds");
  } else {
    if (!isNonEmptyString(raw.decisionRule.text)) {
      problems.push("decisionRule.text must be a non-empty string");
    }
    const thresholds = raw.decisionRule.thresholds;
    if (!isRecord(thresholds) || !Object.values(thresholds).every((value) => typeof value === "number" && Number.isFinite(value))) {
      problems.push("decisionRule.thresholds must be an object of finite numbers (may be empty)");
    }
  }

  if (!Array.isArray(raw.dependsOn)) {
    problems.push("dependsOn must be an array (may be empty)");
  } else {
    raw.dependsOn.forEach((dependency, index) => {
      if (!isRecord(dependency)) {
        problems.push(`dependsOn[${index}] must be an object`);
        return;
      }
      if (typeof dependency.card !== "string" || !CARD_ID.test(dependency.card)) {
        problems.push(`dependsOn[${index}].card must be a four-digit card id`);
      } else if (dependency.card === raw.id) {
        problems.push(`dependsOn[${index}].card is this card itself`);
      }
      if (dependency.verdict !== "pass" && dependency.verdict !== "fail") {
        problems.push(`dependsOn[${index}].verdict must be "pass" or "fail"`);
      }
      if (dependency.reads !== undefined && !(Array.isArray(dependency.reads) && dependency.reads.every(isNonEmptyString))) {
        problems.push(`dependsOn[${index}].reads must be an array of non-empty strings`);
      }
    });
  }

  const maxWallMinutes = isRecord(raw.budget) ? raw.budget.maxWallMinutes : undefined;
  if (typeof maxWallMinutes !== "number" || !Number.isFinite(maxWallMinutes) || maxWallMinutes <= 0) {
    problems.push("budget.maxWallMinutes must be a positive number");
  }

  if (raw.supersedes !== null && !(typeof raw.supersedes === "string" && CARD_ID.test(raw.supersedes))) {
    problems.push("supersedes must be null or a four-digit card id");
  } else if (raw.supersedes !== null && raw.supersedes === raw.id) {
    problems.push("supersedes is this card itself");
  }

  if (raw.kind === "repro" && !(isRecord(raw.config) && isNonEmptyString(raw.config.reproOf))) {
    problems.push('a repro card must name what it reproduces in config.reproOf (e.g. "legacy:2026-09-23-g2-loss-curve")');
  }

  if (problems.length > 0) {
    throw new CardError(dirName, problems);
  }
  return raw as unknown as Card;
}

/** Reads, validates and hashes one card directory. Throws `CardError` if it is not a valid card. */
export function loadCard(dir: string, dirName: string): LoadedCard {
  const cardPath = join(dir, "card.json");
  if (!existsSync(cardPath)) {
    throw new CardError(dirName, ["card.json is missing"]);
  }
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(cardPath, "utf8"));
  } catch (error) {
    throw new CardError(dirName, [`card.json is not valid JSON: ${(error as Error).message}`]);
  }
  const card = parseCard(raw, dirName);
  if (!existsSync(join(dir, "run.ts"))) {
    throw new CardError(dirName, ["run.ts is missing"]);
  }
  return { card, dirName, dir, sha256: hashDir(dir) };
}

/**
 * Every card under `cardsDir`, sorted by id. Only directories named `NNNN-slug` count, so
 * `TEMPLATE/`, `proposed/` and stray files are ignored. Throws if two cards share an id.
 */
export function listCards(cardsDir: string): LoadedCard[] {
  if (!existsSync(cardsDir)) {
    return [];
  }
  const cards = readdirSync(cardsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && CARD_DIR_PATTERN.test(entry.name))
    .map((entry) => loadCard(join(cardsDir, entry.name), entry.name))
    .sort((a, b) => (a.dirName < b.dirName ? -1 : 1));
  const seen = new Map<string, string>();
  for (const loaded of cards) {
    const earlier = seen.get(loaded.card.id);
    if (earlier !== undefined) {
      throw new CardError(loaded.dirName, [`id ${loaded.card.id} is already used by ${earlier}`]);
    }
    seen.set(loaded.card.id, loaded.dirName);
  }
  return cards;
}

/** Finds a card by four-digit id or by full directory name. Throws if there is none. */
export function findCard(cards: readonly LoadedCard[], idOrDirName: string): LoadedCard {
  const found = cards.find((loaded) => loaded.card.id === idOrDirName || loaded.dirName === idOrDirName);
  if (found === undefined) {
    const known = cards.map((loaded) => loaded.dirName).join(", ");
    throw new Error(`no card "${idOrDirName}" (known cards: ${known === "" ? "none" : known})`);
  }
  return found;
}
