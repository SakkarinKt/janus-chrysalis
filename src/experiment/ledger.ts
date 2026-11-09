import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { canonicalJson } from "./provenance.ts";
import { CARD_KINDS } from "./card.ts";
import type { CardKind, LoadedCard, Verdict } from "./card.ts";

/**
 * The results ledger: `experiments/ledger.jsonl`, one JSON object per line, append-only. It is
 * the only record of which cards have a verdict, so a card's status is always derived from here
 * and never stored on the card (a card directory does not change after it is merged).
 */

export interface LedgerEntry {
  /** Unique per run; equal to the run's manifest `runId`. */
  ledgerId: string;
  /** Four-digit card id, or `legacy:<run-dir>` for a reproduction of a pre-harness run. */
  cardId: string;
  /** Hash of the card directory the run executed (for a legacy run, of its script). */
  cardSha256: string;
  /**
   * What kind of run this was: the card's own kind for its one result, or `repro` for anything
   * that re-ran something and compared it to a reference.
   */
  kind: CardKind;
  gitCommit: string;
  configSha256: string;
  /** Repo-relative POSIX path to the run's `manifest.json`. */
  manifestPath: string;
  manifestSha256: string;
  primary: { name: string; value: number | null };
  verdict: Verdict;
  createdAt: string;
  /** For `repro` runs, what was reproduced (a ledger id or `legacy:<run-dir>`); else `null`. */
  reproOf: string | null;
  /** For `repro` runs, whether the outputs matched the reference; else `null`. */
  match: boolean | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function entryProblem(value: unknown): string | null {
  if (!isRecord(value)) {
    return "not a JSON object";
  }
  for (const field of ["ledgerId", "cardId", "cardSha256", "gitCommit", "configSha256", "manifestPath", "manifestSha256", "createdAt"]) {
    if (typeof value[field] !== "string" || value[field] === "") {
      return `${field} must be a non-empty string`;
    }
  }
  if (typeof value.kind !== "string" || !(CARD_KINDS as readonly string[]).includes(value.kind)) {
    return `kind must be one of ${CARD_KINDS.join(", ")}`;
  }
  if (value.verdict !== "pass" && value.verdict !== "fail") {
    return 'verdict must be "pass" or "fail"';
  }
  const primary = value.primary;
  if (!isRecord(primary) || typeof primary.name !== "string" || !(primary.value === null || typeof primary.value === "number")) {
    return "primary must be { name: string, value: number | null }";
  }
  if (!(value.reproOf === null || typeof value.reproOf === "string")) {
    return "reproOf must be a string or null";
  }
  if (!(value.match === null || typeof value.match === "boolean")) {
    return "match must be a boolean or null";
  }
  return null;
}

/**
 * Reads every entry, in file order. A missing file is an empty ledger. Blank lines are skipped.
 * A malformed line throws with its line number: a ledger that cannot be read in full must not
 * be treated as "this card has no verdict yet".
 */
export function readLedger(path: string): LedgerEntry[] {
  if (!existsSync(path)) {
    return [];
  }
  const entries: LedgerEntry[] = [];
  const seen = new Set<string>();
  readFileSync(path, "utf8")
    .split("\n")
    .forEach((line, index) => {
      if (line.trim() === "") {
        return;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch (error) {
        throw new Error(`${path}:${index + 1}: not valid JSON (${(error as Error).message})`);
      }
      const problem = entryProblem(parsed);
      if (problem !== null) {
        throw new Error(`${path}:${index + 1}: ${problem}`);
      }
      const entry = parsed as LedgerEntry;
      if (seen.has(entry.ledgerId)) {
        throw new Error(`${path}:${index + 1}: duplicate ledgerId ${entry.ledgerId}`);
      }
      seen.add(entry.ledgerId);
      entries.push(entry);
    });
  return entries;
}

/** Appends one entry as a single canonical-JSON line. Never rewrites existing lines. */
export function appendLedgerEntry(path: string, entry: LedgerEntry): void {
  const problem = entryProblem(entry);
  if (problem !== null) {
    throw new Error(`refusing to append a malformed ledger entry: ${problem}`);
  }
  const existing = existsSync(path) ? readFileSync(path, "utf8") : "";
  const needsNewline = existing !== "" && !existing.endsWith("\n");
  appendFileSync(path, `${needsNewline ? "\n" : ""}${canonicalJson(entry)}\n`);
}

/**
 * A card's one result: the first entry for it that is not a reproduction. `repro` cards and
 * `npm run repro` runs never produce one.
 */
export function resultEntry(cardId: string, ledger: readonly LedgerEntry[]): LedgerEntry | undefined {
  return ledger.find((entry) => entry.cardId === cardId && entry.kind !== "repro");
}

export type CardStatus =
  /** Has its verdict. Never runs again. */
  | { state: "done"; verdict: Verdict; ledgerId: string }
  /** Can run now. */
  | { state: "ready" }
  /** A dependency has no verdict yet. */
  | { state: "waiting"; on: string[] }
  /** Will never run: a dependency reached the wrong verdict, or a later card supersedes this one. */
  | { state: "dead"; reason: string }
  /** A `repro` card: runs whenever asked, never queued. */
  | { state: "on-demand" };

/** Where a card stands, derived from the ledger and the other cards. */
export function cardStatus(loaded: LoadedCard, cards: readonly LoadedCard[], ledger: readonly LedgerEntry[]): CardStatus {
  const { card } = loaded;
  if (card.kind === "repro") {
    return { state: "on-demand" };
  }
  const result = resultEntry(card.id, ledger);
  if (result !== undefined) {
    return { state: "done", verdict: result.verdict, ledgerId: result.ledgerId };
  }
  const successor = cards.find((other) => other.card.supersedes === card.id);
  if (successor !== undefined) {
    return { state: "dead", reason: `superseded by ${successor.dirName} before it ran` };
  }

  const waiting: string[] = [];
  for (const dependency of card.dependsOn) {
    if (!cards.some((other) => other.card.id === dependency.card)) {
      return { state: "dead", reason: `depends on card ${dependency.card}, which does not exist` };
    }
    const dependencyResult = resultEntry(dependency.card, ledger);
    if (dependencyResult === undefined) {
      waiting.push(dependency.card);
    } else if (dependencyResult.verdict !== dependency.verdict) {
      return {
        state: "dead",
        reason: `needs card ${dependency.card} to ${dependency.verdict}, but it reached "${dependencyResult.verdict}" (${dependencyResult.ledgerId})`,
      };
    }
  }
  return waiting.length > 0 ? { state: "waiting", on: waiting } : { state: "ready" };
}

/** The lowest-numbered card that can run now, or `null` if the queue has nothing runnable. */
export function nextCard(cards: readonly LoadedCard[], ledger: readonly LedgerEntry[]): LoadedCard | null {
  return cards.find((loaded) => cardStatus(loaded, cards, ledger).state === "ready") ?? null;
}

/** One line per card, for `npm run exp -- --list`. */
export function describeStatus(status: CardStatus): string {
  switch (status.state) {
    case "done":
      return `done: ${status.verdict} (${status.ledgerId})`;
    case "ready":
      return "ready";
    case "waiting":
      return `waiting on ${status.on.join(", ")}`;
    case "dead":
      return `will not run: ${status.reason}`;
    case "on-demand":
      return "on demand (repro)";
  }
}
