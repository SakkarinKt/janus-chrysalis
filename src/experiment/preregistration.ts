import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { CARD_DIR_PATTERN } from "./card.ts";
import { readLedger } from "./ledger.ts";
import type { LedgerEntry } from "./ledger.ts";
import { hashDir, hashDirAtCommit, hashFile } from "./provenance.ts";

/**
 * The pre-registration check (`scripts/check-preregistration.ts`, run in CI's `lint` job).
 *
 * "Pre-registered" has to mean something a machine can verify, or it is a habit. Given a base
 * ref (the branch a change is being merged into), this checks that:
 *
 * 1. **The ledger only grows.** Every line of the base's `experiments/ledger.jsonl` is still
 *    there, unaltered and in order.
 * 2. **Recorded manifests are not edited.** Every ledger line's manifest exists and still has
 *    the hash the line recorded.
 * 3. **A card does not change once merged.** Every card directory that exists on the base is
 *    byte-for-byte the same now. To change an experiment, add a new card that supersedes it.
 * 4. **Each new result was produced by code and a card that were already on the base.** For
 *    every new ledger line: the run's tree was clean, the run's commit is an ancestor of the
 *    base, and the card directory *at that commit* hashes to what the line records. This is the
 *    ordering guarantee: squash-merges erase the order of commits inside a PR, so "the card was
 *    committed before the result" is only checkable as "the card was on the base branch, and the
 *    run happened there".
 * 5. **A card has one result.** A confirmatory or calibration card appears in at most one
 *    non-repro line.
 * 6. **Dependencies were met first.** A new result's card dependencies each have an earlier
 *    line with the required verdict.
 * 7. **Committed artifacts are not rewritten.** A file under `artifacts/` that exists on the
 *    base is neither modified nor deleted. New files are fine.
 *
 * Several runs may be recorded from one base commit and committed together: a run's own
 * not-yet-committed outputs do not make the tree dirty (`RECORDED_OUTPUT_DIRS` in
 * `telemetry.ts`), so every run in such a change still executed at a commit that is on the base.
 */

const LEDGER_PATH = "experiments/ledger.jsonl";
const CARDS_DIR = "experiments/cards";
const ARTIFACTS_DIR = "artifacts";

export interface PreregistrationProblem {
  /** Which of the rules above, as a short slug. */
  rule:
    | "ledger-append-only"
    | "ledger-readable"
    | "manifest-immutable"
    | "card-immutable"
    | "artifact-immutable"
    | "run-on-base"
    | "run-once"
    | "dependency";
  /** What the problem is about: a ledger id, a card directory, or a path. */
  where: string;
  detail: string;
}

export interface PreregistrationReport {
  problems: PreregistrationProblem[];
  ledgerLines: number;
  newLines: number;
  cardsOnBase: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function gitOutput(args: string[], root: string): string | null {
  try {
    return execFileSync("git", args, { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
  } catch {
    return null;
  }
}

function nonEmptyLines(text: string): string[] {
  return text.split("\n").filter((line) => line.trim() !== "");
}

/** Throws if `baseRef` does not resolve to a commit in the repository at `root`. */
export function checkPreregistration(root: string, baseRef: string): PreregistrationReport {
  if (gitOutput(["rev-parse", "--verify", "--quiet", `${baseRef}^{commit}`], root) === null) {
    throw new Error(`base ref ${baseRef} does not resolve to a commit`);
  }
  const problems: PreregistrationProblem[] = [];
  const ledgerFile = join(root, LEDGER_PATH);

  // 1. The ledger only grows.
  const baseLines = nonEmptyLines(gitOutput(["show", `${baseRef}:${LEDGER_PATH}`], root) ?? "");
  const headLines = existsSync(ledgerFile) ? nonEmptyLines(readFileSync(ledgerFile, "utf8")) : [];
  let cursor = 0;
  for (const line of headLines) {
    if (cursor < baseLines.length && line === baseLines[cursor]) {
      cursor += 1;
    }
  }
  if (cursor < baseLines.length) {
    problems.push({
      rule: "ledger-append-only",
      where: LEDGER_PATH,
      detail: `line ${cursor + 1} of the ledger on ${baseRef} is missing, altered or out of order: ${(baseLines[cursor] ?? "").slice(0, 120)}`,
    });
  }

  let ledger: LedgerEntry[];
  try {
    ledger = readLedger(ledgerFile);
  } catch (error) {
    problems.push({ rule: "ledger-readable", where: LEDGER_PATH, detail: (error as Error).message });
    return { problems, ledgerLines: headLines.length, newLines: 0, cardsOnBase: 0 };
  }
  const baseIds = new Set(
    baseLines.map((line) => {
      try {
        const parsed: unknown = JSON.parse(line);
        return isRecord(parsed) && typeof parsed.ledgerId === "string" ? parsed.ledgerId : "";
      } catch {
        return "";
      }
    }),
  );

  // 2. Recorded manifests are not edited.
  const manifests = new Map<string, Record<string, unknown>>();
  for (const entry of ledger) {
    const manifestPath = join(root, entry.manifestPath);
    if (!existsSync(manifestPath)) {
      problems.push({ rule: "manifest-immutable", where: entry.ledgerId, detail: `${entry.manifestPath} is missing` });
      continue;
    }
    if (hashFile(manifestPath) !== entry.manifestSha256) {
      problems.push({
        rule: "manifest-immutable",
        where: entry.ledgerId,
        detail: `${entry.manifestPath} no longer has the hash the ledger recorded; a recorded manifest must not be edited`,
      });
      continue;
    }
    const parsed: unknown = JSON.parse(readFileSync(manifestPath, "utf8"));
    if (isRecord(parsed)) {
      manifests.set(entry.ledgerId, parsed);
    }
  }

  // 3. A card does not change once merged.
  const baseCardDirs = (gitOutput(["ls-tree", "-d", "--name-only", baseRef, "--", `${CARDS_DIR}/`], root) ?? "")
    .split("\n")
    .map((path) => path.slice(CARDS_DIR.length + 1))
    .filter((name) => CARD_DIR_PATTERN.test(name));
  for (const name of baseCardDirs) {
    const dir = `${CARDS_DIR}/${name}`;
    const before = hashDirAtCommit(baseRef, dir, root);
    const now = existsSync(join(root, dir)) ? hashDir(join(root, dir)) : null;
    if (before !== now) {
      problems.push({
        rule: "card-immutable",
        where: dir,
        detail:
          now === null
            ? `was removed; a card on ${baseRef} stays, and its status comes from the ledger`
            : `differs from ${baseRef}; a merged card is not edited. Add a new card with "supersedes" instead`,
      });
    }
  }

  // 7. Committed artifacts are not rewritten. (`git diff <base>` compares the base with the work
  // tree for tracked paths, so a brand-new, still-untracked artifact does not appear at all.)
  for (const line of nonEmptyLines(gitOutput(["diff", "--name-status", "--no-renames", baseRef, "--", ARTIFACTS_DIR], root) ?? "")) {
    const [status = "", path = ""] = line.split("\t");
    if (status !== "A") {
      problems.push({
        rule: "artifact-immutable",
        where: path,
        detail: `${status === "D" ? "was deleted" : "was modified"}; an artifact on ${baseRef} is evidence and is never rewritten. Record a new run instead`,
      });
    }
  }

  // 4–6. New results.
  const newEntries = ledger.filter((entry) => !baseIds.has(entry.ledgerId));
  for (const entry of newEntries) {
    const manifest = manifests.get(entry.ledgerId);
    if (manifest === undefined) {
      continue; // already reported under rule 2
    }
    const env = isRecord(manifest.env) ? manifest.env : {};
    const report = (rule: PreregistrationProblem["rule"], detail: string): void => {
      problems.push({ rule, where: entry.ledgerId, detail });
    };

    if (manifest.scratch === true) {
      report("run-on-base", "its manifest is a scratch run; scratch runs are never ledgered");
    }
    if (env.gitDirty !== false) {
      report("run-on-base", "its manifest does not record a clean working tree (env.gitDirty must be false)");
    }
    if (env.gitCommit !== entry.gitCommit) {
      report("run-on-base", `ledger gitCommit ${entry.gitCommit} differs from the manifest's ${String(env.gitCommit)}`);
    }
    const onBase = gitOutput(["merge-base", "--is-ancestor", entry.gitCommit, baseRef], root) !== null;
    if (!onBase) {
      report(
        "run-on-base",
        `it ran at ${entry.gitCommit.slice(0, 12)}, which is not on ${baseRef}. A result is only pre-registered if its card and ` +
          "code were merged before the run: merge the card first, then run it from the base branch",
      );
    }

    const isLegacy = entry.cardId.startsWith("legacy:");
    const cardDirName = typeof manifest.cardDir === "string" ? manifest.cardDir : null;
    if (!isLegacy) {
      if (cardDirName === null || !CARD_DIR_PATTERN.test(cardDirName)) {
        report("run-on-base", "its manifest names no card directory");
      } else if (onBase) {
        const atRun = hashDirAtCommit(entry.gitCommit, `${CARDS_DIR}/${cardDirName}`, root);
        if (atRun === null) {
          report("run-on-base", `card ${cardDirName} did not exist at the run's commit`);
        } else if (atRun !== entry.cardSha256 || manifest.cardSha256 !== entry.cardSha256) {
          report("run-on-base", `card ${cardDirName} at the run's commit does not hash to the cardSha256 the ledger records`);
        }
      }
    }

    if (entry.kind === "repro" || isLegacy) {
      continue;
    }

    // 5. A card has one result.
    const results = ledger.filter((other) => other.cardId === entry.cardId && other.kind !== "repro");
    if (results.length > 1 && results[0]?.ledgerId !== entry.ledgerId) {
      report("run-once", `card ${entry.cardId} already has a result (${results[0]?.ledgerId}); a pre-registered card runs once`);
    }

    // 6. Dependencies were met first.
    if (cardDirName !== null && onBase) {
      const cardJson = gitOutput(["show", `${entry.gitCommit}:${CARDS_DIR}/${cardDirName}/card.json`], root);
      const card: unknown = cardJson === null ? null : JSON.parse(cardJson);
      const dependsOn = isRecord(card) && Array.isArray(card.dependsOn) ? card.dependsOn : [];
      const position = ledger.indexOf(entry);
      for (const dependency of dependsOn) {
        if (!isRecord(dependency)) {
          continue;
        }
        const met = ledger
          .slice(0, position)
          .some((earlier) => earlier.cardId === dependency.card && earlier.kind !== "repro" && earlier.verdict === dependency.verdict);
        if (!met) {
          report("dependency", `needs card ${String(dependency.card)} to ${String(dependency.verdict)} first, and no earlier ledger line says so`);
        }
      }
    }
  }

  return { problems, ledgerLines: ledger.length, newLines: newEntries.length, cardsOnBase: baseCardDirs.length };
}
