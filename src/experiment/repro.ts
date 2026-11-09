import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import type { LoadedCard, RunResult } from "./card.ts";
import { compareSummaryDirs, difference, mergeComparisons } from "./compare.ts";
import type { Comparison, CsvMismatch } from "./compare.ts";
import { HarnessError, MANIFEST_SCHEMA_VERSION, executeCard, makeRunId } from "./harness.ts";
import type { HarnessPaths, RunManifest } from "./harness.ts";
import { appendLedgerEntry } from "./ledger.ts";
import type { LedgerEntry } from "./ledger.ts";
import { canonicalJson, hashFile, hashFiles, listFiles, sha256 } from "./provenance.ts";
import { captureRunEnvironment, writeManifest } from "./telemetry.ts";

/**
 * Reproduction: re-run something that already has a recorded result and say whether the
 * outputs match (`npm run repro`, `scripts/repro.ts`).
 *
 * Two kinds of target:
 *
 * - A **harness manifest** (schema v2). Its card is re-run at HEAD through the same machinery as
 *   `npm run exp`, and the re-run's summary CSVs, primary value and verdict are compared with the
 *   original's. The card must be unchanged since that run.
 * - A **pre-harness manifest** (v1, everything under `artifacts/2026-0*`). Those scripts are
 *   frozen at the commit they ran on and mostly do not run at HEAD (`experiments/README.md`), so
 *   the script is run in a temporary detached worktree at the manifest's recorded commit and its
 *   regenerated summary CSVs are compared with the committed ones.
 *
 * Either way the outcome is a manifest under `artifacts/repro/` and one `repro` ledger line,
 * unless `scratch` is set.
 */

export interface ReproOptions {
  /** Largest allowed `difference()` between a reference number and its re-run. 0 means exact. */
  tolerance: number;
  /** Required for a pre-harness manifest: run at the manifest's commit, in a detached worktree. */
  atCommit: boolean;
  /** Compare and report, but record nothing (works on a dirty tree). */
  scratch: boolean;
  command: string;
  /** Pre-harness runs only: wall-time limit for the script. */
  budgetMinutes: number;
  /** Pre-harness runs only: script path, when it is not `experiments/<run>/run.ts`. */
  script?: string;
  stdio?: "inherit" | "ignore";
}

export interface ReproOutcome {
  /** What was reproduced: a ledger id, or `legacy:<run-dir>`. */
  of: string;
  comparison: Comparison;
  /** Directory holding the reproduction's `manifest.json`. */
  outDir: string;
  /** `null` for a scratch reproduction. */
  ledgerEntry: LedgerEntry | null;
}

const SUMMARY_SUFFIX = ".summary.csv";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toPosix(path: string): string {
  return path.split(sep).join("/");
}

/** A ledger id, or a path to a `manifest.json` (absolute, or relative to the repo root). */
export function resolveManifestPath(target: string, ledger: readonly LedgerEntry[], paths: HarnessPaths): string {
  const entry = ledger.find((candidate) => candidate.ledgerId === target);
  const path = entry !== undefined ? join(paths.root, entry.manifestPath) : isAbsolute(target) ? target : resolve(paths.root, target);
  if (basename(path) !== "manifest.json" || !existsSync(path)) {
    throw new HarnessError(`"${target}" is neither a ledger id nor a path to an existing manifest.json`);
  }
  return path;
}

function fileLevel(detail: string): CsvMismatch {
  return { file: "manifest.json", line: 0, column: 0, detail };
}

/** Compares a card re-run with its original: summary CSVs (if the original has any), primary value, verdict. */
export function judgeRerun(referenceDir: string, original: RunManifest, outDir: string, result: RunResult, tolerance: number): Comparison {
  const parts: Comparison[] = [];
  if (listFiles(referenceDir).some((path) => path.endsWith(SUMMARY_SUFFIX))) {
    parts.push(compareSummaryDirs(referenceDir, outDir, tolerance));
  }

  const mismatches: CsvMismatch[] = [];
  let maxDifference = 0;
  if (result.verdict !== original.verdict) {
    mismatches.push(fileLevel(`verdict: expected "${original.verdict}", got "${result.verdict}"`));
  }
  const expected = original.primary.value;
  const actual = result.primary.value;
  if (expected === null || actual === null) {
    if (expected !== actual) {
      mismatches.push(fileLevel(`${original.primary.name}: expected ${expected}, got ${actual}`));
    }
  } else {
    maxDifference = difference(expected, actual);
    if (maxDifference > tolerance) {
      mismatches.push(fileLevel(`${original.primary.name}: expected ${expected}, got ${actual} (difference ${maxDifference.toExponential(3)})`));
    }
  }
  parts.push({
    identical: mismatches.length === 0 && maxDifference === 0,
    maxDifference,
    mismatches,
    mismatchCount: mismatches.length,
    filesCompared: 0,
    tolerance,
    match: mismatches.length === 0,
  });
  return mergeComparisons(parts, tolerance);
}

function reproduceCardRun(
  original: RunManifest,
  manifestPath: string,
  cards: readonly LoadedCard[],
  ledger: readonly LedgerEntry[],
  paths: HarnessPaths,
  options: ReproOptions,
): ReproOutcome {
  if (options.atCommit) {
    throw new HarnessError(
      "--at-commit is for pre-harness (v1) manifests. A harness manifest is re-run at HEAD; to re-run it at its own commit, " +
        `check out ${original.env.gitCommit} and run \`npm run exp -- ${original.cardId} --scratch\` there.`,
    );
  }
  if (original.scratch) {
    throw new HarnessError("that manifest is a scratch run. Only recorded runs are reproduced.");
  }
  const loaded = cards.find((card) => card.card.id === original.cardId);
  if (loaded === undefined) {
    throw new HarnessError(`card ${original.cardId} (${original.cardDir}) no longer exists`);
  }
  if (loaded.sha256 !== original.cardSha256) {
    throw new HarnessError(
      `card ${loaded.dirName} has changed since run ${original.runId} (hash ${loaded.sha256.slice(0, 12)} now, ` +
        `${original.cardSha256.slice(0, 12)} then), so a re-run would not be the same experiment.`,
    );
  }

  const referenceDir = dirname(manifestPath);
  let comparison: Comparison | undefined;
  const outcome = executeCard(loaded, cards, ledger, paths, {
    mode: options.scratch ? "scratch" : "repro",
    command: options.command,
    repro: {
      of: original.runId,
      judge: ({ outDir, result }) => {
        comparison = judgeRerun(referenceDir, original, outDir, result, options.tolerance);
        return comparison;
      },
    },
    ...(options.stdio === undefined ? {} : { stdio: options.stdio }),
  });
  if (comparison === undefined) {
    throw new Error("reproduceCardRun: the re-run was never judged");
  }
  return { of: original.runId, comparison, outDir: outcome.outDir, ledgerEntry: outcome.ledgerEntry };
}

function git(args: string[], cwd: string): string {
  return execFileSync("git", args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
}

function gitSucceeds(args: string[], cwd: string): boolean {
  try {
    git(args, cwd);
    return true;
  } catch {
    return false;
  }
}

/**
 * Gives the worktree the packages its commit asks for. If its lockfile is byte-identical to the
 * current one, the current `node_modules` is linked in; otherwise `npm ci` installs that
 * commit's own lockfile. A checkout with no lockfile needs nothing.
 */
function provideDependencies(root: string, worktree: string, stdio: "inherit" | "ignore"): "none" | "linked" | "installed" {
  const lockThen = join(worktree, "package-lock.json");
  if (!existsSync(lockThen)) {
    return "none";
  }
  const lockNow = join(root, "package-lock.json");
  const modulesNow = join(root, "node_modules");
  if (existsSync(modulesNow) && existsSync(lockNow) && hashFile(lockNow) === hashFile(lockThen)) {
    symlinkSync(modulesNow, join(worktree, "node_modules"), "dir");
    return "linked";
  }
  const install = spawnSync("npm", ["ci"], { cwd: worktree, stdio });
  if (install.status !== 0) {
    throw new HarnessError(`could not install that commit's dependencies (npm ci exited ${install.status})`);
  }
  return "installed";
}

function reproduceLegacyRun(
  original: Record<string, unknown>,
  manifestPath: string,
  paths: HarnessPaths,
  options: ReproOptions,
): ReproOutcome {
  const insideArtifacts = toPosix(relative(paths.artifactsDir, manifestPath));
  const run = insideArtifacts.split("/")[0] ?? "";
  if (insideArtifacts.startsWith("..") || run === "" || run === "manifest.json" || run === "_scratch" || run === "repro") {
    throw new HarnessError(`${manifestPath} is not a run manifest under artifacts/<run>/`);
  }
  const commit = original.gitCommit;
  if (typeof commit !== "string" || !/^[0-9a-f]{7,40}$/.test(commit)) {
    throw new HarnessError(`${insideArtifacts} records no gitCommit, so there is no commit to reproduce it at`);
  }
  if (!options.atCommit) {
    throw new HarnessError(
      `artifacts/${insideArtifacts} is a pre-harness manifest. Its script is frozen at the commit it ran on and is not kept ` +
        "running at HEAD (experiments/README.md), so it can only be reproduced there: add --at-commit.",
    );
  }
  const script = options.script ?? `experiments/${run}/run.ts`;
  if (!gitSucceeds(["cat-file", "-e", `${commit}^{commit}`], paths.root)) {
    throw new HarnessError(`commit ${commit} is not in this repository (fetch the evidence/* tags?)`);
  }
  if (!gitSucceeds(["cat-file", "-e", `${commit}:${script}`], paths.root)) {
    throw new HarnessError(`${script} does not exist at ${commit.slice(0, 7)}. Name the script with --script.`);
  }

  const env = captureRunEnvironment(null, paths.root);
  if (!options.scratch && env.gitDirty) {
    throw new HarnessError(
      "the working tree has uncommitted or untracked files (other than recorded results under artifacts/ and the ledger). " +
        "Commit first, or use --scratch.",
    );
  }
  const scriptSha256 = sha256(execFileSync("git", ["show", `${commit}:${script}`], { cwd: paths.root }));
  const stdio = options.stdio ?? "inherit";
  const worktree = mkdtempSync(join(tmpdir(), "janus-repro-"));
  const started = Date.now();
  let dependencies: "none" | "linked" | "installed";
  let comparison: Comparison;
  let regenerated: ReturnType<typeof hashFiles>;
  try {
    git(["worktree", "add", "--detach", "--quiet", worktree, commit], paths.root);
    dependencies = provideDependencies(paths.root, worktree, stdio);
    const child = spawnSync(process.execPath, [script], {
      cwd: worktree,
      stdio,
      timeout: options.budgetMinutes * 60_000,
      killSignal: "SIGKILL",
    });
    if (child.error !== undefined || child.signal !== null) {
      throw new HarnessError(`${script} did not finish within ${options.budgetMinutes} minutes at ${commit.slice(0, 7)}. Nothing was recorded.`);
    }
    if (child.status !== 0) {
      throw new HarnessError(`${script} exited ${child.status} at its own commit ${commit.slice(0, 7)}. Nothing was recorded.`);
    }
    const regeneratedDir = join(worktree, "artifacts", run);
    comparison = compareSummaryDirs(join(paths.artifactsDir, run), regeneratedDir, options.tolerance);
    regenerated = existsSync(regeneratedDir) ? hashFiles(regeneratedDir).filter((file) => file.path.endsWith(SUMMARY_SUFFIX)) : [];
  } finally {
    gitSucceeds(["worktree", "remove", "--force", worktree], paths.root);
    rmSync(worktree, { recursive: true, force: true });
    gitSucceeds(["worktree", "prune"], paths.root);
  }

  const of = `legacy:${run}`;
  const runId = makeRunId(env.createdAt, env.gitCommit);
  const config = { script, commit, tolerance: options.tolerance };
  const verdict = comparison.match ? "pass" : "fail";
  const primary = { name: "maxDifference", value: comparison.maxDifference };
  const outDir = join(paths.artifactsDir, options.scratch ? "_scratch" : "repro", `legacy-${run}`, runId);
  mkdirSync(outDir, { recursive: true });
  writeManifest(outDir, {
    schemaVersion: MANIFEST_SCHEMA_VERSION,
    runId,
    cardId: of,
    cardDir: null,
    cardSha256: scriptSha256,
    kind: "repro",
    scratch: options.scratch,
    command: options.command,
    env,
    config,
    configSha256: sha256(canonicalJson(config)),
    seeds: [],
    primaryMetric: primary.name,
    primary,
    verdict,
    resultSummary: {
      reproducedCommit: commit,
      script,
      dependencies,
      reference: { manifest: `artifacts/${insideArtifacts}`, nodeVersion: original.nodeVersion ?? null },
    },
    files: regenerated,
    wallSeconds: (Date.now() - started) / 1000,
    reproOf: of,
    match: comparison.match,
    comparison,
  });
  if (options.scratch) {
    return { of, comparison, outDir, ledgerEntry: null };
  }

  const recordedManifest = join(outDir, "manifest.json");
  const ledgerEntry: LedgerEntry = {
    ledgerId: runId,
    cardId: of,
    cardSha256: scriptSha256,
    kind: "repro",
    gitCommit: env.gitCommit,
    configSha256: sha256(canonicalJson(config)),
    manifestPath: toPosix(relative(paths.root, recordedManifest)),
    manifestSha256: hashFile(recordedManifest),
    primary,
    verdict,
    createdAt: env.createdAt,
    reproOf: of,
    match: comparison.match,
  };
  appendLedgerEntry(paths.ledgerPath, ledgerEntry);
  return { of, comparison, outDir, ledgerEntry };
}

/** Reproduces the run recorded by `target` (a ledger id or a manifest path). */
export function reproduce(
  target: string,
  cards: readonly LoadedCard[],
  ledger: readonly LedgerEntry[],
  paths: HarnessPaths,
  options: ReproOptions,
): ReproOutcome {
  const manifestPath = resolveManifestPath(target, ledger, paths);
  const original: unknown = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (!isRecord(original)) {
    throw new HarnessError(`${manifestPath} is not a JSON object`);
  }
  if (original.schemaVersion === undefined) {
    return reproduceLegacyRun(original, manifestPath, paths, options);
  }
  if (original.schemaVersion !== MANIFEST_SCHEMA_VERSION) {
    throw new HarnessError(`${manifestPath} has schemaVersion ${String(original.schemaVersion)}, which this harness does not know`);
  }
  if (original.kind === "repro") {
    throw new HarnessError(
      `${manifestPath} is itself a reproduction record. Reproduce the run it points at (${String(original.reproOf)}), ` +
        "or run a repro card again with `npm run exp`.",
    );
  }
  return reproduceCardRun(original as unknown as RunManifest, manifestPath, cards, ledger, paths, options);
}
