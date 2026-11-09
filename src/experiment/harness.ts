import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { cardStatus, describeStatus, appendLedgerEntry, resultEntry } from "./ledger.ts";
import type { LedgerEntry } from "./ledger.ts";
import type { Card, CardKind, DependencyResult, LoadedCard, RunResult, Verdict } from "./card.ts";
import { canonicalJson, hashFile, hashFiles, sha256 } from "./provenance.ts";
import type { FileHash } from "./provenance.ts";
import { captureRunEnvironment, writeManifest } from "./telemetry.ts";
import type { RunEnvironment, TfjsInfo } from "./telemetry.ts";
import type { Comparison } from "./compare.ts";

/**
 * Runs one experiment card and records what happened: the bookkeeping half of
 * `npm run exp` (`scripts/exp.ts`). The card's own `run.ts` executes in a child process
 * (`scripts/exp-runner.ts`), so this module never loads tfjs, the wall-time budget is enforced
 * by killing that child, and a crashed run cannot leave a half-written record.
 *
 * What "recorded" means: a `manifest.json` (schema v2) next to the run's outputs under
 * `artifacts/<card>/<runId>/`, and one line appended to `experiments/ledger.jsonl`. A scratch
 * run writes its manifest under `artifacts/_scratch/` (gitignored) and never touches the ledger.
 */

export interface HarnessPaths {
  /** Repository root: where git state is read and where relative paths are anchored. */
  root: string;
  cardsDir: string;
  ledgerPath: string;
  artifactsDir: string;
  /** The child-process entry point. Resolved next to this module, not under `root`. */
  runnerPath: string;
}

export function harnessPaths(root: string = process.cwd()): HarnessPaths {
  return {
    root,
    cardsDir: join(root, "experiments", "cards"),
    ledgerPath: join(root, "experiments", "ledger.jsonl"),
    artifactsDir: join(root, "artifacts"),
    runnerPath: fileURLToPath(new URL("../../scripts/exp-runner.ts", import.meta.url)),
  };
}

/** A refusal or a failed run that the CLI should print without a stack trace. */
export class HarnessError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HarnessError";
  }
}

export const MANIFEST_SCHEMA_VERSION = 2;

/** `artifacts/**\/manifest.json`, schema v2. */
export interface RunManifest {
  schemaVersion: typeof MANIFEST_SCHEMA_VERSION;
  runId: string;
  cardId: string;
  /** Card directory name, e.g. `"0000-g2-canary"`. */
  cardDir: string;
  cardSha256: string;
  kind: CardKind;
  /** True for a run that was not recorded in the ledger. */
  scratch: boolean;
  command: string;
  env: RunEnvironment;
  config: Record<string, unknown>;
  configSha256: string;
  seeds: number[];
  primaryMetric: string;
  decisionRule: Card["decisionRule"];
  primary: RunResult["primary"];
  verdict: Verdict;
  resultSummary: unknown;
  /** Every file the run wrote, including gitignored telemetry, so a local copy can be verified. */
  files: FileHash[];
  wallSeconds: number;
  reproOf: string | null;
  match: boolean | null;
  /** Present on reproductions made by `npm run repro`. */
  comparison?: Comparison;
}

/** What the child process is told. Written to a file whose path is its only argument. */
export interface RunnerRequest {
  cardDir: string;
  card: Card;
  seeds: number[];
  outDir: string;
  dependencies: Record<string, DependencyResult>;
  resultPath: string;
}

/** What the child process writes to `RunnerRequest.resultPath`. */
export interface RunnerResponse {
  result: RunResult;
  tfjs: TfjsInfo | null;
}

/** How a reproduction is judged once the re-run finishes. Returns the comparison to record. */
export type ReproJudge = (rerun: { outDir: string; result: RunResult }) => Comparison;

export interface ExecuteOptions {
  /**
   * - `result`: the card's one recorded run. Refused if it already has a verdict.
   * - `scratch`: explore without recording. Any tree state, any seeds, no ledger line.
   * - `repro`: re-run a card that already has a verdict and record whether it matched.
   */
  mode: "result" | "scratch" | "repro";
  /** Override `card.seeds`. Only allowed in scratch mode. */
  seeds?: number[];
  /** The command line, recorded verbatim in the manifest. */
  command: string;
  /**
   * Compare the re-run against a reference. Required in `repro` mode; also honoured in scratch
   * mode, where the comparison is written to the scratch manifest but not ledgered.
   */
  repro?: { of: string; judge: ReproJudge };
  /** Where the child's stdout and stderr go. Defaults to the parent's. */
  stdio?: "inherit" | "ignore";
}

export interface ExecuteOutcome {
  manifest: RunManifest;
  /** Absolute path of the directory holding `manifest.json` and the run's outputs. */
  outDir: string;
  /** `null` for a scratch run. */
  ledgerEntry: LedgerEntry | null;
}

/** `2026-11-09T14:22:33.123Z` + commit → `20261109T142233123Z-e95ce68`. Sorts chronologically. */
export function makeRunId(createdAt: string, gitCommit: string): string {
  return `${createdAt.replace(/[-:.]/g, "")}-${gitCommit.slice(0, 7)}`;
}

function toPosix(path: string): string {
  return path.split(sep).join("/");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseRunnerResponse(path: string): RunnerResponse {
  if (!existsSync(path)) {
    throw new HarnessError("the run exited cleanly but reported no result (run.ts must return a RunResult)");
  }
  const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
  const result = isRecord(parsed) ? parsed.result : undefined;
  const primary = isRecord(result) ? result.primary : undefined;
  if (
    !isRecord(result) ||
    (result.verdict !== "pass" && result.verdict !== "fail") ||
    !isRecord(primary) ||
    typeof primary.name !== "string" ||
    !(primary.value === null || (typeof primary.value === "number" && Number.isFinite(primary.value)))
  ) {
    throw new HarnessError(
      'run.ts returned a malformed result: expected { primary: { name, value: number | null }, verdict: "pass" | "fail", resultSummary }',
    );
  }
  return parsed as RunnerResponse;
}

/** Loads each dependency's recorded result, refusing a manifest that changed after it was ledgered. */
function resolveDependencies(card: Card, ledger: readonly LedgerEntry[], paths: HarnessPaths): Record<string, DependencyResult> {
  const dependencies: Record<string, DependencyResult> = {};
  for (const dependency of card.dependsOn) {
    const entry = resultEntry(dependency.card, ledger);
    if (entry === undefined) {
      throw new HarnessError(`dependency ${dependency.card} has no recorded result`);
    }
    const manifestPath = join(paths.root, entry.manifestPath);
    if (!existsSync(manifestPath)) {
      throw new HarnessError(`dependency ${dependency.card}: manifest ${entry.manifestPath} is missing`);
    }
    if (hashFile(manifestPath) !== entry.manifestSha256) {
      throw new HarnessError(
        `dependency ${dependency.card}: ${entry.manifestPath} no longer matches the hash in the ledger (${entry.ledgerId}). ` +
          "A recorded manifest must not be edited.",
      );
    }
    const manifest: unknown = JSON.parse(readFileSync(manifestPath, "utf8"));
    dependencies[dependency.card] = {
      ledgerId: entry.ledgerId,
      manifestPath: entry.manifestPath,
      verdict: entry.verdict,
      resultSummary: isRecord(manifest) ? manifest.resultSummary : undefined,
    };
  }
  return dependencies;
}

function sameSeeds(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((seed, index) => seed === b[index]);
}

/**
 * Runs `loaded` and records it according to `options.mode`. Throws `HarnessError` for every
 * refusal (card already has a verdict, dependency unmet, dirty tree) and for every run that
 * produced no result (non-zero exit, budget exceeded, malformed result). In all of those cases
 * nothing is recorded and the ledger is untouched.
 */
export function executeCard(
  loaded: LoadedCard,
  cards: readonly LoadedCard[],
  ledger: readonly LedgerEntry[],
  paths: HarnessPaths,
  options: ExecuteOptions,
): ExecuteOutcome {
  const { card, dirName } = loaded;
  const status = cardStatus(loaded, cards, ledger);

  if (status.state === "waiting" || status.state === "dead") {
    throw new HarnessError(`card ${dirName} cannot run: ${describeStatus(status)}`);
  }
  if (options.mode === "result" && status.state === "done") {
    throw new HarnessError(
      `card ${dirName} already has its verdict (${status.verdict}, ${status.ledgerId}). A pre-registered card runs once. ` +
        "Use --scratch to explore without recording, or `npm run repro` to check that it reproduces.",
    );
  }
  if (options.mode === "repro" && options.repro === undefined) {
    throw new Error("executeCard: repro mode needs options.repro");
  }
  const seeds = options.seeds ?? card.seeds;
  if (options.mode !== "scratch" && !sameSeeds(seeds, card.seeds)) {
    throw new HarnessError("seeds can only be overridden in a scratch run");
  }

  const env = captureRunEnvironment(null, paths.root);
  if (options.mode !== "scratch" && env.gitDirty) {
    throw new HarnessError(
      "the working tree has uncommitted or untracked files (other than recorded results under artifacts/ and the ledger), " +
        "so the recorded commit would not describe what ran. Commit first, or use --scratch.",
    );
  }
  const dependencies = resolveDependencies(card, ledger, paths);

  const runId = makeRunId(env.createdAt, env.gitCommit);
  const stagingParent = join(paths.artifactsDir, "_scratch", dirName);
  const stagingDir = join(stagingParent, runId);
  if (existsSync(stagingDir)) {
    throw new HarnessError(`run directory ${stagingDir} already exists`);
  }
  mkdirSync(stagingDir, { recursive: true });
  const requestPath = join(stagingParent, `${runId}.request.json`);
  const resultPath = join(stagingParent, `${runId}.result.json`);
  const request: RunnerRequest = { cardDir: loaded.dir, card, seeds, outDir: stagingDir, dependencies, resultPath };
  writeFileSync(requestPath, JSON.stringify(request));

  let response: RunnerResponse;
  const started = Date.now();
  try {
    const child = spawnSync(process.execPath, [paths.runnerPath, requestPath], {
      cwd: paths.root,
      stdio: options.stdio ?? "inherit",
      timeout: card.budget.maxWallMinutes * 60_000,
      killSignal: "SIGKILL",
    });
    if (child.error !== undefined || child.signal !== null) {
      const timedOut = (child.error as NodeJS.ErrnoException | undefined)?.code === "ETIMEDOUT" || child.signal === "SIGKILL";
      throw new HarnessError(
        timedOut
          ? `card ${dirName} exceeded its budget of ${card.budget.maxWallMinutes} minutes and was stopped. Nothing was recorded. ` +
              `Partial output: ${stagingDir}`
          : `card ${dirName} could not be run: ${child.error?.message ?? `killed by ${child.signal}`}`,
      );
    }
    if (child.status !== 0) {
      throw new HarnessError(`card ${dirName} failed with exit code ${child.status}. Nothing was recorded. Partial output: ${stagingDir}`);
    }
    response = parseRunnerResponse(resultPath);
  } finally {
    rmSync(requestPath, { force: true });
    rmSync(resultPath, { force: true });
  }
  const wallSeconds = (Date.now() - started) / 1000;
  const { result, tfjs } = response;

  const comparison = options.repro?.judge({ outDir: stagingDir, result });
  const isReproCard = card.kind === "repro";
  const kind: CardKind = options.repro !== undefined ? "repro" : card.kind;
  const reproOf = options.repro !== undefined ? options.repro.of : isReproCard ? String(card.config.reproOf) : null;
  const match = comparison !== undefined ? comparison.match : isReproCard ? result.verdict === "pass" : null;

  const manifest: RunManifest = {
    schemaVersion: MANIFEST_SCHEMA_VERSION,
    runId,
    cardId: card.id,
    cardDir: dirName,
    cardSha256: loaded.sha256,
    kind,
    scratch: options.mode === "scratch",
    command: options.command,
    env: { ...env, tfjsVersion: tfjs?.version ?? null, tfjsBackend: tfjs?.backend ?? null },
    config: card.config,
    configSha256: sha256(canonicalJson(card.config)),
    seeds,
    primaryMetric: card.primaryMetric,
    decisionRule: card.decisionRule,
    primary: result.primary,
    verdict: result.verdict,
    resultSummary: result.resultSummary ?? null,
    files: hashFiles(stagingDir),
    wallSeconds,
    reproOf,
    match,
    ...(comparison === undefined ? {} : { comparison }),
  };
  writeManifest(stagingDir, manifest as unknown as Record<string, unknown>);

  if (options.mode === "scratch") {
    return { manifest, outDir: stagingDir, ledgerEntry: null };
  }

  const finalDir =
    options.mode === "repro" ? join(paths.artifactsDir, "repro", dirName, runId) : join(paths.artifactsDir, dirName, runId);
  mkdirSync(dirname(finalDir), { recursive: true });
  renameSync(stagingDir, finalDir);
  const manifestPath = join(finalDir, "manifest.json");
  const ledgerEntry: LedgerEntry = {
    ledgerId: runId,
    cardId: card.id,
    cardSha256: loaded.sha256,
    kind,
    gitCommit: env.gitCommit,
    configSha256: manifest.configSha256,
    manifestPath: toPosix(relative(paths.root, manifestPath)),
    manifestSha256: hashFile(manifestPath),
    primary: result.primary,
    verdict: result.verdict,
    createdAt: env.createdAt,
    reproOf,
    match,
  };
  appendLedgerEntry(paths.ledgerPath, ledgerEntry);
  return { manifest, outDir: finalDir, ledgerEntry };
}
