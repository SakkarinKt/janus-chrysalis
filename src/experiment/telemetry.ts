import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { gitState, hashFile } from "./provenance.ts";

/**
 * Run identity and the run-artifact writing conventions. First extracted for `loop/GOAL.md`
 * priority 5 ("JSONL telemetry + manifest.json per run") from the shell every
 * `experiments/*\/run.ts` script had reconstructed inline; see
 * `docs/explainers/0010-experiment-run-artifacts.md`.
 *
 * Revised 2026-11-09 for the experiment harness (`scripts/exp.ts`): the environment now records
 * everything manifest v2 needs to say *what* ran (dirty flag, lockfile hash, tfjs version,
 * platform), and this module no longer imports tfjs. The caller passes what it knows about tfjs
 * (`src/experiment/tfjsInfo.ts` reads it), so the harness's bookkeeping process, which never
 * builds a model, can use these helpers without the native binding installed.
 */

/** What `src/experiment/tfjsInfo.ts` reports. `null` where a run did not load tfjs at all. */
export interface TfjsInfo {
  /** The tfjs-node package version, e.g. `"4.22.0"`. */
  version: string;
  /** `tf.getBackend()` — which backend actually served the run's ops. */
  backend: string;
}

export interface RunEnvironment {
  /** `git rev-parse HEAD`. Throws if not run inside a git checkout. */
  gitCommit: string;
  /** True if the work tree had any modified, staged or untracked file when captured. */
  gitDirty: boolean;
  /** `process.version`, e.g. `"v22.18.0"`. */
  nodeVersion: string;
  tfjsVersion: string | null;
  tfjsBackend: string | null;
  /** `process.platform` and `process.arch`: float results are only expected to match within one pair. */
  platform: string;
  arch: string;
  /** SHA-256 of `package-lock.json`, or `null` if the checkout has none. */
  lockfileSha256: string | null;
  /** `new Date().toISOString()`, captured once, at call time. */
  createdAt: string;
}

/**
 * Captures what identifies *what ran*, independent of what the run measured. Call it before the
 * run writes anything: `gitDirty` is meant to describe the code, not the run's own outputs.
 */
export function captureRunEnvironment(tfjs: TfjsInfo | null, cwd: string = process.cwd()): RunEnvironment {
  const lockfile = join(cwd, "package-lock.json");
  return {
    ...gitState(cwd),
    nodeVersion: process.version,
    tfjsVersion: tfjs?.version ?? null,
    tfjsBackend: tfjs?.backend ?? null,
    platform: process.platform,
    arch: process.arch,
    lockfileSha256: existsSync(lockfile) ? hashFile(lockfile) : null,
    createdAt: new Date().toISOString(),
  };
}

/** Creates `dir` (and any missing parents) if it doesn't already exist. */
export function ensureRunDir(dir: string): void {
  mkdirSync(dir, { recursive: true });
}

/**
 * Writes `records` as JSON Lines (one `JSON.stringify`d record per line) to `fileName` inside
 * `dir`, and returns `fileName` — meant to be stored verbatim as a manifest's `telemetryFile`
 * field, so the manifest and the file it points at can't name-drift apart. `dir` must already
 * exist (see `ensureRunDir`); a trailing separator is accepted but not required.
 */
export function writeTelemetry(dir: string, records: readonly unknown[], fileName = "telemetry.jsonl"): string {
  const body = records.map((record) => JSON.stringify(record)).join("\n") + "\n";
  writeFileSync(join(dir, fileName), body);
  return fileName;
}

/**
 * Writes `manifest` as pretty-printed JSON (2-space indent, trailing newline) to `fileName`
 * inside `dir`. Imposes no shape: the harness's manifest v2 (`src/experiment/harness.ts`) and
 * one-off measurement scripts both go through here.
 */
export function writeManifest(dir: string, manifest: Record<string, unknown>, fileName = "manifest.json"): void {
  writeFileSync(join(dir, fileName), JSON.stringify(manifest, null, 2) + "\n");
}
