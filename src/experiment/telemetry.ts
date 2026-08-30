import { execSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import tf from "@tensorflow/tfjs-node";

/**
 * The run-identity fields every `experiments/*\/run.ts` script since
 * `2026-08-12-arm-a-instrument-validation` has independently reconstructed
 * (git commit, Node version, tfjs backend, a timestamp) before writing its
 * own ad hoc `manifest.json` — first extracted here (`loop/GOAL.md`
 * priority 5, "JSONL telemetry + manifest.json per run"). See
 * `docs/explainers/0010-experiment-run-artifacts.md` for why this only
 * factors out the shared shell, not a fixed manifest schema, and why
 * existing `run.ts` scripts are left untouched.
 */
export interface RunEnvironment {
  /** `git rev-parse HEAD`, trimmed. Throws if not run inside a git checkout. */
  gitCommit: string;
  /** `process.version`, e.g. `"v22.18.0"`. */
  nodeVersion: string;
  /** `tf.getBackend()` — which tfjs backend actually served this run's ops. */
  tfjsBackend: string;
  /** `new Date().toISOString()`, captured once, at call time — not per-record. */
  createdAt: string;
}

/**
 * Captures the four fields that identify *what ran*, independent of what
 * the run itself measured. Every existing experiment script calls the
 * equivalent of this inline, once, near the top of `main()`; this is that
 * pattern with one call site instead of eleven.
 */
export function captureRunEnvironment(): RunEnvironment {
  return {
    gitCommit: execSync("git rev-parse HEAD").toString().trim(),
    nodeVersion: process.version,
    tfjsBackend: tf.getBackend(),
    createdAt: new Date().toISOString(),
  };
}

/** Creates `dir` (and any missing parents) if it doesn't already exist. */
export function ensureRunDir(dir: string): void {
  mkdirSync(dir, { recursive: true });
}

/**
 * Writes `records` as JSON Lines (one `JSON.stringify`d record per line) to
 * `${dir}${fileName}`, and returns `fileName` — meant to be stored verbatim
 * as a manifest's `telemetryFile` field, so the manifest and the file it
 * points at can't name-drift apart. `dir` must already exist (see
 * `ensureRunDir`) and end with a trailing separator, matching every
 * existing call site's `${artifactsDir}seed-${seed}-${condition}/` style.
 */
export function writeTelemetry(
  dir: string,
  records: readonly unknown[],
  fileName = "telemetry.jsonl",
): string {
  const body = records.map((record) => JSON.stringify(record)).join("\n") + "\n";
  writeFileSync(`${dir}${fileName}`, body);
  return fileName;
}

/**
 * Writes `manifest` as pretty-printed JSON to `${dir}${fileName}`. Doesn't
 * impose a manifest shape — every run's manifest fields differ (seed,
 * condition, model config, result summary, findings, ...) — only the
 * writing convention (2-space indent, trailing newline) that every existing
 * script already used, but reimplemented.
 */
export function writeManifest(
  dir: string,
  manifest: Record<string, unknown>,
  fileName = "manifest.json",
): void {
  writeFileSync(`${dir}${fileName}`, JSON.stringify(manifest, null, 2) + "\n");
}
