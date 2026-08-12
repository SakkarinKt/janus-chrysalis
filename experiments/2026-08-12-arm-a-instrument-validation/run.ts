/**
 * Arm-A instrument-validation milestone (`loop/GOAL.md` priority 4, second half; proposal
 * `0001`'s "L2 promotion request" gate). Runs proposal `0001`'s freeze intervention and
 * both-frozen control, 3 seeds each, with a real learning policy (`QLearningPolicy`,
 * `src/agent/policy.ts`, landed PR #43/#44) driving both agents and one independent
 * `WorldModel` per agent (Arm A: no sharing). Run with:
 *
 *   node experiments/2026-08-12-arm-a-instrument-validation/run.ts
 *
 * Writes per-run `manifest.json` + `telemetry.jsonl` under
 * `artifacts/2026-08-12-arm-a-instrument-validation/seed-<seed>-<condition>/`, and an
 * aggregate `results.summary.csv` at that directory's root. Per `.gitignore`'s existing
 * `artifacts/**` rule ("keep only manifests + summary CSVs"), only the `manifest.json`s and
 * the `.summary.csv` are committed — `telemetry.jsonl` is a real, reviewable-on-request
 * artifact on disk but not checked in, matching `loop/GOAL.md` boundaries ("every run writes
 * JSONL + manifest.json; commit only manifests + summary artifacts").
 *
 * ## Design choices this run makes
 *
 * - **Dims**: reuses `benchmark.ts`'s `ARM_A_CONFIG` (h=256, 8x4 categorical latent) — the
 *   same "PLAN.html risk table" placeholder, not re-derived here. Using the real Arm-A scale
 *   (not a small test-only config) is deliberate: this milestone's whole point is validating
 *   the *measurement instrument* at the scale the eventual sweep will actually run at, not a
 *   unit-test-speed stand-in.
 * - **freezeStep = 38 of horizon 75**: just past the environment's midpoint (`DEFAULT_CONFIG`,
 *   `src/env/types.ts`) — long enough pre-freeze for both Q-tables to have taken a meaningful
 *   number of update() calls (37 steps) with a matching post-freeze observation window (38
 *   steps) for `postFreezeLossSeries` to read a slope from. Not tuned against this
 *   environment; a placeholder in the same spirit as `QLearningPolicy`'s own defaults
 *   (docs/explainers/0008).
 * - **frozenAgentIndex = 0**, arbitrary but fixed across all 3 seeds for comparability.
 * - **Independent instances per condition, not a shared pre-freeze phase.** The tempting
 *   design would give intervention and control the same seed and let the pre-freeze phase
 *   run once, forking state at the freeze point — a true common-random-numbers pairing. That
 *   does NOT hold here: `RSSMCell`/`ObservationDecoder`'s `tf.layers.dense`/`gruCell` are
 *   built with no `kernelInitializer` seed (`src/model/rssm.ts:95-98`, `src/model/decoder.ts:26`),
 *   so two freshly-constructed `WorldModel`s draw different initial weights from tfjs's own
 *   unseeded global initializer even under an identical `runEpisode` seed — confirmed by this
 *   run's own diagnostic below. So intervention and control here are two fully independent
 *   `runEpisode` calls per seed, exactly the "ordinarily different seeds... trajectories
 *   diverge... regardless" case `docs/explainers/0007-drift-attributable-error-metric.md`
 *   already designed `postFreezeLossSeries`/`driftAttributableError` around (index-aligned by
 *   steps-since-freeze, not by absolute trajectory match) — this run doesn't need pre-freeze
 *   alignment to be valid, it just doesn't get it as a free variance-reduction bonus.
 * - **Weight-init determinism gap** (flagged, not fixed here): the finding above means
 *   `runEpisode`'s `seed` does not make a `WorldModel`-involving run fully bit-reproducible —
 *   only the policy-action and world-model-sampling streams are seeded; layer weight
 *   initialization is not. This is the same class of gap as the quality pass's finding #1
 *   (`sampleHard`, fixed 2026-08-04) but in the *initializer* rather than the *sampling* path,
 *   and previously unflagged. Not fixed in this run: the fix (threading a seed into
 *   `tf.initializers.glorotUniform({ seed })` or equivalent for every `tf.layers.*` call in
 *   `RSSMCell`/`ObservationDecoder`) touches those two files' public construction surface and
 *   several call sites, which is `src/model/` API-surface work belonging to whoever next
 *   touches that surface, not a same-run fix bundled into an experiment script. Recorded in
 *   this run's manifests' `findings` field and this run's stand-up report.
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import tf from "@tensorflow/tfjs-node";
import { CooperativeGridWorld } from "../../src/env/gridworld.ts";
import { QLearningPolicy } from "../../src/agent/policy.ts";
import type { QLearningConfig } from "../../src/agent/policy.ts";
import { WorldModel } from "../../src/model/worldModel.ts";
import type { WorldModelConfig } from "../../src/model/worldModel.ts";
import { runEpisode } from "../../src/experiment/freeze.ts";
import type { EpisodeStepRecord, FreezeCondition, FreezeConfig } from "../../src/experiment/freeze.ts";
import { driftAttributableError, postFreezeLossSeries } from "../../src/experiment/metrics.ts";

const RUN_ID = "2026-08-12-arm-a-instrument-validation";
const SEEDS = [1001, 1002, 1003];
const FREEZE_STEP = 38;
const FROZEN_AGENT_INDEX = 0;
const HORIZON = 75; // DEFAULT_CONFIG.horizon (src/env/types.ts) — not overridden.

// Same placeholder as experiments/2026-07-21-week3-stack-spike/benchmark.ts's ARM_A_CONFIG.
const RSSM_CONFIG = { deterministicSize: 256, latentCategoricals: 8, latentClasses: 4 };
const Q_LEARNING_CONFIG: Required<QLearningConfig> = { alpha: 0.1, gamma: 0.95, epsilon: 0.1 };

const artifactsDir = fileURLToPath(new URL(`../../artifacts/${RUN_ID}/`, import.meta.url));
const gitCommit = execSync("git rev-parse HEAD").toString().trim();

interface RunOutcome {
  seed: number;
  condition: FreezeCondition;
  postFreezeSteps: number;
  meanLoss: number;
  slopeLoss: number;
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** OLS slope of `values` against their index (0, 1, 2, ...) — a coarse "is it rising" read. */
function slope(values: number[]): number {
  const n = values.length;
  const xs = values.map((_, i) => i);
  const xMean = mean(xs);
  const yMean = mean(values);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i]! - xMean) * (values[i]! - yMean);
    den += (xs[i]! - xMean) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

function buildWorldModels(observationSize: number): [WorldModel, WorldModel] {
  const config: WorldModelConfig = { rssm: RSSM_CONFIG, observationSize };
  return [new WorldModel(config), new WorldModel(config)];
}

let weightInitDiagnostic: Record<string, unknown> | undefined;

function writeRun(
  seed: number,
  condition: FreezeCondition,
  records: EpisodeStepRecord[],
  series: number[],
): RunOutcome {
  const dir = `${artifactsDir}seed-${seed}-${condition}/`;
  mkdirSync(dir, { recursive: true });

  const telemetry = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
  writeFileSync(`${dir}telemetry.jsonl`, telemetry);

  const outcome: RunOutcome = {
    seed,
    condition,
    postFreezeSteps: series.length,
    meanLoss: mean(series),
    slopeLoss: slope(series),
  };

  const manifest = {
    runId: RUN_ID,
    seed,
    condition,
    frozenAgentIndex: condition === "intervention" ? FROZEN_AGENT_INDEX : null,
    freezeStep: FREEZE_STEP,
    horizon: HORIZON,
    rssmConfig: RSSM_CONFIG,
    qLearningConfig: Q_LEARNING_CONFIG,
    gitCommit,
    nodeVersion: process.version,
    tfjsBackend: tf.getBackend(),
    createdAt: new Date().toISOString(),
    telemetryFile: "telemetry.jsonl",
    resultSummary: outcome,
    findings: [
      {
        severity: "NOTE",
        confidence: "self_checked, high confidence",
        summary:
          "RSSMCell/ObservationDecoder layer weight init (tf.layers.dense/gruCell) is not seeded " +
          "through this project's Rng — see this file's header comment and weightInitDiagnostic below. " +
          "Not fixed in this run.",
        weightInitDiagnostic,
      },
    ],
  };
  writeFileSync(`${dir}manifest.json`, JSON.stringify(manifest, null, 2) + "\n");

  return outcome;
}

/**
 * Confirms (once, not per-seed) that two freshly constructed WorldModels with identical
 * config draw different initial weights — the finding this run's header comment documents.
 * Returns a plain-object diagnostic for the manifest, and disposes its own tensors.
 */
function checkWeightInitDeterminism(observationSize: number): Record<string, unknown> {
  const [a, b] = buildWorldModels(observationSize);
  const aWeights = a.cell.trainableWeights().map((w) => Array.from(w.dataSync()).slice(0, 4));
  const bWeights = b.cell.trainableWeights().map((w) => Array.from(w.dataSync()).slice(0, 4));
  a.dispose();
  b.dispose();
  return {
    claim: "two fresh WorldModels, identical config, first 4 weight values of each trainable tensor",
    modelA: aWeights,
    modelB: bWeights,
    identical: JSON.stringify(aWeights) === JSON.stringify(bWeights),
  };
}

function runOneCondition(seed: number, condition: FreezeCondition, observationSize: number): {
  outcome: RunOutcome;
  series: number[];
} {
  const env = new CooperativeGridWorld({ seed, horizon: HORIZON });
  const policies = [new QLearningPolicy(Q_LEARNING_CONFIG), new QLearningPolicy(Q_LEARNING_CONFIG)];
  const [wm0, wm1] = buildWorldModels(observationSize);

  const freezeConfig: FreezeConfig =
    condition === "control"
      ? { freezeStep: FREEZE_STEP, condition: "control" }
      : { freezeStep: FREEZE_STEP, condition: "intervention", frozenAgentIndex: FROZEN_AGENT_INDEX };

  const records = runEpisode(env, policies, seed, freezeConfig, [wm0, wm1]);
  const series = postFreezeLossSeries(records, FREEZE_STEP, FROZEN_AGENT_INDEX);

  // Only the two state tensors are expected to free here — RSSMCell/decoder weight
  // Variables and each trained variable's lazily-created Adam m/v slots (allocated once,
  // on that variable's first train=true step) are meant to persist for the model's whole
  // lifetime; tf.memory() before/after this dispose() is not a meaningful leak signal by
  // itself (confirmed isolated: 150 WorldModel.step() calls at these dims hold numTensors
  // flat once past the first train=true call — see this run's stand-up report). The
  // existing tensor-lifecycle contract (docs/explainers/0005) is what's actually being
  // relied on here, already covered by test/model/worldModel.test.ts's own leak check.
  wm0.dispose();
  wm1.dispose();

  const outcome = writeRun(seed, condition, records, series);
  return { outcome, series };
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const probeEnv = new CooperativeGridWorld({ seed: 0, horizon: HORIZON });
  const observationSize = probeEnv.observationLength;
  weightInitDiagnostic = checkWeightInitDeterminism(observationSize);

  const rows: RunOutcome[] = [];
  const diffs: { seed: number; diffMean: number; diffSlope: number }[] = [];

  for (const seed of SEEDS) {
    const control = runOneCondition(seed, "control", observationSize);
    const intervention = runOneCondition(seed, "intervention", observationSize);
    rows.push(control.outcome, intervention.outcome);

    const diff = driftAttributableError(intervention.series, control.series);
    diffs.push({ seed, diffMean: mean(diff), diffSlope: slope(diff) });

    console.log(
      `seed ${seed}: control meanLoss=${control.outcome.meanLoss.toFixed(4)} slope=${control.outcome.slopeLoss.toFixed(6)} | ` +
        `intervention meanLoss=${intervention.outcome.meanLoss.toFixed(4)} slope=${intervention.outcome.slopeLoss.toFixed(6)} | ` +
        `diffMean=${mean(diff).toFixed(4)} diffSlope=${slope(diff).toFixed(6)}`,
    );
  }

  const csvHeader = "seed,condition,postFreezeSteps,meanLoss,slopeLoss\n";
  const csvBody = rows
    .map((r) => `${r.seed},${r.condition},${r.postFreezeSteps},${r.meanLoss},${r.slopeLoss}`)
    .join("\n");
  writeFileSync(`${artifactsDir}results.summary.csv`, csvHeader + csvBody + "\n");

  const controlSlopes = rows.filter((r) => r.condition === "control").map((r) => r.slopeLoss);
  const diffMeans = diffs.map((d) => d.diffMean);
  console.log("\n--- Gate check (descriptive, n=3, not a formal significance test) ---");
  console.log(`control slopes: ${controlSlopes.map((s) => s.toFixed(6)).join(", ")}`);
  console.log(`per-seed diff means (intervention - control): ${diffMeans.map((d) => d.toFixed(4)).join(", ")}`);
  console.log(`mean control slope: ${mean(controlSlopes).toFixed(6)}`);
  console.log(`mean diff mean: ${mean(diffMeans).toFixed(4)}`);
}

main();
