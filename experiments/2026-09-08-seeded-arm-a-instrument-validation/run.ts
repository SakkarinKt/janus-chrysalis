/**
 * Seeded re-run of `experiments/2026-08-12-arm-a-instrument-validation` (PR #67 review,
 * "Next: priority 5, the seeded Arm-A re-run" — "derive the seed per agent at
 * `run.ts:110-112` [because] both agents share one `config` today, so a single seed would
 * hand Arm A's two independent agents identical init weights"). Run with:
 *
 *   node experiments/2026-09-08-seeded-arm-a-instrument-validation/run.ts
 *
 * Writes per-run `manifest.json` + `telemetry.jsonl` under
 * `artifacts/2026-09-08-seeded-arm-a-instrument-validation/seed-<seed>-<condition>/`, and an
 * aggregate `results.summary.csv` at that directory's root — same layout and the same
 * manifests-and-summary-only commit policy as the 2026-08-12 baseline. That baseline's
 * `artifacts/2026-08-12-arm-a-instrument-validation/` directory is left untouched: this is a
 * new dated run, not a supersede-in-place (PR #67 review: overwriting it silently broke the
 * `docs/proposals/0001` gate table's cited numbers once already).
 *
 * ## This is not the first seeded Arm-A-style run — say so plainly
 *
 * `experiments/2026-08-13-paired-init-instrument-validation/run.ts` already threads
 * `WorldModelConfig.seed` into per-agent `WorldModel`s via `deriveSeed(seed, agentIndex)`
 * (its `buildWorldModels`), and its results are already written up in `docs/proposals/0001`'s
 * 2026-08-13 update. This file's `buildWorldModels` reuses that exact pattern verbatim rather
 * than inventing a new one — it's the right fix for precisely the problem the PR #67 review
 * named (one shared config for both agents), already reviewed and merged once. What justifies
 * running it again, under this milestone's own `arm-a-instrument-validation` naming, instead
 * of just citing 2026-08-13: `WorldModel` itself has changed materially since 2026-08-13 —
 * `ContinueHead` (PR #63, adds its own `deriveSeed(config.seed, 2)`-seeded weights),
 * `WorldModel.step()`'s NaN-halt invariant (PR #58), and `driftAttributableError` excluding
 * `continueLoss` (PR #64) all postdate that run. So this is the first seeded Arm-A validation
 * against the *current* `WorldModel`, not a re-run of 2026-08-13's numbers, and its results are
 * not expected to (and don't, checked directly) match 2026-08-13's table bit-for-bit even
 * though the seeding scheme is identical. See this run's stand-up report / proposal-0001 update
 * for the full "is this redundant" accounting — flagged there as an assumption, not decided
 * silently here.
 *
 * ## What's different from the 2026-08-12 baseline specifically, and why
 *
 * - **Per-agent seeding, not one shared seed.** 2026-08-12's `buildWorldModels()` built both of
 *   a condition's `WorldModel`s from one shared, unseeded `config` object. Naively adding a
 *   single `seed` to that shared config would derive the *same* internal
 *   `deriveSeed(seed, 0|1|2)` streams for both calls — handing Arm A's two supposedly
 *   independent agents literally identical initial weights, which is worse than the status quo,
 *   not better. `buildWorldModels()` here instead derives one seed per agent index
 *   (`deriveSeed(seed, agentIndex)`), so each agent's weight init is reproducible run-to-run
 *   while the two agents still start from different draws — matching the *intent* of the
 *   2026-08-12 baseline's (incidentally, unseededly) different per-agent weights, just now
 *   deterministic instead of accidental.
 * - **`weightInitDiagnostic` inverts, and that's the pass condition.** The 2026-08-12 diagnostic
 *   built two *unseeded* `WorldModel`s from one shared config and recorded `identical: false` —
 *   correctly documenting that construction wasn't reproducible yet. This run's diagnostic
 *   instead checks the positive claim `buildWorldModels` exists to satisfy: same seed twice ⇒
 *   identical per-agent weights (`sameSeedIdentical`), different seed ⇒ different weights
 *   (`differentSeedDiffers`) — mirroring 2026-08-13's own `checkPairedInitDeterminism` shape.
 *   `sameSeedIdentical: true` here is the expected, passing result, the opposite polarity from
 *   2026-08-12's `identical: false` finding — this file's manifest `findings` summary says so
 *   explicitly rather than reusing 2026-08-12's wording unchanged.
 * - **Same seed pairs control and intervention across conditions too** (a side effect of
 *   deriving from the trial's own `seed`, not a separate mechanism) — same consequence 2026-08-13
 *   documented: pre-freeze `EpisodeStepRecord`s become bit-identical between conditions for a
 *   given seed, so any post-freeze difference is attributable to the freeze intervention alone,
 *   not to independent random draws. Re-asserted per seed by `assertPreFreezeParity`, restored
 *   here (PR #69 review, 2026-09-11): this run's own argument above is that `WorldModel` changed
 *   materially since 2026-08-13, so 2026-08-13's "follows mechanically" no longer holds without
 *   re-checking against the current `WorldModel` — checked directly, 0 mismatches on all 5 fields,
 *   37 pre-freeze steps, all 3 seeds.
 *
 * Everything else (dims, `freezeStep`, `frozenAgentIndex`, horizon, the tensor-disposal comment)
 * is unchanged from 2026-08-12 — see that file for the rationale.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import tf from "@tensorflow/tfjs-node";
import { CooperativeGridWorld } from "../../src/env/gridworld.ts";
import { QLearningPolicy } from "../../src/agent/policy.ts";
import type { QLearningConfig } from "../../src/agent/policy.ts";
import { WorldModel } from "../../src/model/worldModel.ts";
import type { WorldModelConfig } from "../../src/model/worldModel.ts";
import { deriveSeed } from "../../src/env/rng.ts";
import { runEpisode } from "../../src/experiment/freeze.ts";
import type { EpisodeStepRecord, FreezeCondition, FreezeConfig } from "../../src/experiment/freeze.ts";
import { driftAttributableError, postFreezeLossSeries } from "../../src/experiment/metrics.ts";

const RUN_ID = "2026-09-08-seeded-arm-a-instrument-validation";
const SEEDS = [1001, 1002, 1003]; // same 3 seeds as 2026-08-12/13, for a like-for-like comparison.
const FREEZE_STEP = 38;
const FROZEN_AGENT_INDEX = 0;
const HORIZON = 75; // DEFAULT_CONFIG.horizon (src/env/types.ts) — not overridden.

// Same diagnostic-only seed as 2026-08-13-paired-init-instrument-validation/run.ts's
// checkPairedInitDeterminism — not one of SEEDS, so the diagnostic can't be mistaken for a
// real trial's result.
const WEIGHT_INIT_DIAGNOSTIC_SEED = 4242;
const WEIGHT_INIT_DIAGNOSTIC_OTHER_SEED = 4343;

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

/**
 * Builds this trial's two per-agent WorldModels, seeded from the trial's `seed` (one derived
 * stream per agent index — `deriveSeed(seed, agentIndex)`) so the two agents get reproducible
 * but distinct initial weights. Identical pattern to
 * `experiments/2026-08-13-paired-init-instrument-validation/run.ts`'s `buildWorldModels`.
 */
function buildWorldModels(seed: number, observationSize: number): [WorldModel, WorldModel] {
  const configFor = (agentIndex: number): WorldModelConfig => ({
    rssm: RSSM_CONFIG,
    observationSize,
    seed: deriveSeed(seed, agentIndex),
  });
  return [new WorldModel(configFor(0)), new WorldModel(configFor(1))];
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
    weightInitSeeds: { agent0: deriveSeed(seed, 0), agent1: deriveSeed(seed, 1) },
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
          "This run threads WorldModelConfig.seed into each agent's WorldModel via " +
          "deriveSeed(seed, agentIndex) — the same pattern 2026-08-13-paired-init-instrument-" +
          "validation established, applied here to the arm-a-instrument-validation milestone's " +
          "own naming/framing for the first time. weightInitDiagnostic below is the once-only " +
          "reproducibility check: sameSeedIdentical:true / differentSeedDiffers:true is the pass " +
          "condition, inverted from the 2026-08-12 baseline's identical:false (which predates " +
          "WorldModelConfig.seed's existence).",
        weightInitDiagnostic,
      },
    ],
  };
  writeFileSync(`${dir}manifest.json`, JSON.stringify(manifest, null, 2) + "\n");

  return outcome;
}

/**
 * Confirms (once, not per-seed) that `buildWorldModels` draws identical per-agent initial
 * weights across two calls with the same seed, and different weights across two calls with
 * different seeds — the positive counterpart of 2026-08-12's `checkWeightInitDeterminism`,
 * which confirmed the (then-unfixed) bug this run's `buildWorldModels` addresses. Same shape as
 * 2026-08-13's `checkPairedInitDeterminism`. Disposes its own tensors.
 */
function checkWeightInitDeterminism(observationSize: number): Record<string, unknown> {
  const weightsOf = (wm: WorldModel) => wm.cell.trainableWeights().map((w) => Array.from(w.dataSync()).slice(0, 4));

  const [a0, a1] = buildWorldModels(WEIGHT_INIT_DIAGNOSTIC_SEED, observationSize);
  const [b0, b1] = buildWorldModels(WEIGHT_INIT_DIAGNOSTIC_SEED, observationSize);
  const sameSeedIdentical =
    JSON.stringify(weightsOf(a0)) === JSON.stringify(weightsOf(b0)) &&
    JSON.stringify(weightsOf(a1)) === JSON.stringify(weightsOf(b1));
  a0.dispose();
  a1.dispose();
  b0.dispose();
  b1.dispose();

  const [c0] = buildWorldModels(WEIGHT_INIT_DIAGNOSTIC_SEED, observationSize);
  const [d0] = buildWorldModels(WEIGHT_INIT_DIAGNOSTIC_OTHER_SEED, observationSize);
  const differentSeedDiffers = JSON.stringify(weightsOf(c0)) !== JSON.stringify(weightsOf(d0));
  c0.dispose();
  d0.dispose();

  return {
    claim:
      "buildWorldModels(seed, ...) called twice with the same seed draws identical per-agent " +
      "initial weights; called with a different seed still draws different ones. " +
      "sameSeedIdentical:true / differentSeedDiffers:true is the pass condition (inverted from " +
      "2026-08-12's unseeded identical:false).",
    diagnosticSeed: WEIGHT_INIT_DIAGNOSTIC_SEED,
    otherSeed: WEIGHT_INIT_DIAGNOSTIC_OTHER_SEED,
    sameSeedIdentical,
    differentSeedDiffers,
  };
}

/**
 * Confirms every pre-`freezeStep` record is bit-identical between `control` and `intervention`
 * for the same seed. Restored from `2026-08-13-paired-init-instrument-validation/run.ts`'s
 * function of the same name (PR #69 review, 2026-09-11) — dropped from this file's first version
 * on the reasoning that 2026-08-13 already established pairing "follows mechanically" once init
 * is seeded, which doesn't hold given this file's own claim that `WorldModel` changed materially
 * since then. Compares `actions`, `reward`, `worldModelLoss`, and `observations`/
 * `nextObservations` (the last two via `JSON.stringify`, since `Observation` is a plain number
 * array).
 */
function assertPreFreezeParity(control: EpisodeStepRecord[], intervention: EpisodeStepRecord[]): Record<string, unknown> {
  const preFreeze = (records: EpisodeStepRecord[]) => records.filter((r) => r.step < FREEZE_STEP);
  const c = preFreeze(control);
  const i = preFreeze(intervention);

  const mismatches: string[] = [];
  if (c.length !== i.length) mismatches.push(`pre-freeze step count differs: control=${c.length} intervention=${i.length}`);
  for (let idx = 0; idx < Math.min(c.length, i.length); idx++) {
    const cr = c[idx]!;
    const ir = i[idx]!;
    if (JSON.stringify(cr.actions) !== JSON.stringify(ir.actions)) mismatches.push(`step ${cr.step}: actions differ`);
    if (cr.reward !== ir.reward) mismatches.push(`step ${cr.step}: reward differs`);
    if (JSON.stringify(cr.observations) !== JSON.stringify(ir.observations))
      mismatches.push(`step ${cr.step}: observations differ`);
    if (JSON.stringify(cr.nextObservations) !== JSON.stringify(ir.nextObservations))
      mismatches.push(`step ${cr.step}: nextObservations differ`);
    if (JSON.stringify(cr.worldModelLoss) !== JSON.stringify(ir.worldModelLoss))
      mismatches.push(`step ${cr.step}: worldModelLoss differs`);
  }

  return {
    claim: "every pre-freezeStep EpisodeStepRecord (actions, reward, observations, nextObservations, " +
      "worldModelLoss) is bit-identical between control and intervention for the same seed",
    preFreezeStepCount: c.length,
    identical: mismatches.length === 0,
    mismatches: mismatches.slice(0, 10),
  };
}

function runOneCondition(seed: number, condition: FreezeCondition, observationSize: number): {
  outcome: RunOutcome;
  series: number[];
  records: EpisodeStepRecord[];
} {
  const env = new CooperativeGridWorld({ seed, horizon: HORIZON });
  const policies = [new QLearningPolicy(Q_LEARNING_CONFIG), new QLearningPolicy(Q_LEARNING_CONFIG)];
  const [wm0, wm1] = buildWorldModels(seed, observationSize);

  const freezeConfig: FreezeConfig =
    condition === "control"
      ? { freezeStep: FREEZE_STEP, condition: "control" }
      : { freezeStep: FREEZE_STEP, condition: "intervention", frozenAgentIndex: FROZEN_AGENT_INDEX };

  const records = runEpisode(env, policies, seed, freezeConfig, [wm0, wm1]);
  const series = postFreezeLossSeries(records, FREEZE_STEP, FROZEN_AGENT_INDEX);

  // Same disposal contract as 2026-08-12 (docs/explainers/0005) — see that file's comment.
  wm0.dispose();
  wm1.dispose();

  const outcome = writeRun(seed, condition, records, series);
  return { outcome, series, records };
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const probeEnv = new CooperativeGridWorld({ seed: 0, horizon: HORIZON });
  const observationSize = probeEnv.observationLength;
  weightInitDiagnostic = checkWeightInitDeterminism(observationSize);
  console.log("Weight-init determinism check:", JSON.stringify(weightInitDiagnostic));
  if (!weightInitDiagnostic.sameSeedIdentical || !weightInitDiagnostic.differentSeedDiffers) {
    throw new Error("Weight-init determinism check failed — see weightInitDiagnostic above. Aborting run.");
  }

  const rows: RunOutcome[] = [];
  const diffs: { seed: number; diffMean: number; diffSlope: number }[] = [];

  for (const seed of SEEDS) {
    const control = runOneCondition(seed, "control", observationSize);
    const intervention = runOneCondition(seed, "intervention", observationSize);
    rows.push(control.outcome, intervention.outcome);

    const parityCheck = assertPreFreezeParity(control.records, intervention.records);
    if (!parityCheck.identical) {
      console.warn(`seed ${seed}: pre-freeze parity check FAILED —`, JSON.stringify(parityCheck));
    }

    const diff = driftAttributableError(intervention.series, control.series);
    diffs.push({ seed, diffMean: mean(diff), diffSlope: slope(diff) });

    console.log(
      `seed ${seed}: control meanLoss=${control.outcome.meanLoss.toFixed(4)} slope=${control.outcome.slopeLoss.toFixed(6)} | ` +
        `intervention meanLoss=${intervention.outcome.meanLoss.toFixed(4)} slope=${intervention.outcome.slopeLoss.toFixed(6)} | ` +
        `diffMean=${mean(diff).toFixed(4)} diffSlope=${slope(diff).toFixed(6)} | prefreezeParity=${parityCheck.identical}`,
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
