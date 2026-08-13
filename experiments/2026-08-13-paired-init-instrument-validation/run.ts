/**
 * Re-runs `loop/GOAL.md` priority 1 (processing PR #45's review) — the same 3-seed Arm-A
 * freeze-vs-both-frozen instrument validation as
 * `experiments/2026-08-12-arm-a-instrument-validation/run.ts`, but with the weight-init-
 * determinism gap that run flagged now fixed (`RSSMConfig.seed`/`DecoderConfig.seed`/
 * `WorldModelConfig.seed`, `src/model/rssm.ts`, `src/model/decoder.ts`, `src/model/worldModel.ts`)
 * and used to *pair* control and intervention on the same per-agent initial weights, per the
 * review's "seed or pair world-model init before spending compute on a longer pre-freeze
 * horizon" (@SakkarinKt, PR #45, 2026-08-12):
 *
 *   node experiments/2026-08-13-paired-init-instrument-validation/run.ts
 *
 * ## What pairing buys, and why it was free once init is seeded
 *
 * `src/experiment/freeze.ts`'s `runEpisode(env, policies, seed, freezeConfig, worldModels)`
 * builds its `policyRng`/`worldModelRngs` from `seed` alone, and `freezeConfig` only gates
 * whether `policy.update()`/`WorldModel.step(..., train)` run — it never changes which action
 * gets chosen or which RNG value gets drawn. So for a fixed `seed`, control's and intervention's
 * pre-`freezeStep` env transitions, policy Q-table updates, and world-model losses were already
 * bound to be identical *draw-for-draw* — the only thing standing between that and true common-
 * random-numbers pairing was `WorldModel`'s two instances (one per condition, same seed) drawing
 * different *initial* weights from tfjs's unseeded global initializer. Passing the same
 * `WorldModelConfig.seed` to both conditions' `WorldModel`s removes that last source of
 * divergence — this run's `assertPreFreezeParity` diagnostic below confirms the pre-freeze
 * `EpisodeStepRecord`s now match bit-for-bit, not just approximately.
 *
 * Post-`freezeStep`, the two conditions still diverge (that divergence, driven only by the freeze
 * intervention itself, is the whole point) — pairing only removes *pre*-freeze and *initialization*
 * noise, not the effect being measured.
 *
 * ## Design choices carried over unchanged from 2026-08-12's run
 *
 * Same `SEEDS`, `FREEZE_STEP` (38 of horizon 75), `FROZEN_AGENT_INDEX` (0), `RSSM_CONFIG`
 * (Arm-A dims), `Q_LEARNING_CONFIG` — this run isolates the init-pairing variable only; it does
 * not touch the pre-freeze-horizon question the 2026-08-12 stand-up raised as a separate,
 * *unconfirmed* hypothesis (todo, not this run's job — see that run's proposal-0001 update log
 * entry and this run's own for whether pairing alone was enough to make gate (b) readable).
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

const RUN_ID = "2026-08-13-paired-init-instrument-validation";
const SEEDS = [1001, 1002, 1003];
const FREEZE_STEP = 38;
const FROZEN_AGENT_INDEX = 0;
const HORIZON = 75; // DEFAULT_CONFIG.horizon (src/env/types.ts) — not overridden, matches 2026-08-12's run.

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
 * stream per agent index) so calling this twice with the same `seed` — once for control, once
 * for intervention — draws identical initial weights both times. The pairing this run exists to
 * add.
 */
function buildWorldModels(seed: number, observationSize: number): [WorldModel, WorldModel] {
  const configFor = (agentIndex: number): WorldModelConfig => ({
    rssm: RSSM_CONFIG,
    observationSize,
    seed: deriveSeed(seed, agentIndex),
  });
  return [new WorldModel(configFor(0)), new WorldModel(configFor(1))];
}

/**
 * Confirms (once) that `buildWorldModels` now draws identical initial weights across two calls
 * with the same seed, and different weights across two calls with different seeds — the positive
 * counterpart of 2026-08-12's `checkWeightInitDeterminism`, which confirmed the bug this run's
 * fix addresses.
 */
function checkPairedInitDeterminism(observationSize: number): Record<string, unknown> {
  const weightsOf = (wm: WorldModel) => wm.cell.trainableWeights().map((w) => Array.from(w.dataSync()).slice(0, 4));

  const [a0, a1] = buildWorldModels(4242, observationSize);
  const [b0, b1] = buildWorldModels(4242, observationSize);
  const sameSeedIdentical =
    JSON.stringify(weightsOf(a0)) === JSON.stringify(weightsOf(b0)) &&
    JSON.stringify(weightsOf(a1)) === JSON.stringify(weightsOf(b1));
  a0.dispose();
  a1.dispose();
  b0.dispose();
  b1.dispose();

  const [c0] = buildWorldModels(4242, observationSize);
  const [d0] = buildWorldModels(4343, observationSize);
  const differentSeedDiffers = JSON.stringify(weightsOf(c0)) !== JSON.stringify(weightsOf(d0));
  c0.dispose();
  d0.dispose();

  return {
    claim: "buildWorldModels(seed, ...) called twice with the same seed now draws identical initial " +
      "weights; called with different seeds still draws different ones",
    sameSeedIdentical,
    differentSeedDiffers,
  };
}

/**
 * Confirms every pre-`freezeStep` record is bit-identical between `control` and `intervention`
 * for the same seed — the direct evidence that pairing now holds end to end (env transitions,
 * policy Q-table state, world-model losses), not just at the weight-init step. Compares
 * `actions`, `reward`, `worldModelLoss`, and `observations`/`nextObservations` (the last two via
 * `JSON.stringify`, since `Observation` is a plain number array).
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

function writeRun(
  seed: number,
  condition: FreezeCondition,
  records: EpisodeStepRecord[],
  series: number[],
  parityCheck: Record<string, unknown> | undefined,
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
          "World-model init is now seeded and paired across control/intervention for this seed " +
          "(WorldModelConfig.seed, processing PR #45's review) — see preFreezeParityCheck.",
        preFreezeParityCheck: parityCheck,
      },
    ],
  };
  writeFileSync(`${dir}manifest.json`, JSON.stringify(manifest, null, 2) + "\n");

  return outcome;
}

function runOneCondition(
  seed: number,
  condition: FreezeCondition,
  observationSize: number,
): { outcome: RunOutcome; series: number[]; records: EpisodeStepRecord[] } {
  const env = new CooperativeGridWorld({ seed, horizon: HORIZON });
  const policies = [new QLearningPolicy(Q_LEARNING_CONFIG), new QLearningPolicy(Q_LEARNING_CONFIG)];
  const [wm0, wm1] = buildWorldModels(seed, observationSize);

  const freezeConfig: FreezeConfig =
    condition === "control"
      ? { freezeStep: FREEZE_STEP, condition: "control" }
      : { freezeStep: FREEZE_STEP, condition: "intervention", frozenAgentIndex: FROZEN_AGENT_INDEX };

  const records = runEpisode(env, policies, seed, freezeConfig, [wm0, wm1]);
  const series = postFreezeLossSeries(records, FREEZE_STEP, FROZEN_AGENT_INDEX);

  wm0.dispose();
  wm1.dispose();

  return { outcome: { seed, condition, postFreezeSteps: series.length, meanLoss: mean(series), slopeLoss: slope(series) }, series, records };
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const probeEnv = new CooperativeGridWorld({ seed: 0, horizon: HORIZON });
  const observationSize = probeEnv.observationLength;
  const pairedInitDiagnostic = checkPairedInitDeterminism(observationSize);
  console.log("Paired-init determinism check:", JSON.stringify(pairedInitDiagnostic));
  if (!pairedInitDiagnostic.sameSeedIdentical || !pairedInitDiagnostic.differentSeedDiffers) {
    throw new Error("Paired-init determinism check failed — see pairedInitDiagnostic above. Aborting run.");
  }

  const rows: RunOutcome[] = [];
  const diffs: { seed: number; diffMean: number; diffSlope: number }[] = [];

  for (const seed of SEEDS) {
    const control = runOneCondition(seed, "control", observationSize);
    const intervention = runOneCondition(seed, "intervention", observationSize);

    const parityCheck = assertPreFreezeParity(control.records, intervention.records);
    if (!parityCheck.identical) {
      console.warn(`seed ${seed}: pre-freeze parity check FAILED —`, JSON.stringify(parityCheck));
    }

    rows.push(
      writeRun(seed, "control", control.records, control.series, parityCheck),
      writeRun(seed, "intervention", intervention.records, intervention.series, parityCheck),
    );

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
  console.log(`diff mean sign consistent across seeds: ${diffMeans.every((d) => d > 0) || diffMeans.every((d) => d < 0)}`);
}

main();
