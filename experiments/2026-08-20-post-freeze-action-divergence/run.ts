/**
 * Processes PR #46's review (@SakkarinKt, 2026-08-13, merge comment on the paired-init fix):
 * "Seed 1003's `diffMean` is exactly 0.0000, so only 2 of 3 seeds show any post-freeze
 * behavioural divergence at all — effective n=2, and 'agent 1's actions never diverged' is
 * inferred from equal loss series and only checkable against gitignored telemetry. Record a
 * per-seed count of post-freeze steps where the two conditions' actions differ, next to
 * `preFreezeParityCheck`: that's the instrument's power metric, and it turns the
 * underpowered-horizon story into a measurement." Named as the prerequisite to land "before the
 * longer horizon" (multi-episode training-then-freeze) follow-up.
 *
 *   node experiments/2026-08-20-post-freeze-action-divergence/run.ts
 *
 * Re-runs the identical 3-seed paired-init Arm-A instrument validation
 * (`experiments/2026-08-13-paired-init-instrument-validation/run.ts` — same `SEEDS`,
 * `FREEZE_STEP`, `FROZEN_AGENT_INDEX`, `HORIZON`, `RSSM_CONFIG`, `Q_LEARNING_CONFIG`; paired
 * `WorldModelConfig.seed` per agent), so today's numbers reproduce 2026-08-13's bit-for-bit —
 * this run adds one new measurement (`postFreezeActionDivergenceCount`,
 * `src/experiment/metrics.ts`) on top rather than re-deriving the pairing result, which PR #46
 * already confirmed (`assertPreFreezeParity`, `identical: true`, all 3 seeds).
 *
 * `postFreezeActionDivergenceCount` counts, per seed, how many of the post-freeze steps had
 * control and intervention pick different joint actions. This distinguishes "the freeze
 * intervention had no effect" from "the post-freeze window never gave the two conditions a
 * chance to differ at all" — the second reading is what a `divergentSteps === 0` result would
 * mean, and is exactly what seed 1003's `diffMean === 0.0000` left ambiguous between.
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
import { driftAttributableError, postFreezeActionDivergenceCount, postFreezeLossSeries } from "../../src/experiment/metrics.ts";

const RUN_ID = "2026-08-20-post-freeze-action-divergence";
const SEEDS = [1001, 1002, 1003];
const FREEZE_STEP = 38;
const FROZEN_AGENT_INDEX = 0;
const HORIZON = 75; // Matches 2026-08-12's and 2026-08-13's runs.

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

/** Same pairing as 2026-08-13's `buildWorldModels` — one seed stream per agent index. */
function buildWorldModels(seed: number, observationSize: number): [WorldModel, WorldModel] {
  const configFor = (agentIndex: number): WorldModelConfig => ({
    rssm: RSSM_CONFIG,
    observationSize,
    seed: deriveSeed(seed, agentIndex),
  });
  return [new WorldModel(configFor(0)), new WorldModel(configFor(1))];
}

/** Same diagnostic as 2026-08-13's run — confirms pairing still holds on this commit before spending the run. */
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
  parityCheck: Record<string, unknown>,
  divergence: { postFreezeSteps: number; divergentSteps: number },
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
          "World-model init is seeded and paired across control/intervention for this seed " +
          "(WorldModelConfig.seed, PR #46) — see preFreezeParityCheck.",
        preFreezeParityCheck: parityCheck,
      },
      {
        severity: "NOTE",
        confidence: "self_checked, high confidence",
        summary:
          "Post-freeze action-divergence count (PR #46 review follow-up 1, 2026-08-13): of this " +
          "seed's post-freeze steps, how many had control and intervention pick different joint " +
          "actions. 0 means the post-freeze window never gave the two conditions a chance to " +
          "diverge at all — distinct from, and not detectable by, diffMean alone.",
        postFreezeActionDivergenceCount: divergence,
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

  const rows: RunOutcome[] = [];
  const diffs: { seed: number; diffMean: number; diffSlope: number }[] = [];
  const divergences: { seed: number; postFreezeSteps: number; divergentSteps: number }[] = [];

  for (const seed of SEEDS) {
    const control = runOneCondition(seed, "control", observationSize);
    const intervention = runOneCondition(seed, "intervention", observationSize);

    const parityCheck = assertPreFreezeParity(control.records, intervention.records);
    if (!parityCheck.identical) {
      console.warn(`seed ${seed}: pre-freeze parity check FAILED —`, JSON.stringify(parityCheck));
    }

    const divergence = postFreezeActionDivergenceCount(control.records, intervention.records, FREEZE_STEP);
    divergences.push({ seed, ...divergence });

    rows.push(
      writeRun(seed, "control", control.records, control.series, parityCheck, divergence),
      writeRun(seed, "intervention", intervention.records, intervention.series, parityCheck, divergence),
    );

    const diff = driftAttributableError(intervention.series, control.series);
    diffs.push({ seed, diffMean: mean(diff), diffSlope: slope(diff) });

    console.log(
      `seed ${seed}: diffMean=${mean(diff).toFixed(4)} | prefreezeParity=${parityCheck.identical} | ` +
        `postFreezeActionDivergence=${divergence.divergentSteps}/${divergence.postFreezeSteps}`,
    );
  }

  const csvHeader = "seed,condition,postFreezeSteps,meanLoss,slopeLoss\n";
  const csvBody = rows
    .map((r) => `${r.seed},${r.condition},${r.postFreezeSteps},${r.meanLoss},${r.slopeLoss}`)
    .join("\n");
  writeFileSync(`${artifactsDir}results.summary.csv`, csvHeader + csvBody + "\n");

  const divergenceCsvHeader = "seed,postFreezeSteps,divergentSteps\n";
  const divergenceCsvBody = divergences
    .map((d) => `${d.seed},${d.postFreezeSteps},${d.divergentSteps}`)
    .join("\n");
  writeFileSync(`${artifactsDir}divergence.summary.csv`, divergenceCsvHeader + divergenceCsvBody + "\n");

  console.log("\n--- Post-freeze action-divergence summary (n=3, descriptive) ---");
  for (const d of divergences) {
    console.log(`seed ${d.seed}: ${d.divergentSteps}/${d.postFreezeSteps} post-freeze steps had differing actions`);
  }
  const diffMeans = diffs.map((d) => d.diffMean);
  console.log(`\nper-seed diffMean for reference: ${diffMeans.map((d) => d.toFixed(4)).join(", ")}`);
}

main();
