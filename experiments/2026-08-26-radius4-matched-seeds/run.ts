/**
 * Processes PR #53's review (@SakkarinKt, posted as an issue comment after merge). The review
 * caught two errors in that PR's "What this does and doesn't settle" paragraph, both corrected
 * directly in `docs/proposals/0001-direct-nonstationarity-measurement.md` (not repeated here):
 *
 * 1. `driftAttributableError` is `intervention - control` — 2026-08-13's "gate (b)" convention
 *    reads **positive** `diffMean` as the predicted-direction result. The n=9 radius-6 pool's
 *    8-negative/1-positive split is therefore evidence *against* the predicted drift direction,
 *    not "the strongest evidence to date for some real effect" (what the paragraph originally,
 *    incorrectly, claimed).
 * 2. All nine radius-6 pooled rows are at the *same* radius. That result carries zero radius
 *    contrast — it says something about the freeze-vs-control contrast at radius 6 specifically,
 *    not about partner-`viewRadius` as a variable, which the original paragraph's framing
 *    implied without having the data to support it.
 *
 * The review's "Next": correct the text (done), then run "a matched seed set at radius 4 — the
 * only one of the three [decisions-needed options] that builds the radius contrast the sentence
 * currently assumes." "Matched" here means the *same nine seeds* already run at radius 6
 * (1001-1009), so each seed now has a radius-4 and a radius-6 `diffMean` and the comparison is a
 * true paired design, not a fresh independent sample.
 *
 *   node experiments/2026-08-26-radius4-matched-seeds/run.ts
 *
 * Same decoupled partner-only design as every run since 2026-08-24: landmark gate
 * (`config.viewRadius`) pinned at 2 for the whole episode; only the partner's post-freeze gate
 * moves, via `setPartnerViewRadius(4)` this time instead of 6.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
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
import {
  driftAttributableError,
  postFreezeActionDivergenceCount,
  postFreezeObservationDivergenceCount,
  postFreezePartnerVisibleCount,
  postFreezeLandmarkVisibleCount,
  postFreezeLossSeries,
} from "../../src/experiment/metrics.ts";

const RUN_ID = "2026-08-26-radius4-matched-seeds";
const SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]; // Matches the radius-6 n=9 pool exactly.
const PARTNER_RADIUS = 4;
const LANDMARK_RADIUS = 2; // Matches every prior run since 2026-08-24 — landmark gate pinned whole episode.
const FREEZE_STEP = 38;
const FROZEN_AGENT_INDEX = 0;
const HORIZON = 75; // Matches every prior Arm-A instrument-validation run.

const RSSM_CONFIG = { deterministicSize: 256, latentCategoricals: 8, latentClasses: 4 };
const Q_LEARNING_CONFIG: Required<QLearningConfig> = { alpha: 0.1, gamma: 0.95, epsilon: 0.1 };

const artifactsDir = fileURLToPath(new URL(`../../artifacts/${RUN_ID}/`, import.meta.url));
const priorRadius6CsvPath = fileURLToPath(
  new URL("../../artifacts/2026-08-26-radius6-more-seeds/pooled-radius6.summary.csv", import.meta.url),
);
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

function stddevPopulation(values: number[]): number {
  const m = mean(values);
  return Math.sqrt(mean(values.map((v) => (v - m) ** 2)));
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

/** Same pairing as every prior run's `buildWorldModels` — one seed stream per agent index. */
function buildWorldModels(seed: number, observationSize: number): [WorldModel, WorldModel] {
  const configFor = (agentIndex: number): WorldModelConfig => ({
    rssm: RSSM_CONFIG,
    observationSize,
    seed: deriveSeed(seed, agentIndex),
  });
  return [new WorldModel(configFor(0)), new WorldModel(configFor(1))];
}

function preFreezeParity(a: EpisodeStepRecord[], b: EpisodeStepRecord[]): Record<string, unknown> {
  const preFreeze = (records: EpisodeStepRecord[]) => records.filter((r) => r.step < FREEZE_STEP);
  const ar = preFreeze(a);
  const br = preFreeze(b);

  const mismatches: string[] = [];
  if (ar.length !== br.length) mismatches.push(`pre-freeze step count differs: a=${ar.length} b=${br.length}`);
  for (let idx = 0; idx < Math.min(ar.length, br.length); idx++) {
    const ra = ar[idx]!;
    const rb = br[idx]!;
    if (JSON.stringify(ra.actions) !== JSON.stringify(rb.actions)) mismatches.push(`step ${ra.step}: actions differ`);
    if (ra.reward !== rb.reward) mismatches.push(`step ${ra.step}: reward differs`);
    if (JSON.stringify(ra.observations) !== JSON.stringify(rb.observations))
      mismatches.push(`step ${ra.step}: observations differ`);
    if (JSON.stringify(ra.nextObservations) !== JSON.stringify(rb.nextObservations))
      mismatches.push(`step ${ra.step}: nextObservations differ`);
    if (JSON.stringify(ra.worldModelLoss) !== JSON.stringify(rb.worldModelLoss))
      mismatches.push(`step ${ra.step}: worldModelLoss differs`);
  }

  return {
    preFreezeStepCount: ar.length,
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
  actionDivergence: { postFreezeSteps: number; divergentSteps: number },
  observationDivergence: { postFreezeSteps: number; divergentSteps: number },
  partnerVisibility: { postFreezeSteps: number; visibleSteps: number },
  landmarkVisibility: { postFreezeSteps: number; visibleSteps: number },
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
    partnerRadius: PARTNER_RADIUS,
    landmarkRadius: LANDMARK_RADIUS,
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
        summary: "Pre-freeze parity between control and intervention for this seed.",
        preFreezeParityCheck: parityCheck,
      },
      {
        severity: "NOTE",
        confidence: "self_checked, high confidence",
        summary: "Post-freeze action-divergence count.",
        postFreezeActionDivergenceCount: actionDivergence,
      },
      {
        severity: "NOTE",
        confidence: "self_checked, high confidence",
        summary: `Post-freeze observation-divergence count for the frozen agent (index ${FROZEN_AGENT_INDEX}).`,
        postFreezeObservationDivergenceCount: observationDivergence,
      },
      {
        severity: "NOTE",
        confidence: "self_checked, high confidence",
        summary: "Partner-visible-step tally (union over control/intervention).",
        postFreezePartnerVisibleCount: partnerVisibility,
      },
      {
        severity: "NOTE",
        confidence: "self_checked, high confidence",
        summary: "Landmark-visible-step tally (union over control/intervention) — landmark gate pinned, expected constant.",
        postFreezeLandmarkVisibleCount: landmarkVisibility,
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
  const env = new CooperativeGridWorld({ seed, horizon: HORIZON, viewRadius: LANDMARK_RADIUS });
  const policies = [new QLearningPolicy(Q_LEARNING_CONFIG), new QLearningPolicy(Q_LEARNING_CONFIG)];
  const [wm0, wm1] = buildWorldModels(seed, observationSize);

  const freezeConfig: FreezeConfig =
    condition === "control"
      ? { freezeStep: FREEZE_STEP, condition: "control" }
      : { freezeStep: FREEZE_STEP, condition: "intervention", frozenAgentIndex: FROZEN_AGENT_INDEX };

  const postFreezeEnvMutation = (mutatedEnv: CooperativeGridWorld) => mutatedEnv.setPartnerViewRadius(PARTNER_RADIUS);

  const records = runEpisode(env, policies, seed, freezeConfig, [wm0, wm1], postFreezeEnvMutation);
  const series = postFreezeLossSeries(records, FREEZE_STEP, FROZEN_AGENT_INDEX);

  wm0.dispose();
  wm1.dispose();

  return {
    outcome: { seed, condition, postFreezeSteps: series.length, meanLoss: mean(series), slopeLoss: slope(series) },
    series,
    records,
  };
}

function readRadius6DiffMeans(): Map<number, number> {
  const csv = readFileSync(priorRadius6CsvPath, "utf8").trim().split("\n");
  const header = csv[0]!.split(",");
  const seedIdx = header.indexOf("seed");
  const diffMeanIdx = header.indexOf("diffMean");
  const map = new Map<number, number>();
  for (const line of csv.slice(1)) {
    const cols = line.split(",");
    map.set(Number(cols[seedIdx]), Number(cols[diffMeanIdx]));
  }
  return map;
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const probeEnv = new CooperativeGridWorld({ seed: 0, horizon: HORIZON });
  const observationSize = probeEnv.observationLength;
  const numLandmarks = probeEnv.config.numLandmarks;

  const radius6DiffMeans = readRadius6DiffMeans();
  const rows: { seed: number; diffMeanR4: number; diffMeanR6: number; pairedDiff: number }[] = [];

  for (const seed of SEEDS) {
    const control = runOneCondition(seed, "control", observationSize);
    const intervention = runOneCondition(seed, "intervention", observationSize);

    const parityCheck = preFreezeParity(control.records, intervention.records);
    if (!parityCheck.identical) {
      console.warn(`seed ${seed}: pre-freeze parity check FAILED —`, JSON.stringify(parityCheck));
    }

    const actionDivergence = postFreezeActionDivergenceCount(control.records, intervention.records, FREEZE_STEP);
    const observationDivergence = postFreezeObservationDivergenceCount(
      control.records,
      intervention.records,
      FREEZE_STEP,
      FROZEN_AGENT_INDEX,
    );
    const partnerVisibility = postFreezePartnerVisibleCount(control.records, intervention.records, FREEZE_STEP, FROZEN_AGENT_INDEX);
    const landmarkVisibility = postFreezeLandmarkVisibleCount(
      control.records,
      intervention.records,
      FREEZE_STEP,
      FROZEN_AGENT_INDEX,
      numLandmarks,
    );

    writeRun(seed, "control", control.records, control.series, parityCheck, actionDivergence, observationDivergence, partnerVisibility, landmarkVisibility);
    writeRun(seed, "intervention", intervention.records, intervention.series, parityCheck, actionDivergence, observationDivergence, partnerVisibility, landmarkVisibility);

    const diff = driftAttributableError(intervention.series, control.series);
    const diffMeanR4 = mean(diff);
    const diffMeanR6 = radius6DiffMeans.get(seed)!;
    const pairedDiff = diffMeanR4 - diffMeanR6;
    rows.push({ seed, diffMeanR4, diffMeanR6, pairedDiff });

    console.log(
      `seed ${seed}: diffMean(r4)=${diffMeanR4.toFixed(4)} | diffMean(r6)=${diffMeanR6.toFixed(4)} | ` +
        `pairedDiff(r4-r6)=${pairedDiff.toFixed(4)} | prefreezeParity=${parityCheck.identical}`,
    );
  }

  const csvHeader = "seed,diffMeanR4,diffMeanR6,pairedDiffR4minusR6\n";
  const csvBody = rows.map((r) => `${r.seed},${r.diffMeanR4},${r.diffMeanR6},${r.pairedDiff}`).join("\n");
  writeFileSync(`${artifactsDir}radius4-vs-radius6.summary.csv`, csvHeader + csvBody + "\n");

  const r4Values = rows.map((r) => r.diffMeanR4);
  const r6Values = rows.map((r) => r.diffMeanR6);
  const pairedValues = rows.map((r) => r.pairedDiff);

  console.log(`\n--- radius 4 vs radius 6, n=${rows.length} matched seeds (descriptive) ---`);
  console.log(`radius 4: mean(diffMean) = ${mean(r4Values).toFixed(4)}, mean(|diffMean|) = ${mean(r4Values.map(Math.abs)).toFixed(4)}, stddev = ${stddevPopulation(r4Values).toFixed(4)}`);
  const r4Neg = r4Values.filter((v) => v < 0).length;
  const r4Pos = r4Values.filter((v) => v > 0).length;
  console.log(`radius 4 sign split: ${r4Neg} negative, ${r4Pos} positive`);
  console.log(`radius 6 (for reference): mean(diffMean) = ${mean(r6Values).toFixed(4)}, sign split: ${r6Values.filter((v) => v < 0).length} negative, ${r6Values.filter((v) => v > 0).length} positive`);
  console.log(`paired diff (r4 - r6): mean = ${mean(pairedValues).toFixed(4)}, stddev = ${stddevPopulation(pairedValues).toFixed(4)}`);
  const sameSignCount = rows.filter((r) => Math.sign(r.diffMeanR4) === Math.sign(r.diffMeanR6)).length;
  console.log(`same-sign-at-both-radii count: ${sameSignCount}/${rows.length}`);
  console.log("seed | diffMean(r4) | diffMean(r6) | same sign?");
  for (const r of rows) {
    console.log(`${r.seed} | ${r.diffMeanR4.toFixed(4)} | ${r.diffMeanR6.toFixed(4)} | ${Math.sign(r.diffMeanR4) === Math.sign(r.diffMeanR6)}`);
  }
}

main();
