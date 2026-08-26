/**
 * Processes PR #51's review (@SakkarinKt, 2026-08-24 merge comment) "Next" line: distinct seeds,
 * not a priority pivot. 2026-08-23's and 2026-08-24's runs left an open question (2026-08-23's
 * "Decisions needed", restated 2026-08-24) — whether the large, sign-inconsistent per-seed
 * `diffMean` swings across the viewRadius sweep (seed 1003 in particular: 0.0000 -> +0.4271 ->
 * -0.5103) reflect a real mechanism or n=3 sampling noise, given the harness is fully
 * deterministic (QLearningPolicy.act draws from the seeded Rng runEpisode threads through, not
 * Math.random — no within-seed stochasticity to average over, so the only way to add independent
 * samples is more seeds). The review picked radius 6 specifically: "the axis with the largest
 * swings" (seed 1003's radius-6 diffMean, -0.5103, is the largest-magnitude value seen across any
 * run to date).
 *
 *   node experiments/2026-08-25-radius6-seed-spread/run.ts
 *
 * Runs three new seeds (1004, 1005, 1006) at partner-only viewRadius=6 only — same decoupled
 * design as 2026-08-24's run (landmark gate config.viewRadius pinned at 2 for the whole episode;
 * only the partner's post-freeze gate is set via setPartnerViewRadius(6)) — then pools their
 * diffMean with the three already-committed radius-6 rows from
 * artifacts/2026-08-24-partner-only-viewradius-switch/sweep.summary.csv (seeds 1001-1003) for an
 * n=6 spread on the one axis this question is about. Not a rerun of 1001-1003: those are read
 * verbatim from the prior run's committed CSV, not recomputed, since the harness is deterministic
 * and re-executing them would reproduce the identical numbers at the cost of a second manifest set
 * for no new information.
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

const RUN_ID = "2026-08-25-radius6-seed-spread";
const NEW_SEEDS = [1004, 1005, 1006];
const PARTNER_RADIUS = 6;
const LANDMARK_RADIUS = 2; // Matches 2026-08-24's BASELINE_RADIUS — landmark gate pinned whole episode.
const FREEZE_STEP = 38;
const FROZEN_AGENT_INDEX = 0;
const HORIZON = 75; // Matches every prior Arm-A instrument-validation run.

const RSSM_CONFIG = { deterministicSize: 256, latentCategoricals: 8, latentClasses: 4 };
const Q_LEARNING_CONFIG: Required<QLearningConfig> = { alpha: 0.1, gamma: 0.95, epsilon: 0.1 };

const artifactsDir = fileURLToPath(new URL(`../../artifacts/${RUN_ID}/`, import.meta.url));
const priorSweepCsvPath = fileURLToPath(
  new URL("../../artifacts/2026-08-24-partner-only-viewradius-switch/sweep.summary.csv", import.meta.url),
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

function stddev(values: number[]): number {
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
        summary: "Landmark-visible-step tally (union over control/intervention) — landmark gate pinned, so this should be constant with 2026-08-24's same-seed value at other radii if this run's fix (reset() clearing partnerViewRadiusOverride, PR #51 review) changed nothing observable.",
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

interface PriorRow {
  viewRadius: number;
  seed: number;
  diffMean: number;
}

function readPriorRadius6Rows(): PriorRow[] {
  const csv = readFileSync(priorSweepCsvPath, "utf8").trim().split("\n");
  const header = csv[0]!.split(",");
  const viewRadiusIdx = header.indexOf("viewRadius");
  const seedIdx = header.indexOf("seed");
  const diffMeanIdx = header.indexOf("diffMean");
  return csv
    .slice(1)
    .map((line) => line.split(","))
    .filter((cols) => Number(cols[viewRadiusIdx]) === 6)
    .map((cols) => ({
      viewRadius: 6,
      seed: Number(cols[seedIdx]),
      diffMean: Number(cols[diffMeanIdx]),
    }));
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const probeEnv = new CooperativeGridWorld({ seed: 0, horizon: HORIZON });
  const observationSize = probeEnv.observationLength;
  const numLandmarks = probeEnv.config.numLandmarks;

  const newRows: { seed: number; diffMean: number; diffSlope: number }[] = [];

  for (const seed of NEW_SEEDS) {
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
    const diffMean = mean(diff);
    const diffSlope = slope(diff);
    newRows.push({ seed, diffMean, diffSlope });

    console.log(
      `seed ${seed}: diffMean=${diffMean.toFixed(4)} | prefreezeParity=${parityCheck.identical} | ` +
        `partnerVisible=${partnerVisibility.visibleSteps}/${partnerVisibility.postFreezeSteps} | ` +
        `landmarkVisible=${landmarkVisibility.visibleSteps}/${landmarkVisibility.postFreezeSteps}`,
    );
  }

  const priorRows = readPriorRadius6Rows();
  const pooled = [...priorRows.map((r) => ({ seed: r.seed, diffMean: r.diffMean })), ...newRows.map((r) => ({ seed: r.seed, diffMean: r.diffMean }))];

  const csvHeader = "seed,diffMean,diffSlope,source\n";
  const csvBody = [
    ...priorRows.map((r) => `${r.seed},${r.diffMean},,2026-08-24-partner-only-viewradius-switch`),
    ...newRows.map((r) => `${r.seed},${r.diffMean},${r.diffSlope},${RUN_ID}`),
  ].join("\n");
  writeFileSync(`${artifactsDir}pooled-radius6.summary.csv`, csvHeader + csvBody + "\n");

  console.log("\n--- radius=6 partner-only viewRadius, n=6 pooled seed spread (descriptive) ---");
  console.log(`seeds: ${pooled.map((r) => r.seed).join(", ")}`);
  console.log(`diffMean values: ${pooled.map((r) => r.diffMean.toFixed(4)).join(", ")}`);
  console.log(`mean(diffMean) = ${mean(pooled.map((r) => r.diffMean)).toFixed(4)}`);
  console.log(`mean(|diffMean|) = ${mean(pooled.map((r) => Math.abs(r.diffMean))).toFixed(4)}`);
  console.log(`stddev(diffMean) = ${stddev(pooled.map((r) => r.diffMean)).toFixed(4)}`);
  console.log(`sign flips present: ${pooled.some((r) => r.diffMean > 0) && pooled.some((r) => r.diffMean < 0)}`);
}

main();
