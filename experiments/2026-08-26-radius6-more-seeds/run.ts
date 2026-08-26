/**
 * Processes PR #52's review (@SakkarinKt, 2026-08-25 merge comment... actually posted after merge,
 * as an issue comment on the closed PR). The review found the 2026-08-25 update's "n=6 favors
 * noise" interpretation misread its own numbers in four places (sign split actually 5-negative/
 * 1-positive, not 3/3; stddev is ~1.5x mean(|diffMean|) not "nearly 3x"; seed 1006 alone is 69.8%
 * of the total variance, not "no seed dominating"; the near-zero pooled mean is an artifact of
 * that one outlier cancelling five same-signed values). Those corrections are folded into
 * `docs/proposals/0001-direct-nonstationarity-measurement.md`'s "2026-08-25 update" directly
 * (dated 2026-08-26), not repeated here.
 *
 * This run is the review's answer to the "Decisions needed" item: "more seeds at radius 6, same
 * design — not the geometry pivot, not a priority move," because the corrected sign pattern now
 * carries a prediction the geometry-varying alternative doesn't test — if the five-negative run is
 * noise, more seeds should keep splitting the sign; if it's real, they should keep coming back
 * negative with 1006 as the outlier.
 *
 *   node experiments/2026-08-26-radius6-more-seeds/run.ts
 *
 * Three new seeds (1007, 1008, 1009), same decoupled partner-only viewRadius=6 design as
 * 2026-08-24/25 (landmark gate pinned at 2 for the whole episode). Pools with all six prior
 * radius-6 rows (1001-1003 from 2026-08-24, 1004-1006 from 2026-08-25) for n=9. Fixes the review's
 * nit that 2026-08-25's `readPriorRadius6Rows` didn't parse the `diffSlope` column (present in the
 * source CSV, just not read); this run's own pooling parses it correctly for every prior row.
 *
 * Also adds the review's requested column: partner-visible-step saturation
 * (`partnerVisibleSteps`/`partnerVisiblePostFreezeSteps`). The review's "concrete lead" — seed 1006
 * is the only new seed whose partner-visible tally is off ceiling (28/38, vs 1004's and 1005's
 * 38/38) — is the same "already at ceiling" argument the proposal doc uses for seed 1001's
 * identical radius-4/6 value; if large-|diffMean| seeds turn out to be exactly the unsaturated
 * ones, that would point at a mechanism (visibility headroom) rather than pure noise. This run
 * reports that column for every pooled seed so the pattern (or its absence) is visible without
 * cross-referencing nine separate manifests.
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

const RUN_ID = "2026-08-26-radius6-more-seeds";
const NEW_SEEDS = [1007, 1008, 1009];
const PARTNER_RADIUS = 6;
const LANDMARK_RADIUS = 2; // Matches 2026-08-24's/25's BASELINE_RADIUS — landmark gate pinned whole episode.
const FREEZE_STEP = 38;
const FROZEN_AGENT_INDEX = 0;
const HORIZON = 75; // Matches every prior Arm-A instrument-validation run.

const RSSM_CONFIG = { deterministicSize: 256, latentCategoricals: 8, latentClasses: 4 };
const Q_LEARNING_CONFIG: Required<QLearningConfig> = { alpha: 0.1, gamma: 0.95, epsilon: 0.1 };

const artifactsDir = fileURLToPath(new URL(`../../artifacts/${RUN_ID}/`, import.meta.url));
const priorSweep0824CsvPath = fileURLToPath(
  new URL("../../artifacts/2026-08-24-partner-only-viewradius-switch/sweep.summary.csv", import.meta.url),
);
const prior0825Dir = fileURLToPath(new URL("../../artifacts/2026-08-25-radius6-seed-spread/", import.meta.url));
const gitCommit = execSync("git rev-parse HEAD").toString().trim();

interface RunOutcome {
  seed: number;
  condition: FreezeCondition;
  postFreezeSteps: number;
  meanLoss: number;
  slopeLoss: number;
}

interface PooledRow {
  seed: number;
  diffMean: number;
  diffSlope: number;
  partnerVisibleSteps: number;
  partnerVisiblePostFreezeSteps: number;
  source: string;
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function stddevPopulation(values: number[]): number {
  const m = mean(values);
  return Math.sqrt(mean(values.map((v) => (v - m) ** 2)));
}

function stddevSample(values: number[]): number {
  const m = mean(values);
  const sumSquares = values.reduce((acc, v) => acc + (v - m) ** 2, 0);
  return Math.sqrt(sumSquares / (values.length - 1));
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

/** Reads seeds 1001-1003's radius-6 rows from 2026-08-24's committed sweep CSV — that CSV has
 * every column this run reports, unlike 2026-08-25's pooled CSV (which dropped diffSlope for these
 * three rows, the review's nit; and never had partnerVisibleSteps at all). */
function readPrior0824Rows(): PooledRow[] {
  const csv = readFileSync(priorSweep0824CsvPath, "utf8").trim().split("\n");
  const header = csv[0]!.split(",");
  const col = (name: string) => header.indexOf(name);
  const viewRadiusIdx = col("viewRadius");
  const seedIdx = col("seed");
  const diffMeanIdx = col("diffMean");
  const diffSlopeIdx = col("diffSlope");
  const partnerVisibleIdx = col("partnerVisibleSteps");
  const partnerPostFreezeIdx = col("partnerVisiblePostFreezeSteps");
  return csv
    .slice(1)
    .map((line) => line.split(","))
    .filter((cols) => Number(cols[viewRadiusIdx]) === 6)
    .map((cols) => ({
      seed: Number(cols[seedIdx]),
      diffMean: Number(cols[diffMeanIdx]),
      diffSlope: Number(cols[diffSlopeIdx]),
      partnerVisibleSteps: Number(cols[partnerVisibleIdx]),
      partnerVisiblePostFreezeSteps: Number(cols[partnerPostFreezeIdx]),
      source: "2026-08-24-partner-only-viewradius-switch",
    }));
}

/** Reads seeds 1004-1006's rows from 2026-08-25's own per-seed manifests (not its pooled CSV,
 * which lacks partnerVisibleSteps) — diffMean/diffSlope from the pooled CSV (correct for these
 * three; the parsing bug only affected the 1001-1003 rows it re-read from 2026-08-24), partner
 * visibility from each seed's intervention manifest.json. */
function readPrior0825Rows(): PooledRow[] {
  const pooledCsv = readFileSync(`${prior0825Dir}pooled-radius6.summary.csv`, "utf8").trim().split("\n");
  const header = pooledCsv[0]!.split(",");
  const col = (name: string) => header.indexOf(name);
  const seedIdx = col("seed");
  const diffMeanIdx = col("diffMean");
  const diffSlopeIdx = col("diffSlope");
  const sourceIdx = col("source");

  return pooledCsv
    .slice(1)
    .map((line) => line.split(","))
    .filter((cols) => cols[sourceIdx] === "2026-08-25-radius6-seed-spread")
    .map((cols) => {
      const seed = Number(cols[seedIdx]);
      const manifestPath = `${prior0825Dir}seed-${seed}-intervention/manifest.json`;
      const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
      const partnerFinding = manifest.findings.find((f: { postFreezePartnerVisibleCount?: unknown }) => f.postFreezePartnerVisibleCount)!
        .postFreezePartnerVisibleCount as { visibleSteps: number; postFreezeSteps: number };
      return {
        seed,
        diffMean: Number(cols[diffMeanIdx]),
        diffSlope: Number(cols[diffSlopeIdx]),
        partnerVisibleSteps: partnerFinding.visibleSteps,
        partnerVisiblePostFreezeSteps: partnerFinding.postFreezeSteps,
        source: "2026-08-25-radius6-seed-spread",
      };
    });
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const probeEnv = new CooperativeGridWorld({ seed: 0, horizon: HORIZON });
  const observationSize = probeEnv.observationLength;
  const numLandmarks = probeEnv.config.numLandmarks;

  const newRows: PooledRow[] = [];

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
    newRows.push({
      seed,
      diffMean,
      diffSlope,
      partnerVisibleSteps: partnerVisibility.visibleSteps,
      partnerVisiblePostFreezeSteps: partnerVisibility.postFreezeSteps,
      source: RUN_ID,
    });

    console.log(
      `seed ${seed}: diffMean=${diffMean.toFixed(4)} | prefreezeParity=${parityCheck.identical} | ` +
        `partnerVisible=${partnerVisibility.visibleSteps}/${partnerVisibility.postFreezeSteps}`,
    );
  }

  const pooled: PooledRow[] = [...readPrior0824Rows(), ...readPrior0825Rows(), ...newRows];

  const csvHeader = "seed,diffMean,diffSlope,partnerVisibleSteps,partnerVisiblePostFreezeSteps,source\n";
  const csvBody = pooled
    .map((r) => `${r.seed},${r.diffMean},${r.diffSlope},${r.partnerVisibleSteps},${r.partnerVisiblePostFreezeSteps},${r.source}`)
    .join("\n");
  writeFileSync(`${artifactsDir}pooled-radius6.summary.csv`, csvHeader + csvBody + "\n");

  const diffMeans = pooled.map((r) => r.diffMean);
  console.log(`\n--- radius=6 partner-only viewRadius, n=${pooled.length} pooled seed spread (descriptive) ---`);
  console.log(`seeds: ${pooled.map((r) => r.seed).join(", ")}`);
  console.log(`diffMean values: ${diffMeans.map((v) => v.toFixed(4)).join(", ")}`);
  console.log(`mean(diffMean) = ${mean(diffMeans).toFixed(4)}`);
  console.log(`mean(|diffMean|) = ${mean(diffMeans.map(Math.abs)).toFixed(4)}`);
  console.log(`stddev population(diffMean) = ${stddevPopulation(diffMeans).toFixed(4)}`);
  console.log(`stddev sample(diffMean) = ${stddevSample(diffMeans).toFixed(4)}`);
  const negCount = diffMeans.filter((v) => v < 0).length;
  const posCount = diffMeans.filter((v) => v > 0).length;
  console.log(`sign split: ${negCount} negative, ${posCount} positive`);
  console.log("seed | diffMean | partnerVisible/postFreezeSteps | saturated (== postFreezeSteps)?");
  for (const r of pooled) {
    const saturated = r.partnerVisibleSteps === r.partnerVisiblePostFreezeSteps;
    console.log(
      `${r.seed} | ${r.diffMean.toFixed(4)} | ${r.partnerVisibleSteps}/${r.partnerVisiblePostFreezeSteps} | ${saturated}`,
    );
  }
}

main();
