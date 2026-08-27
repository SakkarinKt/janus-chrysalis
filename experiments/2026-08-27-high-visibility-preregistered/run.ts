/**
 * Processes PR #55's review (@SakkarinKt, posted as an issue comment after merge). The review
 * answered the "Decisions needed" item PR #55 raised (a power-aware pass vs. a from-scratch
 * high-visibility-by-design run): option (a), the high-visibility-by-design run — but
 * pre-registered, stating the seed count and replication criterion in the PR body *before*
 * running, so this investigation's recurring failure mode ("one more look after each
 * inconclusive result") can't repeat itself here.
 *
 * Design, fixed before any of this run's `diffMean` values were computed (see
 * docs/proposals/0001-direct-nonstationarity-measurement.md's "2026-08-27 update (third cycle)"
 * for the full pre-registration text, committed in the same PR before this script's output was
 * generated):
 *
 * - Fresh seeds, unused anywhere earlier in this investigation: 1010, 1011, 1012.
 * - Identical harness to every run since 2026-08-24: decoupled partner-only viewRadius switch,
 *   landmark gate (`config.viewRadius`) pinned at 2 for the whole episode, paired init,
 *   FREEZE_STEP=38, frozenAgentIndex=0, HORIZON=75, Arm-A dims, default QLearningConfig.
 * - `partnerViewRadius = 14` = 2*(gridSize-1), the maximum possible Manhattan distance between any
 *   two cells on this 8x8 grid (`src/env/gridworld.ts`'s `distance()` is `|dx|+|dy|`). This
 *   guarantees the partner is visible to the frozen agent on *every* post-freeze step for *every*
 *   seed by construction, not by empirical tendency the way radius 4/6/8 were in earlier sweeps —
 *   there is no post-hoc conditioning step because no seed can land in a low-power cell. Asserted
 *   below as an invariant: any seed whose `postFreezePartnerVisibleCount` isn't exactly 38/38
 *   indicates a bug in this run, not a data point.
 *
 *   node experiments/2026-08-27-high-visibility-preregistered/run.ts
 *
 * Combined-test data source: artifacts/2026-08-26-radius6-more-seeds/pooled-radius6.summary.csv
 * (the existing 9 radius-6 seeds' diffMean signs, all already 27-38/38 partner-visible — read
 * verbatim, not recomputed, per this investigation's established convention for prior-run data).
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

const RUN_ID = "2026-08-27-high-visibility-preregistered";
const SEEDS = [1010, 1011, 1012]; // Fresh — unused anywhere earlier in this investigation.
const GRID_SIZE = 8; // Matches src/env/types.ts DEFAULT_CONFIG.gridSize.
const PARTNER_RADIUS = 2 * (GRID_SIZE - 1); // = 14: max possible Manhattan distance on this grid.
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

/** One-sided binomial P(X >= k) for X ~ Binomial(n, 0.5). */
function binomialUpperTail(k: number, n: number): number {
  let total = 0;
  let coefficient = 1;
  for (let i = 0; i <= n; i++) {
    if (i > 0) coefficient = (coefficient * (n - i + 1)) / i;
    if (i >= k) total += coefficient;
  }
  return total / 2 ** n;
}

/** Two-sided exact binomial p-value for X ~ Binomial(n, 0.5): 2 * min(P(X>=k), P(X<=k)), capped at 1. */
function binomialTwoSided(k: number, n: number): number {
  const upper = binomialUpperTail(k, n);
  const lower = k === 0 ? 1 : binomialUpperTail(n - k, n); // P(X<=k) = P(X>=n-k) by symmetry at p=0.5
  return Math.min(1, 2 * Math.min(upper, lower));
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
        summary: "Partner-visible-step tally (union over control/intervention) — must be 38/38 by construction at partnerRadius=14.",
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

function readRadius6Signs(): { negative: number; positive: number; n: number } {
  const csv = readFileSync(priorRadius6CsvPath, "utf8").trim().split("\n");
  const header = csv[0]!.split(",");
  const diffMeanIdx = header.indexOf("diffMean");
  const values = csv.slice(1).map((line) => Number(line.split(",")[diffMeanIdx]));
  return {
    negative: values.filter((v) => v < 0).length,
    positive: values.filter((v) => v > 0).length,
    n: values.length,
  };
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const probeEnv = new CooperativeGridWorld({ seed: 0, horizon: HORIZON });
  const observationSize = probeEnv.observationLength;
  const numLandmarks = probeEnv.config.numLandmarks;

  const rows: { seed: number; diffMean: number; partnerVisibleSteps: number; postFreezeSteps: number }[] = [];
  const invariantViolations: string[] = [];

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

    if (partnerVisibility.visibleSteps !== partnerVisibility.postFreezeSteps) {
      invariantViolations.push(
        `seed ${seed}: partnerVisibleSteps ${partnerVisibility.visibleSteps}/${partnerVisibility.postFreezeSteps} — expected full visibility (partnerRadius=${PARTNER_RADIUS} guarantees it by construction)`,
      );
    }

    writeRun(seed, "control", control.records, control.series, parityCheck, actionDivergence, observationDivergence, partnerVisibility, landmarkVisibility);
    writeRun(seed, "intervention", intervention.records, intervention.series, parityCheck, actionDivergence, observationDivergence, partnerVisibility, landmarkVisibility);

    const diff = driftAttributableError(intervention.series, control.series);
    const diffMean = mean(diff);
    rows.push({
      seed,
      diffMean,
      partnerVisibleSteps: partnerVisibility.visibleSteps,
      postFreezeSteps: partnerVisibility.postFreezeSteps,
    });

    console.log(
      `seed ${seed}: diffMean=${diffMean.toFixed(4)} | partnerVisible=${partnerVisibility.visibleSteps}/${partnerVisibility.postFreezeSteps} | prefreezeParity=${parityCheck.identical}`,
    );
  }

  if (invariantViolations.length > 0) {
    console.error("\n--- INVARIANT VIOLATION: partnerRadius=14 did not guarantee full visibility ---");
    for (const v of invariantViolations) console.error(v);
    console.error("This indicates a bug in this run's setup, not a valid data point. Do not interpret the results below.");
  }

  const csvHeader = "seed,diffMean,partnerVisibleSteps,postFreezeSteps\n";
  const csvBody = rows.map((r) => `${r.seed},${r.diffMean},${r.partnerVisibleSteps},${r.postFreezeSteps}`).join("\n");
  writeFileSync(`${artifactsDir}high-visibility-preregistered.summary.csv`, csvHeader + csvBody + "\n");

  const values = rows.map((r) => r.diffMean);
  const newNeg = values.filter((v) => v < 0).length;
  const newPos = values.filter((v) => v > 0).length;
  const priorRadius6 = readRadius6Signs();
  const combinedNeg = newNeg + priorRadius6.negative;
  const combinedN = rows.length + priorRadius6.n;
  const combinedP = binomialTwoSided(combinedNeg, combinedN);

  console.log(`\n--- new seeds (n=${rows.length}), full-visibility-by-design ---`);
  console.log(`mean(diffMean) = ${mean(values).toFixed(4)}, mean(|diffMean|) = ${mean(values.map(Math.abs)).toFixed(4)}, stddev = ${stddevPopulation(values).toFixed(4)}`);
  console.log(`sign split: ${newNeg} negative, ${newPos} positive`);
  console.log("seed | diffMean | partnerVisible/postFreeze");
  for (const r of rows) {
    console.log(`${r.seed} | ${r.diffMean.toFixed(4)} | ${r.partnerVisibleSteps}/${r.postFreezeSteps}`);
  }

  console.log(`\n--- combined test: new n=${rows.length} + prior radius-6 n=${priorRadius6.n} = n=${combinedN} ---`);
  console.log(`prior radius-6 sign split: ${priorRadius6.negative} negative, ${priorRadius6.positive} positive`);
  console.log(`combined negative count: ${combinedNeg}/${combinedN}`);
  console.log(`two-sided exact binomial p (H0: p=0.5): ${combinedP.toFixed(6)}`);

  const preRegisteredCriterionMet = newNeg >= 2 && combinedP < 0.05 && combinedNeg > combinedN / 2;
  console.log(`\npre-registered replication criterion met: ${preRegisteredCriterionMet}`);
}

main();
