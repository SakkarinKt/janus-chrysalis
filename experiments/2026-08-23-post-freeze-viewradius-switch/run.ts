/**
 * Processes PR #49's review (@SakkarinKt, 2026-08-22 merge comment) "Next" line: "(a) the
 * post-freeze-only `viewRadius` switch, with the visibility tally landed alongside as its own
 * column and radii chosen below the 38/38 ceiling, since radii ≥ 8 stop moving the axis."
 *
 *   node experiments/2026-08-23-post-freeze-viewradius-switch/run.ts
 *
 * 2026-08-22's whole-episode `viewRadius` sweep couldn't separate "does more post-freeze
 * visibility raise the drift signal" from "does a different pre-freeze training regime raise it,"
 * because `viewRadius` gated the environment during both the pre-freeze training window (steps
 * 0-37) and the post-freeze measurement window at once. This run fixes that: `viewRadius` stays at
 * the default (2) for every step before `FREEZE_STEP`, identical across the whole sweep, and is
 * switched only via the new `runEpisode` `postFreezeEnvMutation` hook (src/experiment/freeze.ts) —
 * called once, immediately before the `env.step()` call that produces the `FREEZE_STEP` record, so
 * every post-freeze record (including the first) already reflects the new radius. `env.setViewRadius`
 * (src/env/gridworld.ts) is the actual mutation.
 *
 * Same 3-seed paired-init Arm-A instrument-validation harness as 2026-08-13's/20's/21's/22's runs
 * (identical `SEEDS`, `FREEZE_STEP`, `FROZEN_AGENT_INDEX`, `HORIZON`, `RSSM_CONFIG`,
 * `Q_LEARNING_CONFIG`, paired `WorldModelConfig.seed` per agent, `buildWorldModels`).
 *
 * `assertPreFreezeParity` is strengthened for this run: it no longer just checks control vs.
 * intervention for a fixed radius (as in every prior run) — it also checks each non-baseline
 * radius's control run against the `VIEW_RADII[0]` (2) baseline's control run for the same seed,
 * which should now be bit-identical pre-freeze *by construction*, since the mutation cannot fire
 * before `FREEZE_STEP`. A failure here would mean the confound fix itself is broken, not just that
 * the effect under study is absent.
 *
 * VIEW_RADII: 2 (baseline, matches every prior run's implicit whole-episode value), 4, 6 — chosen
 * below 2026-08-22's reported 38/38 visibility ceiling (radius 8 put all three seeds' *whole-episode*
 * partner-visible-step tally at 38/38 of 38; radius 4 was already at 38/34/11). This post-freeze-only
 * design can still land at a different ceiling than the whole-episode sweep did (pre-freeze
 * trajectories are now fixed at radius 2 for every arm, so post-freeze positions at `FREEZE_STEP`
 * differ from 2026-08-22's), so `postFreezePartnerVisibleCount` is reported per radius here rather
 * than assumed from the prior run's numbers.
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
import {
  driftAttributableError,
  postFreezeActionDivergenceCount,
  postFreezeObservationDivergenceCount,
  postFreezePartnerVisibleCount,
  postFreezeLossSeries,
} from "../../src/experiment/metrics.ts";

const RUN_ID = "2026-08-23-post-freeze-viewradius-switch";
const SEEDS = [1001, 1002, 1003];
const FREEZE_STEP = 38;
const FROZEN_AGENT_INDEX = 0;
const HORIZON = 75; // Matches 2026-08-12's/13's/20's/21's/22's runs.
const VIEW_RADII = [2, 4, 6];
const BASELINE_RADIUS = VIEW_RADII[0]!;

const RSSM_CONFIG = { deterministicSize: 256, latentCategoricals: 8, latentClasses: 4 };
const Q_LEARNING_CONFIG: Required<QLearningConfig> = { alpha: 0.1, gamma: 0.95, epsilon: 0.1 };

const artifactsDir = fileURLToPath(new URL(`../../artifacts/${RUN_ID}/`, import.meta.url));
const gitCommit = execSync("git rev-parse HEAD").toString().trim();

interface RunOutcome {
  viewRadius: number;
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

/** Same pairing as 2026-08-13's/20's/21's/22's `buildWorldModels` — one seed stream per agent index. */
function buildWorldModels(seed: number, observationSize: number): [WorldModel, WorldModel] {
  const configFor = (agentIndex: number): WorldModelConfig => ({
    rssm: RSSM_CONFIG,
    observationSize,
    seed: deriveSeed(seed, agentIndex),
  });
  return [new WorldModel(configFor(0)), new WorldModel(configFor(1))];
}

/** Pre-freeze-only equality check between two completed runs' records — used both for the usual
 * control-vs-intervention check and (new this run) the cross-radius check against the baseline. */
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
  viewRadius: number,
  seed: number,
  condition: FreezeCondition,
  records: EpisodeStepRecord[],
  series: number[],
  parityCheck: Record<string, unknown>,
  crossRadiusParityCheck: Record<string, unknown> | null,
  actionDivergence: { postFreezeSteps: number; divergentSteps: number },
  observationDivergence: { postFreezeSteps: number; divergentSteps: number },
  visibility: { postFreezeSteps: number; visibleSteps: number },
): RunOutcome {
  const dir = `${artifactsDir}viewradius-${viewRadius}-seed-${seed}-${condition}/`;
  mkdirSync(dir, { recursive: true });

  const telemetry = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
  writeFileSync(`${dir}telemetry.jsonl`, telemetry);

  const outcome: RunOutcome = {
    viewRadius,
    seed,
    condition,
    postFreezeSteps: series.length,
    meanLoss: mean(series),
    slopeLoss: slope(series),
  };

  const findings: Record<string, unknown>[] = [
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
        "actions.",
      postFreezeActionDivergenceCount: actionDivergence,
    },
    {
      severity: "NOTE",
      confidence: "self_checked, high confidence",
      summary:
        `Post-freeze observation-divergence count for the frozen agent (index ${FROZEN_AGENT_INDEX}), ` +
        `at post-freeze viewRadius=${viewRadius} (PR #48 review "Next", 2026-08-22): of this seed's ` +
        "post-freeze steps, how many had the frozen agent's own nextObservations differ between " +
        "control and intervention.",
      postFreezeObservationDivergenceCount: observationDivergence,
    },
    {
      severity: "NOTE",
      confidence: "self_checked, high confidence",
      summary:
        "Partner-visible-step tally (PR #49 review, 2026-08-22, landed as metrics.ts's " +
        "postFreezePartnerVisibleCount): of this seed's post-freeze steps, how many had the " +
        "partner visible to the frozen agent in control or intervention (union).",
      postFreezePartnerVisibleCount: visibility,
    },
  ];
  if (crossRadiusParityCheck !== null) {
    findings.push({
      severity: "NOTE",
      confidence: "self_checked, high confidence",
      summary:
        `Cross-radius pre-freeze parity (2026-08-23, this run's confound fix): this radius's ${condition} ` +
        `run's pre-freeze records vs. the viewRadius=${BASELINE_RADIUS} baseline's ${condition} run, same ` +
        "seed — verifies the post-freeze-only mutation left pre-freeze training untouched.",
      crossRadiusPreFreezeParityCheck: crossRadiusParityCheck,
    });
  }

  const manifest = {
    runId: RUN_ID,
    viewRadius,
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
    findings,
  };
  writeFileSync(`${dir}manifest.json`, JSON.stringify(manifest, null, 2) + "\n");

  return outcome;
}

function runOneCondition(
  viewRadius: number,
  seed: number,
  condition: FreezeCondition,
  observationSize: number,
): { outcome: RunOutcome; series: number[]; records: EpisodeStepRecord[] } {
  const env = new CooperativeGridWorld({ seed, horizon: HORIZON, viewRadius: BASELINE_RADIUS });
  const policies = [new QLearningPolicy(Q_LEARNING_CONFIG), new QLearningPolicy(Q_LEARNING_CONFIG)];
  const [wm0, wm1] = buildWorldModels(seed, observationSize);

  const freezeConfig: FreezeConfig =
    condition === "control"
      ? { freezeStep: FREEZE_STEP, condition: "control" }
      : { freezeStep: FREEZE_STEP, condition: "intervention", frozenAgentIndex: FROZEN_AGENT_INDEX };

  const postFreezeEnvMutation =
    viewRadius === BASELINE_RADIUS ? undefined : (mutatedEnv: CooperativeGridWorld) => mutatedEnv.setViewRadius(viewRadius);

  const records = runEpisode(env, policies, seed, freezeConfig, [wm0, wm1], postFreezeEnvMutation);
  const series = postFreezeLossSeries(records, FREEZE_STEP, FROZEN_AGENT_INDEX);

  wm0.dispose();
  wm1.dispose();

  return {
    outcome: { viewRadius, seed, condition, postFreezeSteps: series.length, meanLoss: mean(series), slopeLoss: slope(series) },
    series,
    records,
  };
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const probeEnv = new CooperativeGridWorld({ seed: 0, horizon: HORIZON });
  const observationSize = probeEnv.observationLength;

  const rows: RunOutcome[] = [];
  const sweepRows: {
    viewRadius: number;
    seed: number;
    diffMean: number;
    diffSlope: number;
    actionDivergentSteps: number;
    actionPostFreezeSteps: number;
    observationDivergentSteps: number;
    observationPostFreezeSteps: number;
    partnerVisibleSteps: number;
    partnerVisiblePostFreezeSteps: number;
  }[] = [];

  // Baseline (radius 2) control/intervention pair per seed — every other radius's cross-radius
  // parity check compares against this seed's baseline pair.
  const baselineByseed = new Map<number, { control: EpisodeStepRecord[]; intervention: EpisodeStepRecord[] }>();

  for (const viewRadius of VIEW_RADII) {
    for (const seed of SEEDS) {
      const control = runOneCondition(viewRadius, seed, "control", observationSize);
      const intervention = runOneCondition(viewRadius, seed, "intervention", observationSize);

      if (viewRadius === BASELINE_RADIUS) {
        baselineByseed.set(seed, { control: control.records, intervention: intervention.records });
      }
      const baseline = baselineByseed.get(seed)!;

      const parityCheck = preFreezeParity(control.records, intervention.records);
      if (!parityCheck.identical) {
        console.warn(`viewRadius ${viewRadius}, seed ${seed}: pre-freeze parity check FAILED —`, JSON.stringify(parityCheck));
      }

      const crossRadiusControlParity = viewRadius === BASELINE_RADIUS ? null : preFreezeParity(control.records, baseline.control);
      const crossRadiusInterventionParity =
        viewRadius === BASELINE_RADIUS ? null : preFreezeParity(intervention.records, baseline.intervention);
      if (crossRadiusControlParity && !crossRadiusControlParity.identical) {
        console.warn(
          `viewRadius ${viewRadius}, seed ${seed}, control: cross-radius pre-freeze parity check FAILED — the confound fix is broken —`,
          JSON.stringify(crossRadiusControlParity),
        );
      }
      if (crossRadiusInterventionParity && !crossRadiusInterventionParity.identical) {
        console.warn(
          `viewRadius ${viewRadius}, seed ${seed}, intervention: cross-radius pre-freeze parity check FAILED — the confound fix is broken —`,
          JSON.stringify(crossRadiusInterventionParity),
        );
      }

      const actionDivergence = postFreezeActionDivergenceCount(control.records, intervention.records, FREEZE_STEP);
      const observationDivergence = postFreezeObservationDivergenceCount(
        control.records,
        intervention.records,
        FREEZE_STEP,
        FROZEN_AGENT_INDEX,
      );
      const visibility = postFreezePartnerVisibleCount(control.records, intervention.records, FREEZE_STEP, FROZEN_AGENT_INDEX);

      rows.push(
        writeRun(
          viewRadius,
          seed,
          "control",
          control.records,
          control.series,
          parityCheck,
          crossRadiusControlParity,
          actionDivergence,
          observationDivergence,
          visibility,
        ),
        writeRun(
          viewRadius,
          seed,
          "intervention",
          intervention.records,
          intervention.series,
          parityCheck,
          crossRadiusInterventionParity,
          actionDivergence,
          observationDivergence,
          visibility,
        ),
      );

      const diff = driftAttributableError(intervention.series, control.series);
      const diffMean = mean(diff);
      const diffSlope = slope(diff);
      sweepRows.push({
        viewRadius,
        seed,
        diffMean,
        diffSlope,
        actionDivergentSteps: actionDivergence.divergentSteps,
        actionPostFreezeSteps: actionDivergence.postFreezeSteps,
        observationDivergentSteps: observationDivergence.divergentSteps,
        observationPostFreezeSteps: observationDivergence.postFreezeSteps,
        partnerVisibleSteps: visibility.visibleSteps,
        partnerVisiblePostFreezeSteps: visibility.postFreezeSteps,
      });

      console.log(
        `viewRadius ${viewRadius}, seed ${seed}: diffMean=${diffMean.toFixed(4)} | ` +
          `prefreezeParity=${parityCheck.identical} | ` +
          `crossRadiusParity=${viewRadius === BASELINE_RADIUS ? "n/a (baseline)" : `${crossRadiusControlParity!.identical}/${crossRadiusInterventionParity!.identical}`} | ` +
          `postFreezeActionDivergence=${actionDivergence.divergentSteps}/${actionDivergence.postFreezeSteps} | ` +
          `frozenAgentObservationDivergence=${observationDivergence.divergentSteps}/${observationDivergence.postFreezeSteps} | ` +
          `partnerVisible=${visibility.visibleSteps}/${visibility.postFreezeSteps}`,
      );
    }
  }

  const csvHeader = "viewRadius,seed,condition,postFreezeSteps,meanLoss,slopeLoss\n";
  const csvBody = rows
    .map((r) => `${r.viewRadius},${r.seed},${r.condition},${r.postFreezeSteps},${r.meanLoss},${r.slopeLoss}`)
    .join("\n");
  writeFileSync(`${artifactsDir}results.summary.csv`, csvHeader + csvBody + "\n");

  const sweepCsvHeader =
    "viewRadius,seed,diffMean,diffSlope,actionDivergentSteps,actionPostFreezeSteps," +
    "observationDivergentSteps,observationPostFreezeSteps,partnerVisibleSteps,partnerVisiblePostFreezeSteps\n";
  const sweepCsvBody = sweepRows
    .map(
      (s) =>
        `${s.viewRadius},${s.seed},${s.diffMean},${s.diffSlope},${s.actionDivergentSteps},${s.actionPostFreezeSteps},` +
        `${s.observationDivergentSteps},${s.observationPostFreezeSteps},${s.partnerVisibleSteps},${s.partnerVisiblePostFreezeSteps}`,
    )
    .join("\n");
  writeFileSync(`${artifactsDir}sweep.summary.csv`, sweepCsvHeader + sweepCsvBody + "\n");

  console.log("\n--- post-freeze-only viewRadius switch summary (n=3 seeds per radius, descriptive) ---");
  for (const viewRadius of VIEW_RADII) {
    const rowsAtRadius = sweepRows.filter((s) => s.viewRadius === viewRadius);
    const meanAbsDiffMean = mean(rowsAtRadius.map((s) => Math.abs(s.diffMean)));
    const meanObservationDivergence = mean(rowsAtRadius.map((s) => s.observationDivergentSteps));
    const meanPartnerVisible = mean(rowsAtRadius.map((s) => s.partnerVisibleSteps));
    console.log(
      `viewRadius ${viewRadius}: mean|diffMean|=${meanAbsDiffMean.toFixed(4)}, ` +
        `mean frozen-agent observation-divergent steps=${meanObservationDivergence.toFixed(2)}/38, ` +
        `mean partner-visible steps=${meanPartnerVisible.toFixed(2)}/38`,
    );
  }
}

main();
