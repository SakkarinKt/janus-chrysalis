/**
 * Processes PR #48's review (@SakkarinKt, 2026-08-21 merge comment) "Next" line: "viewRadius on
 * the same 3-seed paired-init harness — does raising visible-step counts raise |diffMean|; the
 * multi-episode horizon extension stays queued behind it." This is also the answer to the
 * 2026-08-21 stand-up's "Decisions needed" item (pursue `viewRadius` vs. the multi-episode
 * horizon extension) — the human chose `viewRadius`.
 *
 *   node experiments/2026-08-22-viewradius-sweep/run.ts
 *
 * Same 3-seed paired-init Arm-A instrument-validation harness as 2026-08-13's/20's/21's runs
 * (identical `SEEDS`, `FREEZE_STEP`, `FROZEN_AGENT_INDEX`, `HORIZON`, `RSSM_CONFIG`,
 * `Q_LEARNING_CONFIG`, paired `WorldModelConfig.seed` per agent, `buildWorldModels`,
 * `assertPreFreezeParity`), with one new free variable: `CooperativeGridWorld`'s `viewRadius`
 * (`src/env/types.ts`'s `GridWorldConfig.viewRadius`, default 2 — the value every prior run used
 * implicitly). `observationLength` doesn't depend on `viewRadius` (only on `numLandmarks`,
 * `src/env/gridworld.ts`'s `observationLength` getter), so `WorldModel`'s I/O shape is unaffected
 * across the sweep — only what's *encoded* in the "other agent" slot changes (masked
 * visible=0/dx=dy=0 vs. an actual relative offset).
 *
 * `viewRadius` also gates the environment during the pre-freeze training window (steps 0-37), not
 * only the post-freeze observation window measured by `postFreezeObservationDivergenceCount` — a
 * larger `viewRadius` changes what both agents see and therefore how their Q-tables end up shaped
 * by step 38, not just how visible the partner is afterward. `diffMean` and the divergence counts
 * below are read as one joint measurement of "does a wider view radius change this instrument's
 * output," not as an isolated post-freeze-only manipulation.
 *
 * VIEW_RADII: 2 (baseline — every prior run's implicit value, included here for a same-run,
 * same-commit comparison point rather than citing 2026-08-21's numbers out of context), 4
 * (doubles the baseline), 8 (grid is 8x8, `gridSize: 8` in DEFAULT_CONFIG — a radius equal to the
 * grid side covers most of the board from any position, though corner-to-corner Manhattan
 * distance can reach 14).
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
  postFreezeLossSeries,
} from "../../src/experiment/metrics.ts";

const RUN_ID = "2026-08-22-viewradius-sweep";
const SEEDS = [1001, 1002, 1003];
const FREEZE_STEP = 38;
const FROZEN_AGENT_INDEX = 0;
const HORIZON = 75; // Matches 2026-08-12's/13's/20's/21's runs.
const VIEW_RADII = [2, 4, 8];

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

/** Same pairing as 2026-08-13's/20's/21's `buildWorldModels` — one seed stream per agent index. */
function buildWorldModels(seed: number, observationSize: number): [WorldModel, WorldModel] {
  const configFor = (agentIndex: number): WorldModelConfig => ({
    rssm: RSSM_CONFIG,
    observationSize,
    seed: deriveSeed(seed, agentIndex),
  });
  return [new WorldModel(configFor(0)), new WorldModel(configFor(1))];
}

/** Same diagnostic as prior runs — confirms pairing still holds on this commit before spending the run. */
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
  viewRadius: number,
  seed: number,
  condition: FreezeCondition,
  records: EpisodeStepRecord[],
  series: number[],
  parityCheck: Record<string, unknown>,
  actionDivergence: { postFreezeSteps: number; divergentSteps: number },
  observationDivergence: { postFreezeSteps: number; divergentSteps: number },
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
          "actions.",
        postFreezeActionDivergenceCount: actionDivergence,
      },
      {
        severity: "NOTE",
        confidence: "self_checked, high confidence",
        summary:
          `Post-freeze observation-divergence count for the frozen agent (index ${FROZEN_AGENT_INDEX}), ` +
          `at viewRadius=${viewRadius} (PR #48 review "Next", 2026-08-22): of this seed's ` +
          "post-freeze steps, how many had the frozen agent's own nextObservations differ between " +
          "control and intervention.",
        postFreezeObservationDivergenceCount: observationDivergence,
      },
    ],
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
  const env = new CooperativeGridWorld({ seed, horizon: HORIZON, viewRadius });
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
  }[] = [];

  for (const viewRadius of VIEW_RADII) {
    for (const seed of SEEDS) {
      const control = runOneCondition(viewRadius, seed, "control", observationSize);
      const intervention = runOneCondition(viewRadius, seed, "intervention", observationSize);

      const parityCheck = assertPreFreezeParity(control.records, intervention.records);
      if (!parityCheck.identical) {
        console.warn(`viewRadius ${viewRadius}, seed ${seed}: pre-freeze parity check FAILED —`, JSON.stringify(parityCheck));
      }

      const actionDivergence = postFreezeActionDivergenceCount(control.records, intervention.records, FREEZE_STEP);
      const observationDivergence = postFreezeObservationDivergenceCount(
        control.records,
        intervention.records,
        FREEZE_STEP,
        FROZEN_AGENT_INDEX,
      );

      rows.push(
        writeRun(viewRadius, seed, "control", control.records, control.series, parityCheck, actionDivergence, observationDivergence),
        writeRun(viewRadius, seed, "intervention", intervention.records, intervention.series, parityCheck, actionDivergence, observationDivergence),
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
      });

      console.log(
        `viewRadius ${viewRadius}, seed ${seed}: diffMean=${diffMean.toFixed(4)} | ` +
          `prefreezeParity=${parityCheck.identical} | ` +
          `postFreezeActionDivergence=${actionDivergence.divergentSteps}/${actionDivergence.postFreezeSteps} | ` +
          `frozenAgentObservationDivergence=${observationDivergence.divergentSteps}/${observationDivergence.postFreezeSteps}`,
      );
    }
  }

  const csvHeader = "viewRadius,seed,condition,postFreezeSteps,meanLoss,slopeLoss\n";
  const csvBody = rows
    .map((r) => `${r.viewRadius},${r.seed},${r.condition},${r.postFreezeSteps},${r.meanLoss},${r.slopeLoss}`)
    .join("\n");
  writeFileSync(`${artifactsDir}results.summary.csv`, csvHeader + csvBody + "\n");

  const sweepCsvHeader =
    "viewRadius,seed,diffMean,diffSlope,actionDivergentSteps,actionPostFreezeSteps,observationDivergentSteps,observationPostFreezeSteps\n";
  const sweepCsvBody = sweepRows
    .map(
      (s) =>
        `${s.viewRadius},${s.seed},${s.diffMean},${s.diffSlope},${s.actionDivergentSteps},${s.actionPostFreezeSteps},` +
        `${s.observationDivergentSteps},${s.observationPostFreezeSteps}`,
    )
    .join("\n");
  writeFileSync(`${artifactsDir}sweep.summary.csv`, sweepCsvHeader + sweepCsvBody + "\n");

  console.log("\n--- viewRadius sweep summary (n=3 seeds per viewRadius, descriptive) ---");
  for (const viewRadius of VIEW_RADII) {
    const rowsAtRadius = sweepRows.filter((s) => s.viewRadius === viewRadius);
    const meanAbsDiffMean = mean(rowsAtRadius.map((s) => Math.abs(s.diffMean)));
    const meanObservationDivergence = mean(rowsAtRadius.map((s) => s.observationDivergentSteps));
    console.log(
      `viewRadius ${viewRadius}: mean|diffMean|=${meanAbsDiffMean.toFixed(4)}, ` +
        `mean frozen-agent observation-divergent steps=${meanObservationDivergence.toFixed(2)}/38`,
    );
  }
}

main();
