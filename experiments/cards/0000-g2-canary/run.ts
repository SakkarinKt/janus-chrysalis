/**
 * Card 0000 — canary. Re-runs the Gate G2 loss-curve experiment
 * (`experiments/2026-09-23-g2-loss-curve/run.ts`) at HEAD and compares the regenerated
 * `loss-curve.summary.csv` with the committed one. Run it with `npm run exp -- 0000`.
 *
 * The training loop below is that script's `runSeed`, kept step for step: same construction
 * order, same per-episode seed derivation, same summation order in the means, same CSV row
 * order. Any of those differing would change the output for a reason that has nothing to do
 * with the code under test. What is left out is only what the comparison does not need
 * (telemetry, the gate statistics, the per-seed manifests).
 *
 * Everything that parameterises the run comes from `card.json`; there are no constants here to
 * drift away from it.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { RandomPolicy } from "../../../src/agent/policy.ts";
import { CooperativeGridWorld } from "../../../src/env/gridworld.ts";
import { deriveSeed } from "../../../src/env/rng.ts";
import type { RunContext, RunResult } from "../../../src/experiment/card.ts";
import { compareCsv } from "../../../src/experiment/compare.ts";
import { runEpisode } from "../../../src/experiment/freeze.ts";
import type { EpisodeStepRecord } from "../../../src/experiment/freeze.ts";
import { WorldModel, WorldModelNaNError } from "../../../src/model/worldModel.ts";
import type { WorldModelConfig } from "../../../src/model/worldModel.ts";

const COMPONENTS = ["loss", "reconstructionLoss", "klLoss", "continueLoss", "rewardLoss"] as const;
type Component = (typeof COMPONENTS)[number];
/** One agent's per-episode mean of each loss component. */
type Curves = Record<Component, number[]>;

const NUM_AGENTS = 2;
const OUTPUT = "loss-curve.summary.csv";

interface CanaryConfig {
  reference: string;
  episodes: number;
  horizon: number;
  rssm: WorldModelConfig["rssm"];
}

function positiveInteger(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`card config.${name} must be a positive integer`);
  }
  return value;
}

function readConfig(config: Record<string, unknown>): CanaryConfig {
  const { reference, rssm } = config;
  if (typeof reference !== "string" || reference === "") {
    throw new Error("card config.reference must be a path");
  }
  if (typeof rssm !== "object" || rssm === null) {
    throw new Error("card config.rssm must be an object");
  }
  const sizes = rssm as Record<string, unknown>;
  return {
    reference,
    episodes: positiveInteger(config.episodes, "episodes"),
    horizon: positiveInteger(config.horizon, "horizon"),
    rssm: {
      deterministicSize: positiveInteger(sizes.deterministicSize, "rssm.deterministicSize"),
      latentCategoricals: positiveInteger(sizes.latentCategoricals, "rssm.latentCategoricals"),
      latentClasses: positiveInteger(sizes.latentClasses, "rssm.latentClasses"),
    },
  };
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Per-agent mean of one loss component over one episode's records. */
function episodeMean(records: EpisodeStepRecord[], agent: number, component: Component): number {
  return mean(
    records.map((record) => {
      const breakdown = record.worldModelLossBreakdown[agent];
      const total = record.worldModelLoss[agent];
      if (breakdown === undefined || total === undefined) {
        throw new Error(`step ${record.step}: agent ${agent} has no world-model loss`);
      }
      return component === "loss" ? total : breakdown[component];
    }),
  );
}

/** Trains two fresh world models for `config.episodes` episodes and returns each agent's curves. */
function runSeed(seed: number, config: CanaryConfig, observationSize: number, rewardScale: number): Curves[] {
  const worldModels = Array.from(
    { length: NUM_AGENTS },
    (_, agentIndex) => new WorldModel({ rssm: config.rssm, observationSize, rewardScale, seed: deriveSeed(seed, agentIndex) }),
  );
  const policies = [new RandomPolicy(), new RandomPolicy()];
  const curves: Curves[] = worldModels.map(() => ({ loss: [], reconstructionLoss: [], klLoss: [], continueLoss: [], rewardLoss: [] }));
  try {
    for (let episode = 0; episode < config.episodes; episode++) {
      const env = new CooperativeGridWorld({ seed: deriveSeed(seed, 1000 + episode), horizon: config.horizon });
      const records = runEpisode(env, policies, deriveSeed(seed, 2000 + episode), undefined, worldModels);
      curves.forEach((curve, agent) => {
        for (const component of COMPONENTS) {
          curve[component].push(episodeMean(records, agent, component));
        }
      });
    }
  } finally {
    worldModels.forEach((worldModel) => worldModel.dispose());
  }
  return curves;
}

export function run(context: RunContext): RunResult {
  const config = readConfig(context.card.config);
  const tolerance = context.card.decisionRule.thresholds.maxDifference;
  if (tolerance === undefined) {
    throw new Error("card decisionRule.thresholds.maxDifference is missing");
  }
  const probe = new CooperativeGridWorld({ seed: 0, horizon: config.horizon });
  const { observationLength, rewardScale } = probe;

  const rows: string[] = [];
  const wallSecondsPerSeed: Record<string, number> = {};
  try {
    for (const seed of context.seeds) {
      const started = Date.now();
      const curves = runSeed(seed, config, observationLength, rewardScale);
      wallSecondsPerSeed[String(seed)] = (Date.now() - started) / 1000;
      context.log(`seed ${seed}: ${wallSecondsPerSeed[String(seed)]} s`);
      curves.forEach((curve, agent) => {
        for (let episode = 0; episode < config.episodes; episode++) {
          rows.push([seed, agent, episode + 1, ...COMPONENTS.map((component) => curve[component][episode])].join(","));
        }
      });
    }
  } catch (error) {
    if (error instanceof WorldModelNaNError) {
      // The reference run had no NaN halt, so halting here is a failure to reproduce, not "no result".
      return { primary: { name: "maxDifference", value: null }, verdict: "fail", resultSummary: { nanHalt: error.message } };
    }
    throw error;
  }

  const regenerated = `seed,agent,episode,${COMPONENTS.join(",")}\n${rows.join("\n")}\n`;
  writeFileSync(join(context.outDir, OUTPUT), regenerated);
  const comparison = compareCsv(OUTPUT, readFileSync(config.reference, "utf8"), regenerated, tolerance);
  context.log(
    `${comparison.identical ? "byte-identical" : `largest difference ${comparison.maxDifference}`} ` +
      `(${comparison.mismatchCount} cell(s) beyond tolerance ${tolerance})`,
  );
  return {
    primary: { name: "maxDifference", value: comparison.maxDifference },
    verdict: comparison.match ? "pass" : "fail",
    resultSummary: { reference: config.reference, comparison, wallSecondsPerSeed },
  };
}
