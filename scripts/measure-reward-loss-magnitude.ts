/**
 * Measures rewardLoss's actual magnitude vs. continueLoss at world-model init, to check
 * docs/explainers/0014-reward-head-spec.md's "one to two orders above continueLoss/
 * reconstructionLoss" premise (loop/GOAL.md priority-1 processing of PR #77's review,
 * 2026-09-21: "the numbers don't reproduce ... the step Rng seed is unstated and the script is
 * gone" — every seed used here is stated explicitly, this file is that script).
 *
 * For each of 24 seed combos: build a `CooperativeGridWorld` (DEFAULT_CONFIG) with `envSeed`,
 * reset it, pick one action per agent from `actionRng` (seeded from the same combo), take one
 * real `env.step()`, then build a fresh `WorldModel` with `modelSeed` and run one `step()`
 * (`train: false`, agent 0's transition) using `stepRng` (seeded from the same combo) — the RNG
 * `RSSMCell.prior`/`.posterior` consume for their Gumbel/categorical sampling, i.e. exactly the
 * "step Rng" the PR #77 review's reproduction attempt found unstated. Reports median/max
 * normalized `rewardLoss`, median `continueLoss`, and median/max raw (unnormalized) reward MSE
 * (`rewardLoss * rewardScale^2`, exact given `rewardLoss`'s division-before-squaring definition
 * — see `src/model/losses.ts`).
 *
 * Run with: node scripts/measure-reward-loss-magnitude.ts
 */
import { CooperativeGridWorld } from "../src/env/gridworld.ts";
import { WorldModel } from "../src/model/worldModel.ts";
import { Action, DEFAULT_CONFIG } from "../src/env/types.ts";
import { Rng, deriveSeed } from "../src/env/rng.ts";

const NUM_COMBOS = 24;
const RSSM_CONFIG = { deterministicSize: 256, latentCategoricals: 8, latentClasses: 4 };
const ACTIONS = [Action.Stay, Action.Up, Action.Down, Action.Left, Action.Right];

interface Sample {
  seed: number;
  reward: number;
  reconstructionLoss: number;
  continueLoss: number;
  rewardLossNormalized: number;
  rewardLossRaw: number;
}

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function runOneCombo(seed: number): Sample {
  // Three independent streams derived from one combo seed, same
  // deriveSeed-per-stream convention WorldModelConfig.seed itself uses
  // internally — kept distinct so this script's own seed choices don't
  // accidentally correlate the env's spawn RNG, the action-selection RNG,
  // and the RSSM's step RNG.
  const envSeed = deriveSeed(seed, 0);
  const actionSeed = deriveSeed(seed, 1);
  const modelSeed = deriveSeed(seed, 2);
  const stepRngSeed = deriveSeed(seed, 3);

  const env = new CooperativeGridWorld({ ...DEFAULT_CONFIG, seed: envSeed });
  env.reset();
  const actionRng = new Rng(actionSeed);
  const actions = [
    ACTIONS[actionRng.nextInt(ACTIONS.length)],
    ACTIONS[actionRng.nextInt(ACTIONS.length)],
  ];
  const result = env.step(actions);

  const worldModel = new WorldModel({
    rssm: RSSM_CONFIG,
    observationSize: env.observationLength,
    rewardScale: env.rewardScale,
    seed: modelSeed,
  });
  const stepRng = new Rng(stepRngSeed);
  const stepResult = worldModel.step(
    actions[0],
    result.observations[0],
    stepRng,
    false,
    result.done,
    result.reward,
  );
  worldModel.dispose();

  return {
    seed,
    reward: result.reward,
    reconstructionLoss: stepResult.reconstructionLoss,
    continueLoss: stepResult.continueLoss,
    rewardLossNormalized: stepResult.rewardLoss,
    rewardLossRaw: stepResult.rewardLoss * env.rewardScale ** 2,
  };
}

const samples: Sample[] = [];
for (let seed = 0; seed < NUM_COMBOS; seed++) {
  samples.push(runOneCombo(seed));
}

console.log("seed, reward, reconstructionLoss, continueLoss, rewardLoss(normalized), rewardLoss(raw MSE)");
for (const s of samples) {
  console.log(
    `${s.seed}, ${s.reward.toFixed(4)}, ${s.reconstructionLoss.toFixed(4)}, ` +
      `${s.continueLoss.toFixed(4)}, ${s.rewardLossNormalized.toFixed(4)}, ${s.rewardLossRaw.toFixed(4)}`,
  );
}

const rewards = samples.map((s) => s.reward);
const reconstructionLosses = samples.map((s) => s.reconstructionLoss);
const continueLosses = samples.map((s) => s.continueLoss);
const normalized = samples.map((s) => s.rewardLossNormalized);
const raw = samples.map((s) => s.rewardLossRaw);

console.log(`\n${NUM_COMBOS} seed combos (seed 0..${NUM_COMBOS - 1}, each deriveSeed-split into env/action/model/stepRng streams):`);
console.log(`reward: median=${median(rewards).toFixed(4)}, min=${Math.min(...rewards).toFixed(4)}, max=${Math.max(...rewards).toFixed(4)}`);
console.log(`reconstructionLoss: median=${median(reconstructionLosses).toFixed(4)}, max=${Math.max(...reconstructionLosses).toFixed(4)}`);
console.log(`continueLoss: median=${median(continueLosses).toFixed(4)}, max=${Math.max(...continueLosses).toFixed(4)}`);
console.log(`rewardLoss (normalized): median=${median(normalized).toFixed(4)}, max=${Math.max(...normalized).toFixed(4)}`);
console.log(`rewardLoss (raw MSE): median=${median(raw).toFixed(4)}, max=${Math.max(...raw).toFixed(4)}`);
