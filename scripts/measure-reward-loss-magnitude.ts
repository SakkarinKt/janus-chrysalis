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
 * Run with: node scripts/measure-reward-loss-magnitude.ts [--scratch]
 *
 * Since 2026-11-09 the run is recorded, not just printed: it writes `samples.summary.csv` (one
 * row per seed combo, full precision) and a `manifest.json` to
 * `artifacts/reward-loss-magnitude/<runId>/`, so a number quoted from it has an artifact behind
 * it. A recorded run needs a clean tree; `--scratch` writes under `artifacts/_scratch/` instead
 * and runs on any tree. The measurement itself (seeds, draws, order) is unchanged.
 */
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { CooperativeGridWorld } from "../src/env/gridworld.ts";
import { WorldModel } from "../src/model/worldModel.ts";
import { Action, DEFAULT_CONFIG } from "../src/env/types.ts";
import { Rng, deriveSeed } from "../src/env/rng.ts";
import { makeRunId } from "../src/experiment/harness.ts";
import { canonicalJson, hashFile, hashFiles, sha256 } from "../src/experiment/provenance.ts";
import { captureRunEnvironment, ensureRunDir, writeManifest } from "../src/experiment/telemetry.ts";
import { tfjsInfo } from "../src/experiment/tfjsInfo.ts";

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

/** `items[index]`, or a thrown error where TypeScript only sees "possibly undefined". */
function at<T>(items: readonly T[], index: number): T {
  const item = items[index];
  if (item === undefined) {
    throw new Error(`index ${index} is out of range (length ${items.length})`);
  }
  return item;
}

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (at(sorted, mid - 1) + at(sorted, mid)) / 2 : at(sorted, mid);
}

function runOneCombo(seed: number): Sample {
  // Four independent streams derived from one combo seed, same
  // deriveSeed-per-stream convention WorldModelConfig.seed itself uses
  // internally — kept distinct so this script's own seed choices don't
  // accidentally correlate the env's spawn RNG, the action-selection RNG,
  // the model's init RNG, and the RSSM's step RNG.
  const envSeed = deriveSeed(seed, 0);
  const actionSeed = deriveSeed(seed, 1);
  const modelSeed = deriveSeed(seed, 2);
  const stepRngSeed = deriveSeed(seed, 3);

  const env = new CooperativeGridWorld({ ...DEFAULT_CONFIG, seed: envSeed });
  env.reset();
  const actionRng = new Rng(actionSeed);
  const actions = [at(ACTIONS, actionRng.nextInt(ACTIONS.length)), at(ACTIONS, actionRng.nextInt(ACTIONS.length))];
  const result = env.step(actions);

  const worldModel = new WorldModel({
    rssm: RSSM_CONFIG,
    observationSize: env.observationLength,
    rewardScale: env.rewardScale,
    seed: modelSeed,
  });
  const stepRng = new Rng(stepRngSeed);
  const stepResult = worldModel.step(
    at(actions, 0),
    at(result.observations, 0),
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

const scratch = process.argv.slice(2).includes("--scratch");
const unknownArgs = process.argv.slice(2).filter((arg) => arg !== "--scratch");
if (unknownArgs.length > 0) {
  console.error(`measure-reward-loss-magnitude: unknown argument(s) ${unknownArgs.join(" ")} (only --scratch is accepted)`);
  process.exit(2);
}

// Captured before anything is written, so gitDirty describes the code and not this run's output.
const root = fileURLToPath(new URL("..", import.meta.url));
const env = captureRunEnvironment(tfjsInfo(), root);
if (env.gitDirty && !scratch) {
  console.error(
    "measure-reward-loss-magnitude: the working tree has uncommitted or untracked files (other than recorded results under " +
      "artifacts/ and the ledger), so the recorded commit would not describe what ran. Commit first, or pass --scratch.",
  );
  process.exit(1);
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

// --- Record the run -----------------------------------------------------------------------

const RUN_NAME = "reward-loss-magnitude";
const runId = makeRunId(env.createdAt, env.gitCommit);
const outDir = join(root, "artifacts", ...(scratch ? ["_scratch"] : []), RUN_NAME, runId);
ensureRunDir(outDir);

const COLUMNS = ["seed", "reward", "reconstructionLoss", "continueLoss", "rewardLossNormalized", "rewardLossRaw"] as const;
writeFileSync(
  join(outDir, "samples.summary.csv"),
  `${COLUMNS.join(",")}\n${samples.map((s) => COLUMNS.map((column) => s[column]).join(",")).join("\n")}\n`,
);

const config = { combos: NUM_COMBOS, rssm: RSSM_CONFIG, envConfig: "DEFAULT_CONFIG", train: false, agent: 0 };
const scriptPath = fileURLToPath(import.meta.url);
writeManifest(outDir, {
  schemaVersion: 2,
  runId,
  script: "scripts/measure-reward-loss-magnitude.ts",
  scriptSha256: hashFile(scriptPath),
  scratch,
  command: `node scripts/measure-reward-loss-magnitude.ts${scratch ? " --scratch" : ""}`,
  env,
  config,
  configSha256: sha256(canonicalJson(config)),
  seeds: samples.map((s) => s.seed),
  resultSummary: {
    reward: { median: median(rewards), min: Math.min(...rewards), max: Math.max(...rewards) },
    reconstructionLoss: { median: median(reconstructionLosses), max: Math.max(...reconstructionLosses) },
    continueLoss: { median: median(continueLosses), max: Math.max(...continueLosses) },
    rewardLossNormalized: { median: median(normalized), max: Math.max(...normalized) },
    rewardLossRaw: { median: median(raw), max: Math.max(...raw) },
  },
  files: hashFiles(outDir),
});
console.log(`\nRecorded: ${join("artifacts", ...(scratch ? ["_scratch"] : []), RUN_NAME, runId, "manifest.json")}`);
