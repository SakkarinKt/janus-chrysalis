/**
 * Gate G2 criterion 1 — "loss decreases across 3 seeds" (`PLAN.html`, Gate G2). Run with:
 *
 *   node experiments/2026-09-23-g2-loss-curve/run.ts
 *
 * No committed artifact showed this before today. Every Arm-A manifest records only the
 * *frozen* agent's post-freeze evaluation loss over one 75-step episode, and the only
 * "loss goes down" evidence was three single-seed unit tests on a fixed synthetic observation
 * (`test/model/worldModel.test.ts`). Session audit 2026-09-23 + Gate G2 evidence inventory.
 *
 * ## Design
 *
 * - Base: `experiments/2026-09-08-seeded-arm-a-instrument-validation/run.ts` (seeded per-agent
 *   `buildWorldModels`, same Arm-A dims), plus the now-required `rewardScale` (PR #77).
 * - **No freeze**: `freezeConfig` omitted, both agents' world models train on every step.
 * - **20 episodes per seed**, same two `WorldModel` instances throughout (`runEpisode` resets the
 *   recurrent state only, never weights). Each episode gets its own env seed and action seed
 *   (`deriveSeed(seed, 1000 + e)` / `deriveSeed(seed, 2000 + e)`), because
 *   `CooperativeGridWorld.reset()` reseeds from `config.seed` and `runEpisode` seeds its action
 *   stream from `seed`. With fixed seeds all 20 episodes would be the identical trajectory, and a
 *   falling loss would only show memorisation.
 * - **`RandomPolicy`** for both agents: a stationary data distribution, the cleanest test of
 *   "does the world model learn". Co-learning non-stationarity is the research question, not
 *   this gate's.
 * - **The loss is prequential.** `WorldModel.step()` reports each step's loss from the forward
 *   pass *before* that step's Adam update, so every number is the model's error on a transition
 *   it has not trained on yet. A falling curve therefore reflects learning that generalises
 *   across fresh episodes, not fit to data already seen.
 * - Seeds 1001, 1002, 1003 (same as every Arm-A run).
 *
 * ## Pre-registered pass rule (written and committed before the first run)
 *
 * For **every** one of the 3 seeds × 2 agents:
 *   1. mean total `loss` over episode 20 < mean total `loss` over episode 1, and
 *   2. the OLS slope of the 20 per-episode mean total losses (against episode index) is < 0, and
 *   3. no `WorldModelNaNError` is thrown anywhere in the run.
 * G2-1 is MET only if all 6 (seed, agent) pairs pass 1–2 and 3 holds. Anything else is reported
 * as NOT MET in `docs/adr/0004`, not re-run with different settings.
 *
 * Reported but **not** part of the rule: the same two statistics per loss component
 * (reconstruction, KL, continue, reward). KL is floored by free bits (default
 * `freeBits = 1`, β_dyn 1.0 + β_rep 0.1 → floor ≈ 1.1), so most of any decrease is expected
 * from the other three terms.
 *
 * Outputs under `artifacts/2026-09-23-g2-loss-curve/`: `seed-<seed>/manifest.json` (+ local,
 * gitignored `telemetry.jsonl`), and `loss-curve.summary.csv` (one row per seed × agent ×
 * episode) plus `gate.summary.csv` (one row per seed × agent: the pass-rule statistics).
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import tf from "@tensorflow/tfjs-node";
import { CooperativeGridWorld } from "../../src/env/gridworld.ts";
import { RandomPolicy } from "../../src/agent/policy.ts";
import { WorldModel } from "../../src/model/worldModel.ts";
import type { WorldModelConfig } from "../../src/model/worldModel.ts";
import { deriveSeed } from "../../src/env/rng.ts";
import { runEpisode } from "../../src/experiment/freeze.ts";
import type { EpisodeStepRecord } from "../../src/experiment/freeze.ts";

const RUN_ID = "2026-09-23-g2-loss-curve";
const SEEDS = [1001, 1002, 1003];
const EPISODES = 20;
const HORIZON = 75; // DEFAULT_CONFIG.horizon — not overridden.
const NUM_AGENTS = 2;
// Same dims as every Arm-A run (2026-07-21 benchmark's ARM_A_CONFIG onward).
const RSSM_CONFIG = { deterministicSize: 256, latentCategoricals: 8, latentClasses: 4 };

const COMPONENTS = ["loss", "reconstructionLoss", "klLoss", "continueLoss", "rewardLoss"] as const;
type Component = (typeof COMPONENTS)[number];

const artifactsDir = fileURLToPath(new URL(`../../artifacts/${RUN_ID}/`, import.meta.url));
const gitCommit = execSync("git rev-parse HEAD").toString().trim();

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** OLS slope of `values` against their index (0, 1, 2, ...). */
function slope(values: number[]): number {
  const xMean = (values.length - 1) / 2;
  const yMean = mean(values);
  let num = 0;
  let den = 0;
  values.forEach((y, x) => {
    num += (x - xMean) * (y - yMean);
    den += (x - xMean) ** 2;
  });
  return den === 0 ? 0 : num / den;
}

/** Per-agent mean of one loss component over one episode's records. */
function episodeMean(records: EpisodeStepRecord[], agent: number, component: Component): number {
  return mean(
    records.map((r) => {
      const b = r.worldModelLossBreakdown[agent];
      const total = r.worldModelLoss[agent];
      if (b === undefined || total === undefined) {
        throw new Error(`step ${r.step}: agent ${agent} has no world-model loss`);
      }
      return component === "loss" ? total : b[component];
    }),
  );
}

interface PairStats {
  seed: number;
  agent: number;
  first: number;
  last: number;
  slope: number;
  pass: boolean;
  components: Record<Component, { first: number; last: number; slope: number }>;
}

function runSeed(seed: number, observationSize: number, rewardScale: number) {
  const configFor = (agentIndex: number): WorldModelConfig => ({
    rssm: RSSM_CONFIG,
    observationSize,
    rewardScale,
    seed: deriveSeed(seed, agentIndex),
  });
  const worldModels = [new WorldModel(configFor(0)), new WorldModel(configFor(1))];
  const policies = [new RandomPolicy(), new RandomPolicy()];

  // perEpisode[agent][component][episode]
  const perEpisode = Array.from({ length: NUM_AGENTS }, () =>
    Object.fromEntries(COMPONENTS.map((c) => [c, [] as number[]])) as Record<Component, number[]>,
  );
  const telemetry: string[] = [];
  const started = Date.now();

  for (let e = 0; e < EPISODES; e++) {
    const env = new CooperativeGridWorld({ seed: deriveSeed(seed, 1000 + e), horizon: HORIZON });
    const records = runEpisode(env, policies, deriveSeed(seed, 2000 + e), undefined, worldModels);
    for (const r of records) telemetry.push(JSON.stringify({ episode: e + 1, ...r }));
    for (let a = 0; a < NUM_AGENTS; a++) {
      for (const c of COMPONENTS) perEpisode[a]![c].push(episodeMean(records, a, c));
    }
    console.log(
      `seed ${seed} episode ${e + 1}/${EPISODES}: ` +
        perEpisode.map((p, a) => `agent${a} loss=${p.loss.at(-1)!.toFixed(4)}`).join(" "),
    );
  }
  const wallSeconds = (Date.now() - started) / 1000;
  worldModels.forEach((wm) => wm.dispose());

  const stats: PairStats[] = perEpisode.map((p, agent) => {
    const components = Object.fromEntries(
      COMPONENTS.map((c) => [c, { first: p[c][0]!, last: p[c][EPISODES - 1]!, slope: slope(p[c]) }]),
    ) as PairStats["components"];
    const { first, last, slope: s } = components.loss;
    return { seed, agent, first, last, slope: s, pass: last < first && s < 0, components };
  });

  const dir = `${artifactsDir}seed-${seed}/`;
  mkdirSync(dir, { recursive: true });
  writeFileSync(`${dir}telemetry.jsonl`, telemetry.join("\n") + "\n");
  const manifest = {
    runId: RUN_ID,
    seed,
    episodes: EPISODES,
    horizon: HORIZON,
    policy: "RandomPolicy (both agents)",
    freeze: null,
    rssmConfig: RSSM_CONFIG,
    rewardScale,
    lossConfig: "defaults (freeBits 1, betaDyn 1.0, betaRep 0.1)",
    weightInitSeeds: { agent0: deriveSeed(seed, 0), agent1: deriveSeed(seed, 1) },
    episodeSeeds: {
      env: "deriveSeed(seed, 1000 + episodeIndex)",
      runEpisode: "deriveSeed(seed, 2000 + episodeIndex)",
    },
    worldModelSteps: EPISODES * HORIZON * NUM_AGENTS,
    wallSeconds,
    nanHalt: false,
    gitCommit,
    nodeVersion: process.version,
    tfjsBackend: tf.getBackend(),
    createdAt: new Date().toISOString(),
    telemetryFile: "telemetry.jsonl",
    preRegisteredRule:
      "per (seed, agent): episode-20 mean total loss < episode-1 mean AND OLS slope over 20 episode means < 0; no WorldModelNaNError",
    resultSummary: stats,
  };
  writeFileSync(`${dir}manifest.json`, JSON.stringify(manifest, null, 2) + "\n");
  return { perEpisode, stats, wallSeconds };
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const probeEnv = new CooperativeGridWorld({ seed: 0, horizon: HORIZON });
  const { observationLength, rewardScale } = probeEnv;

  const curveRows: string[] = [];
  const gateRows: string[] = [];
  const allStats: PairStats[] = [];

  for (const seed of SEEDS) {
    // A WorldModelNaNError propagates and aborts the run: the pre-registered rule counts it as
    // a failure, and no manifest claiming nanHalt: false gets written for that seed.
    const { perEpisode, stats, wallSeconds } = runSeed(seed, observationLength, rewardScale);
    console.log(`seed ${seed}: ${wallSeconds.toFixed(1)} s`);
    perEpisode.forEach((p, agent) => {
      for (let e = 0; e < EPISODES; e++) {
        curveRows.push([seed, agent, e + 1, ...COMPONENTS.map((c) => p[c][e])].join(","));
      }
    });
    for (const s of stats) {
      allStats.push(s);
      gateRows.push(
        [
          s.seed,
          s.agent,
          ...COMPONENTS.flatMap((c) => [s.components[c].first, s.components[c].last, s.components[c].slope]),
          s.pass,
        ].join(","),
      );
    }
  }

  writeFileSync(
    `${artifactsDir}loss-curve.summary.csv`,
    `seed,agent,episode,${COMPONENTS.join(",")}\n${curveRows.join("\n")}\n`,
  );
  writeFileSync(
    `${artifactsDir}gate.summary.csv`,
    `seed,agent,${COMPONENTS.flatMap((c) => [`${c}_ep1`, `${c}_ep${EPISODES}`, `${c}_slope`]).join(",")},pass\n` +
      `${gateRows.join("\n")}\n`,
  );

  console.log("\n--- Gate G2-1 (pre-registered) ---");
  for (const s of allStats) {
    console.log(
      `seed ${s.seed} agent ${s.agent}: ep1=${s.first.toFixed(4)} ep${EPISODES}=${s.last.toFixed(4)} ` +
        `slope=${s.slope.toFixed(6)} → ${s.pass ? "PASS" : "FAIL"}`,
    );
  }
  const met = allStats.every((s) => s.pass);
  console.log(`G2-1 "loss decreases across 3 seeds": ${met ? "MET" : "NOT MET"} (${allStats.filter((s) => s.pass).length}/${allStats.length} pairs pass)`);
}

main();
