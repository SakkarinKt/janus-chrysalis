/**
 * Week-3 stack-validation spike (ADR-0002 decision 5, loop/GOAL.md priority 2):
 * measures tfjs-node CPU steps/sec for the RSSM cell at Arm-A dims. Run with
 * `node experiments/2026-07-21-week3-stack-spike/benchmark.ts`; writes
 * `summary.json` next to this file. Not a training run (no manifest.json —
 * synthetic throughput only, no environment steps or seeds), so it's exempt
 * from loop/GOAL.md's training-run manifest requirement.
 *
 * Only forward-pass rollout and a *single* differentiable training step are
 * measured. Multi-step gradient throughput can't be measured: chaining
 * step()+prior() across >=2 timesteps and differentiating crashes
 * tf.variableGrads on this stack (test/model/rssm.test.ts's "chaining ...
 * crashes tf.variableGrads" test; notes/adr-0002-js-ml-stack.md's
 * 2026-07-21 entry; tensorflow/tfjs#1529, #3550).
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import tf from "@tensorflow/tfjs-node";
import { RSSMCell } from "../../src/model/rssm.ts";
import { Action } from "../../src/env/types.ts";

// h≈256, z≈32 per PLAN.html's risk table ("Small dims (h≈256, z≈32)"). The
// 8x4 categorical/class split (vs. e.g. 32x1 or 4x8) isn't specified
// anywhere in the repo — picked as a DreamerV3-style multi-categorical
// latent scaled down to laptop size; flagged as an assumption in the
// 2026-07-21 stand-up.
const ARM_A_CONFIG = { deterministicSize: 256, latentCategoricals: 8, latentClasses: 4 };
// Not specified anywhere either — a plausible small-scale sequence batch.
const BATCH = 16;
const WARMUP_ITERS = 20;
const TIMED_ITERS = 200;

function fixedHard(batch: number, categoricals: number, classes: number): tf.Tensor3D {
  return tf
    .oneHot(tf.fill([batch * categoricals], 0, "int32"), classes)
    .toFloat()
    .reshape([batch, categoricals, classes]) as tf.Tensor3D;
}

function timeIters(iters: number, fn: () => void): { iters: number; seconds: number; stepsPerSec: number } {
  const start = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  const seconds = Number(process.hrtime.bigint() - start) / 1e9;
  return { iters, seconds, stepsPerSec: iters / seconds };
}

/** Forward-only imagination rollout: step() + prior(), no gradients. */
function benchmarkForwardRollout() {
  const rssm = new RSSMCell(ARM_A_CONFIG);
  const actions: Action[] = Array(BATCH).fill(Action.Up);
  let state = rssm.initialState(BATCH);

  const advance = () => {
    state = tf.tidy(() => {
      const deterministic = rssm.step(state, actions);
      const { sample } = rssm.prior(deterministic);
      return { deterministic, stochastic: sample };
    });
  };

  for (let i = 0; i < WARMUP_ITERS; i++) advance();
  return timeIters(TIMED_ITERS, advance);
}

/**
 * A single differentiable training step: step() + prior(), then
 * tf.variableGrads over the GRU/dense weights. Loss built from `probs` (see
 * test/model/rssm.test.ts's comment on why: `sample`'s STE forward value is
 * piecewise-constant and not representative of steps/sec on the smooth path
 * real losses would use).
 */
function benchmarkSingleStepGradient() {
  const rssm = new RSSMCell(ARM_A_CONFIG);
  const actions: Action[] = Array(BATCH).fill(Action.Up);
  const priorHard = fixedHard(BATCH, ARM_A_CONFIG.latentCategoricals, ARM_A_CONFIG.latentClasses);
  const stochasticSize = ARM_A_CONFIG.latentCategoricals * ARM_A_CONFIG.latentClasses;
  const lossWeights = tf.randomNormal([BATCH, stochasticSize]);

  function forward(): tf.Scalar {
    const state = rssm.initialState(BATCH);
    const deterministic = rssm.step(state, actions);
    const { probs } = rssm.prior(deterministic, priorHard);
    const flatProbs = probs.reshape([BATCH, stochasticSize]) as tf.Tensor2D;
    return tf.sum(tf.mul(flatProbs, lossWeights)) as tf.Scalar;
  }

  const before = new Set(Object.keys(tf.engine().registeredVariables));
  forward();
  const varList = Object.keys(tf.engine().registeredVariables)
    .filter((name) => !before.has(name))
    .map((name) => tf.engine().registeredVariables[name]);

  const step = () => {
    const { value } = tf.tidy(() => tf.variableGrads(forward, varList));
    value.dispose();
  };

  for (let i = 0; i < WARMUP_ITERS; i++) step();
  return timeIters(TIMED_ITERS, step);
}

const forwardRollout = benchmarkForwardRollout();
const singleStepGradient = benchmarkSingleStepGradient();

const summary = {
  date: "2026-07-21",
  purpose: "week-3 stack-validation spike: tfjs-node CPU steps/sec at Arm-A dims",
  backend: tf.getBackend(),
  config: ARM_A_CONFIG,
  batch: BATCH,
  warmupIters: WARMUP_ITERS,
  timedIters: TIMED_ITERS,
  forwardRollout,
  singleStepGradient,
  notMeasured: {
    multiStepGradient:
      "blocked — chaining step()+prior() across >=2 timesteps and differentiating crashes tf.variableGrads " +
      "on this stack (see test/model/rssm.test.ts, notes/adr-0002-js-ml-stack.md 2026-07-21 entry)",
  },
};

console.log(JSON.stringify(summary, null, 2));
const outPath = fileURLToPath(new URL("summary.json", import.meta.url));
writeFileSync(outPath, JSON.stringify(summary, null, 2) + "\n");

