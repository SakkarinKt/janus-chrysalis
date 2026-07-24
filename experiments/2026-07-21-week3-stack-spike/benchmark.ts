/**
 * Week-3 stack-validation spike (ADR-0002 decision 5, loop/GOAL.md priority 2):
 * measures tfjs-node CPU steps/sec for the RSSM cell at Arm-A dims. Run with
 * `node experiments/2026-07-21-week3-stack-spike/benchmark.ts`; writes
 * `summary.json` next to this file. Not a training run (no manifest.json —
 * synthetic throughput only, no environment steps or seeds), so it's exempt
 * from loop/GOAL.md's training-run manifest requirement.
 *
 * 2026-07-21: only forward-pass rollout and a *single* differentiable
 * training step could be measured — multi-step gradient throughput was
 * blocked because chaining step()+prior() across >=2 timesteps and
 * differentiating crashed tf.variableGrads on this stack.
 *
 * 2026-07-23: that block is gone (2026-07-22's direct-cell-call fix,
 * ADR-0002 decision 5 addendum — the kill criterion did not fire), so this
 * now also measures multi-step BPTT gradient throughput, chained across a
 * handful of sequence lengths. The lengths themselves aren't specified
 * anywhere in the repo (no truncated-BPTT chunk length is fixed by PLAN.html
 * or proposal 0001 — only the ≤200K-env-steps-per-run total budget is), so
 * they're picked to span "shorter than a plausible training chunk" to
 * "longer than one," same spirit as this file's existing BATCH assumption.
 *
 * Correction (2026-07-23, PR #24 review): the multi-step benchmark reports
 * `modelTimestepsPerSec` — chainLength * BATCH * gradient-steps/sec, i.e.
 * how many (batch row, timestep) pairs the stack actually differentiates
 * per second — not "environment-steps/sec". An earlier version of this
 * comment equated the two by computing chainLength * gradient-steps/sec,
 * dropping the batch term entirely. Converting to actual environment-
 * steps/sec needs a replay ratio (how many times each collected transition
 * is replayed through a gradient step, on average), which isn't fixed
 * anywhere in this repo yet — see notes/adr-0002-js-ml-stack.md §10 for the
 * worked example of how much that choice moves the wall-clock estimate
 * against proposal 0001's budget.
 *
 * Second correction (2026-07-24, PR #24 review, same-day comment): the
 * paragraph above originally called the earlier bug "silently assuming
 * replay ratio 1" — backwards. Dropping the batch term is arithmetically
 * `modelTimestepsPerSec / BATCH`, which is what environment-steps/sec
 * equals at replay ratio = BATCH (16), not ratio 1. So the pre-correction
 * ~70-83 figure was a coincidentally-valid point on the replay-ratio curve
 * at ratio 16 (see notes/adr-0002-js-ml-stack.md §10's bracket table) and a
 * conservative (~16x-too-low) read of the stack's raw throughput — not an
 * optimistic one assuming no replay at all.
 *
 * Also as of 2026-07-24: `tf.variableGrads`'s `grads` return value (one
 * tensor per differentiated variable) was never disposed in the timed loop
 * below — only `value` was. Fixed; each timed benchmark now also reports a
 * `tensorLeakCheck` (tensor count immediately before vs. after the timed
 * loop) so a future regression shows up in the numbers instead of silently
 * growing memory.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
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
// Truncated-BPTT chain lengths to sweep for the multi-step gradient
// benchmark — not specified anywhere in the repo (see the file doc comment).
// Fewer warmup/timed iters than the single-step benchmark: each iter does
// O(chainLength) work, and this is an order-of-magnitude throughput read,
// not a precision benchmark.
const CHAIN_LENGTHS = [2, 4, 8, 16, 32];
const MULTI_STEP_WARMUP_ITERS = 5;
const MULTI_STEP_TIMED_ITERS = 30;

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
    const { value, grads } = tf.tidy(() => tf.variableGrads(forward, varList));
    value.dispose();
    Object.values(grads).forEach((g) => g.dispose());
  };

  for (let i = 0; i < WARMUP_ITERS; i++) step();
  const tensorsBeforeTimed = tf.memory().numTensors;
  const timed = timeIters(TIMED_ITERS, step);
  const tensorsAfterTimed = tf.memory().numTensors;
  return {
    ...timed,
    tensorLeakCheck: {
      before: tensorsBeforeTimed,
      after: tensorsAfterTimed,
      leaked: tensorsAfterTimed - tensorsBeforeTimed,
    },
  };
}

/**
 * A differentiable training step chained across `chainLength` timesteps:
 * step()+prior() repeated, differentiating the summed deterministic state
 * back through the whole chain via a single tf.variableGrads call — the
 * truncated-BPTT shape a real training loop would use, at whatever chunk
 * length it picks. `priorHard` is fixed (same reason as the single-step
 * benchmark) so the forward pass is reproducible run to run; unlike the
 * single-step benchmark this doesn't need `probs`-vs-`sample` care, since
 * nothing here finite-differences it — only wall-clock is measured.
 */
function benchmarkMultiStepGradient(chainLength: number) {
  const rssm = new RSSMCell(ARM_A_CONFIG);
  const actions: Action[] = Array(BATCH).fill(Action.Up);
  const priorHard = fixedHard(BATCH, ARM_A_CONFIG.latentCategoricals, ARM_A_CONFIG.latentClasses);

  function forward(): tf.Scalar {
    let state = rssm.initialState(BATCH);
    for (let t = 0; t < chainLength; t++) {
      const deterministic = rssm.step(state, actions);
      const { sample } = rssm.prior(deterministic, priorHard);
      state = { deterministic, stochastic: sample };
    }
    return tf.sum(state.deterministic) as tf.Scalar;
  }

  const before = new Set(Object.keys(tf.engine().registeredVariables));
  forward();
  const varList = Object.keys(tf.engine().registeredVariables)
    .filter((name) => !before.has(name))
    .map((name) => tf.engine().registeredVariables[name]);

  const step = () => {
    const { value, grads } = tf.tidy(() => tf.variableGrads(forward, varList));
    value.dispose();
    Object.values(grads).forEach((g) => g.dispose());
  };

  for (let i = 0; i < MULTI_STEP_WARMUP_ITERS; i++) step();
  const tensorsBeforeTimed = tf.memory().numTensors;
  const timed = timeIters(MULTI_STEP_TIMED_ITERS, step);
  const tensorsAfterTimed = tf.memory().numTensors;
  // Actual (batch row, timestep) pairs differentiated per second — see the
  // 2026-07-23 correction in this file's doc comment for why this isn't
  // labeled "environment-steps/sec".
  const modelTimestepsPerSec = timed.stepsPerSec * chainLength * BATCH;
  return {
    chainLength,
    ...timed,
    modelTimestepsPerSec,
    tensorLeakCheck: {
      before: tensorsBeforeTimed,
      after: tensorsAfterTimed,
      leaked: tensorsAfterTimed - tensorsBeforeTimed,
    },
  };
}

const forwardRollout = benchmarkForwardRollout();
const singleStepGradient = benchmarkSingleStepGradient();
const multiStepGradient = CHAIN_LENGTHS.map(benchmarkMultiStepGradient);

const run = {
  // Filled in below once the existing history (if any) is loaded, so this
  // run gets a unique id even if it's not the first one today.
  runId: "",
  date: "",
  // Not knowable from inside the script — the commit this run ships in is
  // created after the script runs. Fill in by hand once committed (see the
  // 2026-07-21/2026-07-23 backfilled entries below for the pattern), or
  // leave null.
  commit: null,
  purpose: "week-3 stack-validation spike: tfjs-node CPU steps/sec at Arm-A dims",
  backend: tf.getBackend(),
  config: ARM_A_CONFIG,
  batch: BATCH,
  warmupIters: WARMUP_ITERS,
  timedIters: TIMED_ITERS,
  forwardRollout,
  singleStepGradient,
  multiStepGradient: {
    note:
      "2026-07-21 this was blocked (chaining step()+prior() across >=2 timesteps and differentiating " +
      "crashed tf.variableGrads); unblocked by 2026-07-22's direct-cell-call fix (ADR-0002 decision 5 " +
      "addendum) and measured here across a chain-length sweep (chunk length not specified anywhere " +
      "in the repo, see this file's doc comment). modelTimestepsPerSec = chainLength * BATCH * " +
      "stepsPerSec is the (batch row, timestep) throughput the stack actually computes; converting " +
      "to environment-steps/sec needs a replay ratio not yet fixed anywhere in this repo (2026-07-23 " +
      "correction, see this file's doc comment and notes/adr-0002-js-ml-stack.md §10)",
    multiStepWarmupIters: MULTI_STEP_WARMUP_ITERS,
    multiStepTimedIters: MULTI_STEP_TIMED_ITERS,
    byChainLength: multiStepGradient,
  },
};

// summary.json is append-only as of 2026-07-24 (PR #24 review: overwriting
// it twice on 2026-07-23 destroyed the evidence behind numbers already
// cited in notes/adr-0002-js-ml-stack.md and docs/proposals/0001-*.md).
// Each run gets a unique runId ("YYYY-MM-DD" plus a letter suffix if more
// than one run happens the same day) instead of replacing the file.
const outPath = fileURLToPath(new URL("summary.json", import.meta.url));
const existing = existsSync(outPath) ? JSON.parse(readFileSync(outPath, "utf8")) : { runs: [] };
const existingRuns = Array.isArray(existing.runs) ? existing.runs : [];

const date = new Date().toISOString().slice(0, 10);
const sameDayCount = existingRuns.filter((r) => r.date === date).length;
run.runId = date + String.fromCharCode("a".charCodeAt(0) + sameDayCount);
run.date = date;

const runs = [...existingRuns, run];
console.log(JSON.stringify(run, null, 2));
writeFileSync(outPath, JSON.stringify({ runs }, null, 2) + "\n");

